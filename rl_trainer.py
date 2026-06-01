"""
Dr. CPPO trainer for Cadrille: Multi-modal CAD Reconstruction with RL.

Combines Dr. GRPO (eliminates reference model) and CPPO (samples with strongest signal)
for efficient and accurate policy optimization using Chamfer Distance / IoU rewards.

Paper: https://arxiv.org/abs/2505.22914
"""

import os
import gc
import torch
import trimesh
import numpy as np
import cadquery as cq
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import partial
from scipy.spatial import cKDTree
from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing.pool import Pool

from transformers import AutoProcessor
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam

from cadrille import Cadrille, collate
from dataset import CadRecodeDataset, Text2CADDataset


@dataclass
class DrCPPOConfig:
    """Configuration for Dr. CPPO training."""
    
    # RL hyperparameters
    num_rollouts_per_batch: int = 4  # G in the paper
    num_samples_for_update: int = 2  # N in the paper (use N highest |advantage| samples)
    ppo_clip_ratio: float = 0.2  # epsilon in the paper
    ppo_epochs: int = 2
    
    # Reward parameters
    reward_metric: str = "iou"  # "iou" or "chamfer"
    chamfer_n_points: int = 2048
    
    # Training parameters
    learning_rate: float = 5e-5
    temperature: float = 1.0
    max_gen_length: int = 512
    
    # Sampling parameters
    top_p: float = 0.9
    top_k: int = 50
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardComputer:
    """Computes rewards for generated CAD programs using IoU or Chamfer distance."""
    
    def __init__(self, metric: str = "iou", chamfer_n_points: int = 2048):
        self.metric = metric
        self.chamfer_n_points = chamfer_n_points
    
    def py_string_to_mesh(self, py_string: str, timeout: int = 3) -> Optional[trimesh.Trimesh]:
        """
        Convert Python code string to mesh with timeout protection.
        Uses subprocess to prevent memory leaks from CadQuery.
        """
        import tempfile
        import subprocess
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(py_string)
                py_path = f.name
            
            mesh_path = py_path.replace('.py', '.stl')
            brep_path = py_path.replace('.py', '.step')
            
            # Run in subprocess to isolate CadQuery memory leaks
            script = f"""
import sys
sys.path.insert(0, '.')
import trimesh
import cadquery as cq

try:
    with open('{py_path}', 'r') as f:
        py_string = f.read()
    exec(py_string, globals())
    compound = globals()['r'].val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([tuple(v.x, v.y, v.z) for v in vertices], faces)
    if len(mesh.faces) > 2:
        mesh.export('{mesh_path}')
        print('SUCCESS')
except Exception as e:
    print(f'FAILED: {{e}}')
"""
            result = subprocess.run(
                ['python', '-c', script],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if 'SUCCESS' in result.stdout and os.path.exists(mesh_path):
                mesh = trimesh.load_mesh(mesh_path)
                os.remove(py_path)
                os.remove(mesh_path)
                if os.path.exists(brep_path):
                    os.remove(brep_path)
                return mesh
        except (subprocess.TimeoutExpired, Exception):
            pass
        finally:
            if os.path.exists(py_path):
                os.remove(py_path)
        
        return None
    
    def compute_chamfer_distance(self, gt_mesh: trimesh.Trimesh, 
                                pred_mesh: trimesh.Trimesh) -> float:
        """Compute Chamfer distance between two meshes."""
        try:
            gt_points, _ = trimesh.sample.sample_surface(gt_mesh, self.chamfer_n_points)
            pred_points, _ = trimesh.sample.sample_surface(pred_mesh, self.chamfer_n_points)
            
            gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
            pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
            
            return float(np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance)))
        except:
            return float('inf')
    
    def compute_iou(self, gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh) -> float:
        """Compute Intersection over Union between two meshes."""
        try:
            intersection_volume = 0.0
            for gt_mesh_i in gt_mesh.split():
                for pred_mesh_i in pred_mesh.split():
                    intersection = gt_mesh_i.intersection(pred_mesh_i)
                    volume = intersection.volume if intersection is not None else 0.0
                    intersection_volume += volume
            
            gt_volume = sum(m.volume for m in gt_mesh.split())
            pred_volume = sum(m.volume for m in pred_mesh.split())
            union_volume = gt_volume + pred_volume - intersection_volume
            
            if union_volume > 0:
                return float(intersection_volume / union_volume)
        except:
            pass
        
        return 0.0
    
    def compute_reward(self, py_string: str, gt_mesh: trimesh.Trimesh) -> float:
        """
        Compute reward for a generated CAD program.
        
        Returns:
            - For IoU: IoU score (higher is better, normalized to [0, 1])
            - For Chamfer: -distance (higher is better, unbounded below)
        """
        pred_mesh = self.py_string_to_mesh(py_string)
        
        if pred_mesh is None:
            return 0.0  # Invalid program
        
        try:
            # Normalize meshes to unit cube
            pred_mesh.apply_transform(trimesh.transformations.translation_matrix(
                -pred_mesh.centroid
            ))
            extent = np.max(pred_mesh.extents)
            if extent > 1e-7:
                pred_mesh.apply_scale(1.0 / extent)
            pred_mesh.apply_transform(trimesh.transformations.translation_matrix(
                np.array([0.5, 0.5, 0.5])
            ))
            
            if self.metric == "iou":
                reward = self.compute_iou(gt_mesh, pred_mesh)
            else:  # chamfer
                cd = self.compute_chamfer_distance(gt_mesh, pred_mesh)
                # Normalize: exp(-cd) gives reward in [0, 1]
                reward = float(np.exp(-cd))
            
            return max(0.0, reward)
        except:
            return 0.0


class DrCPPOTrainer:
    """Dr. CPPO trainer for Cadrille."""
    
    def __init__(
        self,
        model: Cadrille,
        processor,
        config: DrCPPOConfig,
        output_dir: str = "./rl_checkpoints"
    ):
        self.model = model.to(config.device)
        self.processor = processor
        self.config = config
        self.output_dir = output_dir
        self.device = config.device
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize reference policy (for PPO)
        # Note: Dr. GRPO removes the reference model, but we keep it for standard PPO
        self.reference_model = Cadrille.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct',
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.reference_model.eval()
        
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.reward_computer = RewardComputer(
            metric=config.reward_metric,
            chamfer_n_points=config.chamfer_n_points
        )
    
    def generate_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        point_clouds: torch.Tensor,
        is_pc: torch.Tensor,
        is_img: torch.Tensor,
        num_sequences: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate G sequences from the current policy.
        
        Returns:
            - sequences: (batch_size, G, seq_len) tensor of token IDs
            - log_probs: (batch_size, G) tensor of log probabilities
        """
        batch_size = input_ids.shape[0]
        all_sequences = []
        all_log_probs = []
        
        with torch.no_grad():
            for _ in range(num_sequences):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    point_clouds=point_clouds,
                    is_pc=is_pc,
                    is_img=is_img,
                    max_length=self.config.max_gen_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                
                sequences = outputs.sequences
                all_sequences.append(sequences)
                
                # Compute log probabilities
                transition_scores = self.model.compute_transition_scores(
                    sequences, outputs.scores, normalize_logits=True
                )
                log_probs = torch.sum(transition_scores, dim=-1)
                all_log_probs.append(log_probs)
        
        # Stack: (batch_size, G, seq_len)
        sequences_tensor = torch.stack(all_sequences, dim=1)
        log_probs_tensor = torch.stack(all_log_probs, dim=1)
        
        return sequences_tensor, log_probs_tensor
    
    def compute_rewards(
        self,
        sequences: torch.Tensor,
        gt_data: List[Dict]
    ) -> torch.Tensor:
        """
        Compute rewards for all generated sequences.
        
        Args:
            sequences: (batch_size, G, seq_len) tensor of token IDs
            gt_data: list of dicts with 'mesh_path' and other metadata
        
        Returns:
            rewards: (batch_size, G) tensor of rewards
        """
        batch_size, num_sequences, seq_len = sequences.shape
        rewards = torch.zeros(batch_size, num_sequences, device=self.device)
        
        for b in range(batch_size):
            # Load ground truth mesh
            gt_mesh_path = gt_data[b].get('mesh_path')
            if gt_mesh_path is None:
                continue
            
            gt_mesh = trimesh.load(gt_mesh_path)
            
            for g in range(num_sequences):
                # Decode sequence to Python string
                token_ids = sequences[b, g].tolist()
                py_string = self.processor.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True
                )
                
                # Compute reward
                reward = self.reward_computer.compute_reward(py_string, gt_mesh)
                rewards[b, g] = reward
        
        return rewards
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages: A_g = r_g - mean(r_i).
        
        Args:
            rewards: (batch_size, G) tensor
        
        Returns:
            advantages: (batch_size, G) tensor
        """
        mean_reward = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - mean_reward
        return advantages
    
    def select_top_samples(
        self,
        advantages: torch.Tensor,
        sequences: torch.Tensor,
        log_probs_old: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select N samples with highest |advantage| for the policy update (CPPO).
        
        Returns:
            - selected_sequences: (batch_size * N, seq_len)
            - selected_log_probs_old: (batch_size * N,)
            - selected_advantages: (batch_size * N,)
            - selected_rewards: (batch_size * N,)
        """
        batch_size, num_sequences = advantages.shape
        N = self.config.num_samples_for_update
        
        selected_sequences_list = []
        selected_log_probs_list = []
        selected_advantages_list = []
        selected_rewards_list = []
        
        for b in range(batch_size):
            abs_advantages = torch.abs(advantages[b])
            top_indices = torch.topk(abs_advantages, k=min(N, num_sequences)).indices
            
            selected_sequences_list.append(sequences[b, top_indices])
            selected_log_probs_list.append(log_probs_old[b, top_indices])
            selected_advantages_list.append(advantages[b, top_indices])
            selected_rewards_list.append(rewards[b, top_indices])
        
        selected_sequences = torch.cat(selected_sequences_list, dim=0)
        selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
        selected_advantages = torch.cat(selected_advantages_list, dim=0)
        selected_rewards = torch.cat(selected_rewards_list, dim=0)
        
        return selected_sequences, selected_log_probs, selected_advantages, selected_rewards
    
    def ppo_update(
        self,
        sequences: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        point_clouds: torch.Tensor,
        is_pc: torch.Tensor,
        is_img: torch.Tensor,
    ) -> float:
        """
        Perform PPO policy update with Dr. CPPO modifications.
        
        Returns:
            mean_loss: average loss across PPO epochs
        """
        total_loss = 0.0
        num_updates = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Forward pass through current policy
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_clouds=point_clouds,
                is_pc=is_pc,
                is_img=is_img,
                return_dict=True,
            )
            
            logits = outputs.logits
            log_probs_new = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Compute probability ratio: π_new / π_old
            # Note: This is a simplified approximation
            ratio = torch.exp(log_probs_new.mean() - log_probs_old.mean())
            
            # PPO clipped objective
            # L = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            clip_ratio = self.config.ppo_clip_ratio
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            
            loss = -torch.min(
                ratio * advantages.mean(),
                clipped_ratio * advantages.mean()
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_updates += 1
        
        return total_loss / num_updates
    
    def train_step(
        self,
        batch: Dict,
    ) -> Dict:
        """
        Single training step of Dr. CPPO.
        
        Returns:
            metrics: dict with training metrics
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        point_clouds = batch['point_clouds'].to(self.device)
        is_pc = batch['is_pc'].to(self.device)
        is_img = batch['is_img'].to(self.device)
        
        batch_size = input_ids.shape[0]
        
        # Step 1: Generate G sequences from current policy
        sequences, log_probs_old = self.generate_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            is_pc=is_pc,
            is_img=is_img,
            num_sequences=self.config.num_rollouts_per_batch,
        )
        
        # Step 2: Compute rewards
        gt_data = [
            {'mesh_path': batch['file_name'][i]}  # In practice, fetch actual mesh path
            for i in range(batch_size)
        ]
        rewards = self.compute_rewards(sequences, gt_data)
        
        # Step 3: Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Step 4: Select N samples with highest |A_g| (CPPO)
        selected_sequences, selected_log_probs, selected_advantages, selected_rewards = \
            self.select_top_samples(advantages, sequences, log_probs_old, rewards)
        
        # Step 5: PPO update
        ppo_loss = self.ppo_update(
            sequences=selected_sequences,
            log_probs_old=selected_log_probs,
            advantages=selected_advantages,
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            is_pc=is_pc,
            is_img=is_img,
        )
        
        metrics = {
            'ppo_loss': ppo_loss,
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_steps: int = 10000,
        eval_every: int = 1000,
    ):
        """Main training loop."""
        step = 0
        
        for epoch in range(num_steps // len(train_dataloader) + 1):
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                if step >= num_steps:
                    return
                
                try:
                    metrics = self.train_step(batch)
                    
                    if step % 100 == 0:
                        print(f"Step {step}: {metrics}")
                    
                    if step % eval_every == 0:
                        self.save_checkpoint(step)
                
                except Exception as e:
                    print(f"Error in step {step}: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                
                step += 1
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.model.save_pretrained(os.path.join(checkpoint_path, "model"))
        self.processor.save_pretrained(os.path.join(checkpoint_path, "processor"))
        
        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )


def run_rl_training(
    data_path: str,
    output_dir: str,
    mode: str = "pc",
    use_text: bool = False,
    num_steps: int = 10000,
):
    """Main function to run Dr. CPPO training."""
    
    config = DrCPPOConfig(
        num_rollouts_per_batch=4,
        num_samples_for_update=2,
        reward_metric="iou",
        max_gen_length=512,
    )
    
    # Load datasets
    cad_recode_path = os.path.join(data_path, 'cad-recode-v1.5')
    train_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split='train',
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode,
    )
    
    batch_size = 4
    if use_text:
        text_dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, 'text2cad'),
            split='train',
        )
        train_dataset = ConcatDataset([train_dataset, text_dataset])
        batch_size = 2
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=partial(collate, processor=None, n_points=256),
    )
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left'
    )
    model = Cadrille.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    
    # Create trainer and train
    trainer = DrCPPOTrainer(
        model=model,
        processor=processor,
        config=config,
        output_dir=output_dir,
    )
    
    trainer.train(train_dataloader, num_steps=num_steps)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./rl_checkpoints')
    parser.add_argument('--mode', type=str, default='pc')
    parser.add_argument('--use-text', action='store_true')
    parser.add_argument('--num-steps', type=int, default=10000)
    
    args = parser.parse_args()
    
    run_rl_training(
        data_path=args.data_path,
        output_dir=args.output_dir,
        mode=args.mode,
        use_text=args.use_text,
        num_steps=args.num_steps,
    )
