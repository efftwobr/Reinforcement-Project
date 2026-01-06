import torch
import numpy as np
import random
from Wordle_Env import create_env
import Wordle_Env
from Models.Hierarchical_PG import HierarchicalAgent, train_once


class CurriculumTrainer:
    """Curriculum learning: progressively scale from small to large word sets"""
    
    def __init__(self, device, seed=42):
        self.device = device
        self.seed = seed
        
        # Curriculum stages: (env_name, episodes, win_rate_threshold)
        self.curriculum = [
            ("WordleEnv10", 20000, 70.0),    # Start small
            ("WordleEnv100", 50000, 40.0),   # Scale up
            ("WordleEnv1000", 100000, 20.0), # Final scale
        ]
        
        # Hyperparameters that scale with difficulty
        self.base_config = {
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "record": 1000,
            "num_options": 5,
            "entropy_coef": 0.02,
            "grad_clip": 1.0,
            "dropout_rate": 0.3,
            "use_batchnorm": True,
        }
        
    def get_scaled_config(self, stage_idx, output_dim):
        """Adjust hyperparameters based on curriculum stage"""
        config = self.base_config.copy()
        
        # Scale hidden layer size with output dimension
        if output_dim <= 10:
            config["hidden_layer_size"] = (64, 128, 64)
        elif output_dim <= 100:
            config["hidden_layer_size"] = (128, 256, 128)
        else:  # 1000+
            config["hidden_layer_size"] = (256, 512, 256)
        
        # Increase warmup for larger action spaces
        config["warmup_episodes"] = min(5000, output_dim * 10)
        
        # Scale entropy coefficient (more exploration for larger spaces)
        config["entropy_coef"] = 0.01 + (0.02 * stage_idx)
        
        return config
    
    def train_stage(self, env_name, episodes, win_threshold, agent=None, stage_idx=0):
        """Train on a single curriculum stage"""
        print(f"\n{'='*70}")
        print(f"CURRICULUM STAGE {stage_idx + 1}: {env_name}")
        print(f"Target Episodes: {episodes}, Win Rate Threshold: {win_threshold}%")
        print(f"{'='*70}\n")
        
        # Create environment
        env, input_dim, output_dim = create_env(env_name)
        print(f"State Dim: {input_dim}, Action Dim: {output_dim}")
        
        # Get scaled configuration
        config = self.get_scaled_config(stage_idx, output_dim)
        print(f"Hidden layers: {config['hidden_layer_size']}")
        print(f"Warmup episodes: {config['warmup_episodes']}")
        print(f"Entropy coefficient: {config['entropy_coef']}")
        
        # Create or transfer agent
        if agent is None:
            # First stage - create new agent
            agent = HierarchicalAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                device=self.device,
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                record=config["record"],
                hidden_layer_size=config["hidden_layer_size"],
                num_options=config["num_options"],
                entropy_coef=config["entropy_coef"],
                grad_clip=config["grad_clip"],
                use_lr_scheduler=True,
                dropout_rate=config["dropout_rate"],
                use_batchnorm=config["use_batchnorm"],
                warmup_episodes=config["warmup_episodes"],
            )
        else:
            # Transfer learning - create new agent with transferred weights
            print("Transferring knowledge from previous stage...")
            old_agent = agent
            agent = HierarchicalAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                device=self.device,
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                record=config["record"],
                hidden_layer_size=config["hidden_layer_size"],
                num_options=config["num_options"],
                entropy_coef=config["entropy_coef"],
                grad_clip=config["grad_clip"],
                use_lr_scheduler=True,
                dropout_rate=config["dropout_rate"],
                use_batchnorm=config["use_batchnorm"],
                warmup_episodes=config["warmup_episodes"],
            )
            
            # Transfer shared weights (baseline and high-level policy)
            self._transfer_weights(old_agent, agent)
            
            # Reset epsilon for new stage exploration
            agent.epsilon = 0.5  # Start with moderate exploration
        
        # Training loop
        best_win_rate = 0.0
        patience = 10  # Early stopping patience
        patience_counter = 0
        
        for episode in range(1, episodes + 1):
            train_once(agent, env)
            
            if episode % config["record"] == 0:
                win_rate = np.mean(agent.recent_wins) * 100
                avg_reward = np.mean(agent.recent_rewards)
                
                print(f"Episode {episode:7d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Win Rate: {win_rate:5.2f}% | "
                      f"Epsilon: {agent.epsilon:.4f}")
                
                # Update schedulers
                agent.update_schedulers(win_rate)
                
                # Track best performance
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent.save_checkpoint(f"best_{env_name}.pt")
                    print(f"  → New best! Saved checkpoint.")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping if we exceed threshold
                if win_rate >= win_threshold:
                    print(f"\n✓ Reached target win rate of {win_threshold}%!")
                    print(f"  Moving to next stage...")
                    break
                
                # Early stopping if no improvement
                if patience_counter >= patience:
                    print(f"\n⚠ No improvement for {patience * config['record']} episodes")
                    if win_rate >= win_threshold * 0.8:  # 80% of target
                        print(f"  Win rate is close enough ({win_rate:.2f}%), moving on...")
                        break
        
        print(f"\nStage completed! Best win rate: {best_win_rate:.2f}%")
        
        # Test the agent
        print("\nTesting agent on this stage:")
        Wordle_Env.test_agent(env, agent)
        
        env.close()
        return agent, best_win_rate
    
    def _transfer_weights(self, old_agent, new_agent):
        """Transfer compatible weights from old agent to new agent"""
        try:
            # Transfer baseline network (state representation doesn't change)
            new_agent.baseline_net.load_state_dict(old_agent.baseline_net.state_dict())
            print("  ✓ Transferred baseline network")
        except:
            print("  ✗ Could not transfer baseline (architecture mismatch)")
        
        try:
            # Transfer high-level policy (state representation doesn't change)
            new_agent.highlevel_net.load_state_dict(old_agent.highlevel_net.state_dict())
            print("  ✓ Transferred high-level policy")
        except:
            print("  ✗ Could not transfer high-level policy (architecture mismatch)")
        
        # For policy network, we can try to transfer the early layers
        try:
            old_state = old_agent.policy_net.state_dict()
            new_state = new_agent.policy_net.state_dict()
            
            # Transfer shared layers (before output layer)
            transferred = 0
            for key in old_state.keys():
                if 'output_layer' not in key and key in new_state:
                    if old_state[key].shape == new_state[key].shape:
                        new_state[key] = old_state[key]
                        transferred += 1
            
            new_agent.policy_net.load_state_dict(new_state)
            print(f"  ✓ Transferred {transferred} policy layers")
        except Exception as e:
            print(f"  ✗ Could not transfer policy layers: {e}")
    
    def run(self):
        """Run the full curriculum"""
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print("="*70)
        print("CURRICULUM LEARNING FOR WORDLE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        print(f"\nCurriculum stages:")
        for i, (env_name, eps, threshold) in enumerate(self.curriculum):
            print(f"  {i+1}. {env_name:20s} - {eps:6d} episodes, {threshold:5.1f}% target")
        
        agent = None
        results = []
        
        for stage_idx, (env_name, episodes, threshold) in enumerate(self.curriculum):
            agent, best_win_rate = self.train_stage(
                env_name, episodes, threshold, agent, stage_idx
            )
            results.append((env_name, best_win_rate))
        
        # Final summary
        print("\n" + "="*70)
        print("CURRICULUM LEARNING COMPLETED")
        print("="*70)
        print("\nResults summary:")
        for env_name, win_rate in results:
            print(f"  {env_name:20s}: {win_rate:5.2f}% best win rate")
        
        return agent


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = CurriculumTrainer(device=device, seed=42)
    final_agent = trainer.run()
    
    print("\n" + "="*70)
    print("Training complete! Final agent is ready for deployment.")
    print("="*70)
