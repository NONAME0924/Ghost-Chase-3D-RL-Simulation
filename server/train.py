"""
Training script for the Ghost Chase RL agents.

Trains both Hunter and Prey agents adversarially using DQN.
"""

import argparse
import os
import sys
import numpy as np
from environment import ChaseEnv, NUM_ACTIONS
from agent import DQNAgent


def train(args):
    """Main training loop."""
    env = ChaseEnv()

    # State dimension = 12 (see environment.py observation)
    state_dim = 12

    hunter_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=NUM_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        target_update=args.target_update,
    )

    prey_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=NUM_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        target_update=args.target_update,
    )

    # Load existing models if available
    hunter_path = os.path.join(args.model_dir, "hunter.pth")
    prey_path = os.path.join(args.model_dir, "prey.pth")

    if args.resume:
        hunter_agent.load(hunter_path)
        prey_agent.load(prey_path)

    # Training stats
    total_captures = 0
    recent_rewards_h = []
    recent_rewards_p = []
    recent_captures = []

    print("=" * 60)
    print("  Ghost Chase RL Training")
    print("=" * 60)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  Epsilon:        {args.epsilon_start} → {args.epsilon_end}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Model Dir:      {args.model_dir}")
    print("=" * 60)

    for episode in range(1, args.episodes + 1):
        hunter_obs, prey_obs = env.reset()
        episode_reward_h = 0
        episode_reward_p = 0
        done = False

        while not done:
            # Select actions
            hunter_action = hunter_agent.select_action(hunter_obs, training=True)
            prey_action = prey_agent.select_action(prey_obs, training=True)

            # Step environment
            next_hunter_obs, next_prey_obs, reward_h, reward_p, terminated, truncated, info = env.step(
                hunter_action, prey_action
            )
            done = terminated or truncated

            # Store transitions
            hunter_agent.store_transition(hunter_obs, hunter_action, reward_h, next_hunter_obs, done)
            prey_agent.store_transition(prey_obs, prey_action, reward_p, next_prey_obs, done)

            # Learn
            hunter_agent.learn()
            prey_agent.learn()

            # Update
            hunter_obs = next_hunter_obs
            prey_obs = next_prey_obs
            episode_reward_h += reward_h
            episode_reward_p += reward_p

        # Episode end
        hunter_agent.decay_epsilon()
        prey_agent.decay_epsilon()

        captured = info.get('captured', False)
        if captured:
            total_captures += 1

        recent_rewards_h.append(episode_reward_h)
        recent_rewards_p.append(episode_reward_p)
        recent_captures.append(1 if captured else 0)

        # Logging
        if episode % args.log_interval == 0:
            avg_h = np.mean(recent_rewards_h[-args.log_interval:])
            avg_p = np.mean(recent_rewards_p[-args.log_interval:])
            cap_rate = np.mean(recent_captures[-args.log_interval:]) * 100

            print(
                f"Episode {episode:5d} | "
                f"Hunter Reward: {avg_h:7.2f} | "
                f"Prey Reward: {avg_p:7.2f} | "
                f"Capture Rate: {cap_rate:5.1f}% | "
                f"Epsilon: {hunter_agent.epsilon:.3f} | "
                f"Total Captures: {total_captures}"
            )

        # Save models periodically
        if episode % args.save_interval == 0:
            hunter_agent.save(hunter_path)
            prey_agent.save(prey_path)

    # Final save
    hunter_agent.save(hunter_path)
    prey_agent.save(prey_path)

    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Total episodes:  {args.episodes}")
    print(f"  Total captures:  {total_captures}")
    print(f"  Capture rate:    {total_captures / args.episodes * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ghost Chase RL agents")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--target-update", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N episodes")
    parser.add_argument("--save-interval", type=int, default=500, help="Save every N episodes")
    parser.add_argument("--resume", action="store_true", help="Resume from saved models")

    args = parser.parse_args()
    train(args)
