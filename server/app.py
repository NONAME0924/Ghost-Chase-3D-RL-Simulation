"""
Flask + SocketIO server for the Ghost Chase game.

Unified server that handles BOTH training AND visualization.
Training runs in a background thread and streams every step
to the Three.js frontend via WebSocket in real-time.

Modes:
  - TRAIN: Trains agents from scratch, visualizing each step live
  - DEMO:  Loads pre-trained models and runs inference only
"""

import os
import sys
import time
import math
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from environment import ChaseEnv, NUM_ACTIONS
from agent import DQNAgent

# === Configuration ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "ghost-chase-dev-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# === Global game state ===
env = ChaseEnv()
hunter_agent = DQNAgent(state_dim=12, action_dim=NUM_ACTIONS)
prey_agent = DQNAgent(state_dim=12, action_dim=NUM_ACTIONS)

game_running = False
paused = False
current_mode = "train"  # "train" or "demo"

# Training settings (can be changed by frontend)
train_settings = {
    "speed": 3,          # 1=slow(30fps) 2=medium(60fps) 3=fast(no-render-delay) 4=turbo(skip frames)
    "max_episodes": 5000,
    "visualize_every": 1,  # In turbo mode, only render every Nth episode
}

# Training stats
train_stats = {
    "episode": 0,
    "total_captures": 0,
    "recent_rewards_h": [],
    "recent_rewards_p": [],
    "recent_captures": [],
    "epsilon": 1.0,
    "loss_h": 0.0,
    "loss_p": 0.0,
}


def get_tick_interval():
    """Get frame delay based on speed setting."""
    speed = train_settings["speed"]
    if speed == 1:
        return 1.0 / 15   # Slow: 15 FPS
    elif speed == 2:
        return 1.0 / 30   # Medium: 30 FPS
    elif speed == 3:
        return 1.0 / 120  # Fast: 120 FPS
    else:
        return 0           # Turbo: no delay


def load_models():
    """Load trained models if available."""
    hunter_path = os.path.join(MODEL_DIR, "hunter.pth")
    prey_path = os.path.join(MODEL_DIR, "prey.pth")

    h_loaded = hunter_agent.load(hunter_path)
    p_loaded = prey_agent.load(prey_path)

    if not h_loaded or not p_loaded:
        print("  No pre-trained models found. Will train from scratch.")
        return False

    hunter_agent.epsilon = 0.0
    prey_agent.epsilon = 0.0
    return True


def training_loop():
    """
    Main training + visualization loop.
    Trains both agents and streams every step to the frontend.
    """
    global game_running, paused, train_stats

    game_running = True
    max_episodes = train_settings["max_episodes"]

    print(f"\n  Starting live training for {max_episodes} episodes...")
    print(f"  Speed: {train_settings['speed']} | Visualize: every episode\n")

    for episode in range(1, max_episodes + 1):
        if not game_running:
            break

        hunter_obs, prey_obs = env.reset()
        train_stats["episode"] = episode
        episode_reward_h = 0.0
        episode_reward_p = 0.0
        done = False
        step_count = 0

        # Emit episode start
        state = env.get_state()
        state["episode"] = episode
        state["captured"] = False
        socketio.emit("game_reset", state)

        while not done and game_running:
            # Pause handling
            while paused and game_running:
                socketio.sleep(0.1)

            if not game_running:
                break

            step_count += 1

            # Select actions (with exploration during training)
            hunter_action = hunter_agent.select_action(hunter_obs, training=True)
            prey_action = prey_agent.select_action(prey_obs, training=True)

            # Step environment
            next_h_obs, next_p_obs, reward_h, reward_p, terminated, truncated, info = env.step(
                hunter_action, prey_action
            )
            done = terminated or truncated

            # Store transitions
            hunter_agent.store_transition(hunter_obs, hunter_action, reward_h, next_h_obs, done)
            prey_agent.store_transition(prey_obs, prey_action, reward_p, next_p_obs, done)

            # Learn
            loss_h = hunter_agent.learn()
            loss_p = prey_agent.learn()
            train_stats["loss_h"] = loss_h
            train_stats["loss_p"] = loss_p

            # Update observations
            hunter_obs = next_h_obs
            prey_obs = next_p_obs
            episode_reward_h += reward_h
            episode_reward_p += reward_p

            # --- Stream to frontend ---
            speed = train_settings["speed"]
            should_render = (speed <= 3) or (speed == 4 and step_count % 5 == 0)

            if should_render:
                state = env.get_state()
                state["episode"] = episode
                state["captured"] = info.get("captured", False)
                socketio.emit("game_state", state)

                # Frame delay
                tick = get_tick_interval()
                if tick > 0:
                    socketio.sleep(tick)

        # --- Episode done ---
        captured = info.get("captured", False)
        if captured:
            train_stats["total_captures"] += 1
            socketio.emit("captured", {"episode": episode})
            # Brief pause on capture for visual feedback
            if train_settings["speed"] <= 2:
                socketio.sleep(0.5)

        # Decay epsilon
        hunter_agent.decay_epsilon()
        prey_agent.decay_epsilon()
        train_stats["epsilon"] = hunter_agent.epsilon

        # Track stats
        train_stats["recent_rewards_h"].append(episode_reward_h)
        train_stats["recent_rewards_p"].append(episode_reward_p)
        train_stats["recent_captures"].append(1 if captured else 0)

        # Keep only last 100 for averaging
        if len(train_stats["recent_rewards_h"]) > 100:
            train_stats["recent_rewards_h"] = train_stats["recent_rewards_h"][-100:]
            train_stats["recent_rewards_p"] = train_stats["recent_rewards_p"][-100:]
            train_stats["recent_captures"] = train_stats["recent_captures"][-100:]

        # Compute averages
        avg_h = np.mean(train_stats["recent_rewards_h"][-50:]) if train_stats["recent_rewards_h"] else 0
        avg_p = np.mean(train_stats["recent_rewards_p"][-50:]) if train_stats["recent_rewards_p"] else 0
        cap_rate = np.mean(train_stats["recent_captures"][-50:]) * 100 if train_stats["recent_captures"] else 0

        # Emit training stats to frontend
        socketio.emit("train_stats", {
            "episode": episode,
            "max_episodes": max_episodes,
            "epsilon": round(hunter_agent.epsilon, 4),
            "avg_reward_h": round(avg_h, 2),
            "avg_reward_p": round(avg_p, 2),
            "capture_rate": round(cap_rate, 1),
            "total_captures": train_stats["total_captures"],
            "loss_h": round(loss_h, 4) if loss_h else 0,
            "loss_p": round(loss_p, 4) if loss_p else 0,
            "steps": step_count,
        })

        # Console log every 50 episodes
        if episode % 50 == 0:
            print(
                f"  Episode {episode:5d} | "
                f"Cap Rate: {cap_rate:5.1f}% | "
                f"ε: {hunter_agent.epsilon:.3f} | "
                f"Avg R(H): {avg_h:7.2f} | "
                f"Total Caps: {train_stats['total_captures']}"
            )

        # Auto-save every 200 episodes
        if episode % 200 == 0:
            hunter_agent.save(os.path.join(MODEL_DIR, "hunter.pth"))
            prey_agent.save(os.path.join(MODEL_DIR, "prey.pth"))

    # Final save
    hunter_agent.save(os.path.join(MODEL_DIR, "hunter.pth"))
    prey_agent.save(os.path.join(MODEL_DIR, "prey.pth"))

    socketio.emit("train_complete", {
        "episodes": train_stats["episode"],
        "total_captures": train_stats["total_captures"],
    })
    print(f"\n  Training complete! {train_stats['episode']} episodes, {train_stats['total_captures']} captures")

    # Switch to demo mode
    demo_loop()


def demo_loop():
    """Run inference with trained models (no learning)."""
    global game_running, paused

    game_running = True
    hunter_agent.epsilon = 0.0
    prey_agent.epsilon = 0.0
    episode_count = 0

    print("  Switched to DEMO mode (inference only)")

    while game_running:
        episode_count += 1
        hunter_obs, prey_obs = env.reset()

        state = env.get_state()
        state["episode"] = episode_count
        state["captured"] = False
        socketio.emit("game_reset", state)

        done = False
        while not done and game_running:
            while paused and game_running:
                socketio.sleep(0.1)

            hunter_action = hunter_agent.select_action(hunter_obs, training=False)
            prey_action = prey_agent.select_action(prey_obs, training=False)

            hunter_obs, prey_obs, _, _, terminated, truncated, info = env.step(hunter_action, prey_action)
            done = terminated or truncated

            state = env.get_state()
            state["episode"] = episode_count
            state["captured"] = info.get("captured", False)
            socketio.emit("game_state", state)

            socketio.sleep(1.0 / 30)

        if info.get("captured", False):
            socketio.emit("captured", {"episode": episode_count})
            socketio.sleep(1.0)


# === Routes ===
@app.route("/")
def index():
    return send_from_directory(CLIENT_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(CLIENT_DIR, path)


# === SocketIO Events ===
@socketio.on("connect")
def on_connect():
    global game_running
    print("  Client connected")
    if not game_running:
        if current_mode == "train":
            socketio.start_background_task(training_loop)
        else:
            socketio.start_background_task(demo_loop)


@socketio.on("disconnect")
def on_disconnect():
    print("  Client disconnected")


@socketio.on("pause")
def on_pause():
    global paused
    paused = True


@socketio.on("resume")
def on_resume():
    global paused
    paused = False


@socketio.on("set_speed")
def on_set_speed(data):
    speed = data.get("speed", 3)
    train_settings["speed"] = max(1, min(4, speed))
    print(f"  Speed set to: {train_settings['speed']}")


@socketio.on("reset")
def on_reset(*args):
    hunter_obs, prey_obs = env.reset()
    state = env.get_state()
    state["episode"] = train_stats["episode"]
    state["captured"] = False
    socketio.emit("game_reset", state)


# === Main ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ghost Chase - 3D RL Server")
    parser.add_argument("--mode", choices=["train", "demo"], default="train",
                        help="'train' = live training, 'demo' = load & run pre-trained models")
    parser.add_argument("--episodes", type=int, default=5000, help="Max training episodes")
    parser.add_argument("--speed", type=int, default=3, choices=[1, 2, 3, 4],
                        help="1=slow 2=medium 3=fast 4=turbo")
    parser.add_argument("--port", type=int, default=5000, help="Server port")

    args = parser.parse_args()

    current_mode = args.mode
    train_settings["max_episodes"] = args.episodes
    train_settings["speed"] = args.speed

    print("=" * 50)
    print("  👻 Ghost Chase — 3D RL Simulation")
    print("=" * 50)
    print(f"  Mode:     {current_mode.upper()}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Speed:    {args.speed}")
    print(f"  Port:     {args.port}")
    print("=" * 50)

    if current_mode == "demo":
        loaded = load_models()
        if not loaded:
            print("\n  No models found! Switching to TRAIN mode.\n")
            current_mode = "train"

    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    print("=" * 50)

    socketio.run(app, host="127.0.0.1", port=args.port, debug=False)
