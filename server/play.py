"""
Ghost Chase — Play Mode Server
Allows manual control of either the Hunter or the Prey against pre-trained AI agents.
"""

import os
import time
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from environment import ChaseEnv, NUM_ACTIONS
from agent import DQNAgent

# === Configuration ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")
app.config["SECRET_KEY"] = "ghost-chase-play-key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# === Global state ===
env = ChaseEnv()
hunter_agent = DQNAgent(state_dim=14, action_dim=NUM_ACTIONS)
prey_agent = DQNAgent(state_dim=14, action_dim=NUM_ACTIONS)

game_running = False
paused = False
reset_requested = False # Flag to force a reset
play_mode = "auto"    # "auto", "hunter", "prey"
player_action = 0      # Current action from user input

def load_models():
    """Load trained models."""
    hunter_path = os.path.join(MODEL_DIR, "hunter.pth")
    prey_path = os.path.join(MODEL_DIR, "prey.pth")

    h_loaded = hunter_agent.load(hunter_path)
    p_loaded = prey_agent.load(prey_path)

    if h_loaded: 
        hunter_agent.epsilon = 0.0
        print("  [Loaded] Hunter AI model")
    if p_loaded: 
        prey_agent.epsilon = 0.0
        print("  [Loaded] Prey AI model")
    
    return h_loaded and p_loaded

def play_loop():
    """Main game loop for play mode."""
    global game_running, paused, play_mode, player_action, reset_requested
    
    game_running = True
    episode_count = 0
    print("\n  [PLAY MODE] Server started. Listening for player input...")
    
    try:
        while game_running:
            episode_count += 1
            reset_requested = False
            
            # Reset environment
            hunter_obs, prey_obs = env.reset(episode=episode_count)
            
            state = env.get_state()
            state["episode"] = episode_count
            state["captured"] = False
            socketio.emit("game_reset", state)
            
            done = False
            print(f"  [Episode {episode_count}] Started")
            
            while not done and game_running and not reset_requested:
                # Use a while loop for pause to be highly responsive
                while paused and game_running and not reset_requested:
                    socketio.sleep(0.05)
                
                if not game_running:
                    break
                
                # Yield at the start to ensure inputs are processed
                socketio.sleep(0) 

                # Decide actions
                try:
                    if play_mode == "hunter":
                        h_act = player_action
                        p_act = prey_agent.select_action(prey_obs, training=False)
                    elif play_mode == "prey":
                        h_act = hunter_agent.select_action(hunter_obs, training=False)
                        p_act = player_action
                    else: # Auto
                        h_act = hunter_agent.select_action(hunter_obs, training=False)
                        p_act = prey_agent.select_action(prey_obs, training=False)
                    
                    # Step environment
                    hunter_obs, prey_obs, _, _, terminated, truncated, info = env.step(h_act, p_act)
                    done = terminated or truncated
                    
                    # Stream state
                    state = env.get_state()
                    state["episode"] = episode_count
                    state["captured"] = info.get("captured", False)
                    socketio.emit("game_state", state)
                except Exception as step_err:
                    print(f"  [Error] Step logic failure: {step_err}")
                    socketio.sleep(1) # Wait before retry
                    break
                
                # Target ~30 FPS
                socketio.sleep(1.0 / 30)
                
            # Episode end feedback
            if reset_requested:
                print(f"  [Episode {episode_count}] Reset requested.")
                continue

            if info.get("captured", False):
                print(f"  [Episode {episode_count}] Captured!")
                socketio.emit("captured", {"episode": episode_count})
                socketio.sleep(0.5)
            elif info.get("points_win", False):
                print(f"  [Episode {episode_count}] Prey Victory!")
                socketio.emit("prey_win", {"episode": episode_count})
                socketio.sleep(0.5)
            else:
                print(f"  [Episode {episode_count}] Draw/Time-out")

    except Exception as e:
        print(f"  [CRITICAL] Play loop crashed: {e}")
        game_running = False

# === Routes ===
@app.route("/")
def index():
    return send_from_directory(CLIENT_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(CLIENT_DIR, path)

@app.route("/favicon.ico")
def favicon():
    return "", 204

# === SocketIO Events ===
@socketio.on("connect")
def on_connect():
    global game_running
    print("  Client connected to Play Server")
    if not game_running:
        socketio.start_background_task(play_loop)

@socketio.on("set_play_mode")
def on_set_play_mode(data, *args):
    global play_mode
    play_mode = data.get("mode", "auto")
    print(f"  Play Mode set to: {play_mode.upper()}")

@socketio.on("player_input")
def on_player_input(data):
    global player_action
    player_action = data.get("action", 0)

@socketio.on("pause")
def on_pause(*args):
    global paused
    paused = True

@socketio.on("resume")
def on_resume(*args):
    global paused
    paused = False

@socketio.on("reset")
def on_reset(*args):
    global reset_requested
    reset_requested = True
    print("  Reset requested by client")

if __name__ == "__main__":
    print("=" * 50)
    print("  🎮 Ghost Chase — Player vs AI Mode")
    print("=" * 50)
    
    loaded = load_models()
    if not loaded:
        print("\n  [Warning] Models not found in /models. AI will be untrained.")
    
    print("\n  Open http://localhost:5000 in your browser")
    print("=" * 50)
    
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
