# 🎮 Ghost Chase — 3D RL Simulation / 3D 鬼抓人 RL 模擬

A 3D "Ghost Chase" simulation powered by reinforcement learning. Watch AI agents **train live** and play hunter vs prey in real-time — rendered with Three.js.

一款基於強化學習的 3D「鬼抓人」模擬遊戲。**即時觀看** AI 代理的訓練過程 — 使用 Three.js 渲染。

---

## 📁 Project Structure / 專案結構

```
RL/
├── server/
│   ├── environment.py      # Game environment (20×20 arena, 120° FOV)
│   ├── agent.py            # DQN agent implementation
│   ├── train.py            # CLI-only training (optional, no visualization)
│   ├── app.py              # Unified server: live training + visualization
│   ├── models/             # Saved trained models (.pth)
│   └── requirements.txt    # Python dependencies
├── client/
│   ├── index.html          # Main HTML page + training panel
│   ├── css/style.css       # Dark theme styling
│   └── js/
│       ├── main.js         # Entry point, game loop, training chart
│       ├── scene.js        # Three.js scene setup
│       ├── characters.js   # Hunter/Prey rendering + FOV cones
│       └── network.js      # WebSocket client
└── README.md
```

---

## 🚀 Getting Started / 快速開始

### 1. Install Dependencies / 安裝依賴

```bash
cd server
pip install -r requirements.txt
```

### 2. Run Live Training + Visualization / 啟動即時訓練

```bash
cd server
python app.py --mode train --episodes 5000 --speed 3
```

Then open **http://localhost:5000** in your browser to watch the training live!

在瀏覽器中開啟 **http://localhost:5000** 即可即時觀看訓練過程！

### 3. Run Demo Mode (Pre-trained) / 展示模式

```bash
cd server
python app.py --mode demo
```

---

## ⚡ Speed Settings / 速度控制

Speed can be set via CLI `--speed` flag or changed live via the UI buttons:

| Speed | Icon | Description |
|-------|------|-------------|
| 1 | 🐢 | Slow (15 FPS) — watch every detail |
| 2 | 🚶 | Medium (30 FPS) — balanced |
| 3 | 🏃 | Fast (120 FPS) — efficient training |
| 4 | ⚡ | Turbo — skip frames, max speed |

---

## 🎮 Controls / 操作說明

| Button | Function |
|--------|----------|
| ⏸ | Pause / Resume |
| 🔄 | Reset current episode |
| 👁 | Toggle FOV cone visualization |
| ✨ | Toggle movement trails |
| 🐢🚶🏃⚡ | Training speed control |
| 🖱 Drag | Orbit camera |
| 🖱 Scroll | Zoom in/out |

---

## 🧠 RL Design / 強化學習設計

- **Algorithm**: DQN with target network and experience replay
- **FOV**: Each agent has a **120° forward-facing** field of view
- **Actions**: 9 discrete (8 directions + stay)
- **Hunter reward**: Minimize distance + capture bonus (+10)
- **Prey reward**: Maximize distance + capture penalty (-10)
- **Arena**: 20×20 bounded plane
- **Capture**: Distance < 1.0 unit

---

## 🔧 Extending / 擴展指南

| Goal | File to Modify |
|------|---------------|
| Change arena size | `server/environment.py` → `ARENA_SIZE` |
| Adjust FOV angle | `server/environment.py` → `FOV_ANGLE` |
| Modify rewards | `server/environment.py` → `step()` |
| Change network architecture | `server/agent.py` → `QNetwork` |
| Adjust visual style | `client/css/style.css` |
| Modify 3D scene | `client/js/scene.js` |
| Change character appearance | `client/js/characters.js` |
