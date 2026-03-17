/**
 * main.js — Application Entry Point
 *
 * Initializes the Three.js scene, characters, and network,
 * then runs the animation loop and handles UI interactions.
 * Now also handles live training stats and chart visualization.
 */

(function () {
    'use strict';

    // === DOM Elements ===
    const loadingScreen = document.getElementById('loading-screen');
    const loaderStatus = document.getElementById('loader-status');
    const capturedPopup = document.getElementById('captured-popup');
    const preyWinPopup = document.getElementById('prey-win-popup');
    const episodeEl = document.getElementById('episode-count');
    const captureEl = document.getElementById('capture-count');
    const victoryEl = document.getElementById('victory-count');
    const drawEl = document.getElementById('draw-count');
    const stepEl = document.getElementById('step-count');
    const distanceEl = document.getElementById('distance-value');
    const connectionEl = document.getElementById('connection-status');
    const btnPause = document.getElementById('btn-pause');
    const btnReset = document.getElementById('btn-reset');
    const btnFov = document.getElementById('btn-fov');
    const btnTrails = document.getElementById('btn-trails');

    // Training panel elements
    const trainPanel = document.getElementById('train-panel');
    const trainStatusBadge = document.getElementById('train-status-badge');
    const trainProgressBar = document.getElementById('train-progress-bar');
    const trainProgressText = document.getElementById('train-progress-text');
    const trainEpsilon = document.getElementById('train-epsilon');
    const trainCapRate = document.getElementById('train-cap-rate');
    const trainVicRate = document.getElementById('train-vic-rate');
    const trainDrawRate = document.getElementById('train-draw-rate');
    const trainRewardH = document.getElementById('train-reward-h');
    const trainRewardP = document.getElementById('train-reward-p');
    const trainChart = document.getElementById('train-chart');
    const speedButtons = document.querySelectorAll('.speed-btn');

    // === State ===
    let captureCount = 0;
    let victoryCount = 0;
    let drawCount = 0;
    let stepCount = 0;
    let isPaused = false;
    let lastTimestamp = 0;

    // Play Mode State
    let playMode = 'auto'; // 'auto', 'hunter', 'prey'
    const keys = {
        w: false, a: false, s: false, d: false,
        ArrowUp: false, ArrowLeft: false, ArrowDown: false, ArrowRight: false,
        ' ': false
    };

    // Chart data
    const chartData = {
        captureRates: [],
        maxPoints: 100,
    };

    // === Initialize ===
    function init() {
        // 1. Scene
        const container = document.getElementById('canvas-container');
        SceneManager.init(container);

        // 2. Characters
        CharacterManager.init(SceneManager.getScene());

        // 3. Network
        NetworkManager.init();

        // 4. Event handlers
        _setupNetworkEvents();
        _setupUI();

        // 5. Start animation
        requestAnimationFrame(animate);

        loaderStatus.textContent = 'Connecting to server...';
    }

    // === Network Events ===
    function _setupNetworkEvents() {
        NetworkManager.on('connected', () => {
            loaderStatus.textContent = 'Connected! Starting simulation...';
            connectionEl.classList.add('connected');
            connectionEl.querySelector('span').textContent = 'Connected';

            // Hide loading after short delay
            setTimeout(() => {
                loadingScreen.classList.add('hidden');
            }, 800);
        });

        NetworkManager.on('disconnected', () => {
            connectionEl.classList.remove('connected');
            connectionEl.querySelector('span').textContent = 'Disconnected';
        });

        NetworkManager.on('error', () => {
            loaderStatus.textContent = 'Connection failed. Retrying...';
        });

        NetworkManager.on('gameState', (state) => {
            stepCount = state.steps !== undefined ? state.steps : stepCount + 1;
            CharacterManager.setState(state);

            // Update HUD
            episodeEl.textContent = state.episode || 0;
            stepEl.textContent = stepCount;
            
            if (state.points_collected !== undefined) {
                document.getElementById('points-count').innerText = `${state.points_collected} / 3`;
            }
            
            const dist = CharacterManager.getDistance();
            distanceEl.textContent = dist.toFixed(1);
        });

        NetworkManager.on('captured', (data) => {
            captureCount++;
            captureEl.textContent = captureCount;

            // Show popup
            capturedPopup.classList.remove('hidden');
            capturedPopup.classList.add('visible');

            setTimeout(() => {
                capturedPopup.classList.remove('visible');
                capturedPopup.classList.add('hidden');
            }, 1200);
        });

        NetworkManager.on('preyWin', (data) => {
            victoryCount++;
            if (victoryEl) victoryEl.textContent = victoryCount;

            // Show victory popup
            preyWinPopup.classList.remove('hidden');
            preyWinPopup.classList.add('visible');

            setTimeout(() => {
                preyWinPopup.classList.remove('visible');
                preyWinPopup.classList.add('hidden');
            }, 1500);
        });

        NetworkManager.on('gameReset', (state) => {
            stepCount = 0;
            CharacterManager.resetPositions(state);
            SceneManager.setObstacles(state.obstacles);
            episodeEl.textContent = state.episode || 0;
            
            if (state.points_collected !== undefined) {
                document.getElementById('points-count').innerText = `${state.points_collected} / 3`;
            }
        });

        // === Training Events ===
        NetworkManager.on('trainStats', (data) => {
            _updateTrainPanel(data);
        });

        NetworkManager.on('trainComplete', (data) => {
            trainStatusBadge.textContent = 'COMPLETE';
            trainStatusBadge.classList.add('complete');
            trainProgressBar.style.width = '100%';
            trainProgressText.textContent = `${data.episodes} / ${data.episodes}`;
        });
    }

    // === Update Training Panel ===
    function _updateTrainPanel(data) {
        // Progress bar
        const progress = (data.episode / data.max_episodes) * 100;
        trainProgressBar.style.width = progress + '%';
        trainProgressText.textContent = `${data.episode} / ${data.max_episodes}`;

        // Stats
        trainEpsilon.textContent = data.epsilon.toFixed(4);
        trainCapRate.textContent = data.capture_rate + '%';
        if (trainVicRate) trainVicRate.textContent = data.victory_rate + '%';
        if (trainDrawRate) trainDrawRate.textContent = data.draw_rate + '%';
        trainRewardH.textContent = data.avg_reward_h.toFixed(2);
        trainRewardP.textContent = data.avg_reward_p.toFixed(2);

        // Update counts from server
        captureEl.textContent = data.total_captures;
        captureCount = data.total_captures;
        
        if (victoryEl) {
            victoryEl.textContent = data.total_victories;
            victoryCount = data.total_victories;
        }
        if (drawEl) {
            drawEl.textContent = data.total_draws;
            drawCount = data.total_draws;
        }

        // Chart data
        chartData.captureRates.push(data.capture_rate);
        if (chartData.captureRates.length > chartData.maxPoints) {
            chartData.captureRates.shift();
        }

        _drawChart();
    }

    // === Chart Drawing ===
    function _drawChart() {
        const canvas = trainChart;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const data = chartData.captureRates;

        ctx.clearRect(0, 0, w, h);

        if (data.length < 2) return;

        // Background grid
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = (h / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Data line
        const maxVal = Math.max(100, ...data);
        const stepX = w / (chartData.maxPoints - 1);

        // Fill gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, 'rgba(255, 111, 60, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 111, 60, 0.0)');

        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < data.length; i++) {
            const x = i * stepX;
            const y = h - (data[i] / maxVal) * (h - 4);
            ctx.lineTo(x, y);
        }
        ctx.lineTo((data.length - 1) * stepX, h);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Line
        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = i * stepX;
            const y = h - (data[i] / maxVal) * (h - 4);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#ff6f3c';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Current value label
        const lastVal = data[data.length - 1];
        const lastX = (data.length - 1) * stepX;
        const lastY = h - (lastVal / maxVal) * (h - 4);

        ctx.fillStyle = '#ff6f3c';
        ctx.beginPath();
        ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(lastVal.toFixed(1) + '%', w - 4, 12);
    }

    // === UI Controls ===
    function _setupUI() {
        // Pause button
        btnPause.addEventListener('click', () => {
            isPaused = !isPaused;
            if (isPaused) {
                NetworkManager.pause();
                btnPause.querySelector('.btn-icon').textContent = '▶';
                btnPause.classList.add('active');
            } else {
                NetworkManager.resume();
                btnPause.querySelector('.btn-icon').textContent = '⏸';
                btnPause.classList.remove('active');
            }
        });

        // Reset button
        btnReset.addEventListener('click', () => {
            NetworkManager.reset();
            stepCount = 0;
        });

        // FOV toggle
        btnFov.classList.add('active');
        btnFov.addEventListener('click', () => {
            const show = CharacterManager.toggleFov();
            btnFov.classList.toggle('active', show);
        });

        // Trails toggle
        btnTrails.classList.add('active');
        btnTrails.addEventListener('click', () => {
            const show = CharacterManager.toggleTrails();
            btnTrails.classList.toggle('active', show);
        });

        // Speed buttons
        speedButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const speed = parseInt(btn.dataset.speed);
                NetworkManager.setSpeed(speed);

                // Update active state
                speedButtons.forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Play Mode buttons
        const playButtons = document.querySelectorAll('.play-btn');
        const playHint = document.getElementById('play-hint');
        playButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                playMode = btn.dataset.mode;
                playButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                if (playMode === 'auto') {
                    playHint.classList.add('hidden');
                } else {
                    playHint.classList.remove('hidden');
                }
                
                NetworkManager.emit('set_play_mode', { mode: playMode });
            });
        });

        // Keyboard listeners
        window.addEventListener('keydown', (e) => {
            if (keys.hasOwnProperty(e.key)) {
                keys[e.key] = true;
                if (e.key === ' ') e.preventDefault(); // Stop scrolling
            }
        });
        window.addEventListener('keyup', (e) => {
            if (keys.hasOwnProperty(e.key)) keys[e.key] = false;
        });
    }

    let lastAction = -1;
    function _processInput() {
        if (playMode === 'auto') return;

        const up = keys.w || keys.ArrowUp;
        const down = keys.s || keys.ArrowDown;
        const left = keys.a || keys.ArrowLeft;
        const right = keys.d || keys.ArrowRight;
        const jump = keys[' '];

        let action = 0; // Stay

        if (jump) {
            action = 9;
        } else if (up && left) {
            action = 8;
        } else if (up && right) {
            action = 2;
        } else if (down && left) {
            action = 6;
        } else if (down && right) {
            action = 4;
        } else if (up) {
            action = 1;
        } else if (down) {
            action = 5;
        } else if (left) {
            action = 7;
        } else if (right) {
            action = 3;
        }

        // Throttle: Only send if action changed OR every 10 frames (heartbeat)
        if (action !== lastAction || stepCount % 10 === 0) {
            NetworkManager.emit('player_input', { action });
            lastAction = action;
        }
    }

    // === Animation Loop ===
    function animate(timestamp) {
        requestAnimationFrame(animate);

        const deltaTime = (timestamp - lastTimestamp) / 1000;
        lastTimestamp = timestamp;

        if (!isPaused) {
            _processInput();
        }

        // Update characters
        CharacterManager.update(deltaTime);

        // Update distance display
        const dist = CharacterManager.getDistance();
        distanceEl.textContent = dist.toFixed(1);

        // Render
        SceneManager.render();
    }

    // === Start ===
    window.addEventListener('DOMContentLoaded', init);
})();
