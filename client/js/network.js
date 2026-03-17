/**
 * network.js — WebSocket Client (Socket.IO)
 *
 * Connects to the Python Flask-SocketIO server and
 * dispatches game state and training events to the application.
 */

const NetworkManager = (function () {
    let socket = null;
    let connected = false;
    let callbacks = {};

    function init(serverUrl) {
        const url = serverUrl || window.location.origin;

        socket = io(url, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
        });

        socket.on('connect', () => {
            connected = true;
            console.log('[Network] Connected to server');
            _trigger('connected');
        });

        socket.on('disconnect', () => {
            connected = false;
            console.log('[Network] Disconnected from server');
            _trigger('disconnected');
        });

        socket.on('connect_error', (err) => {
            console.warn('[Network] Connection error:', err.message);
            _trigger('error', err);
        });

        // Game events
        socket.on('game_state', (state) => {
            _trigger('gameState', state);
        });

        socket.on('captured', (data) => {
            _trigger('captured', data);
        });

        socket.on('prey_win', (data) => {
            _trigger('preyWin', data);
        });

        socket.on('game_reset', (state) => {
            _trigger('gameReset', state);
        });

        // Training events
        socket.on('train_stats', (data) => {
            _trigger('trainStats', data);
        });

        socket.on('train_complete', (data) => {
            _trigger('trainComplete', data);
        });
    }

    function _trigger(event, data) {
        if (callbacks[event]) {
            callbacks[event].forEach((cb) => cb(data));
        }
    }

    function on(event, callback) {
        if (!callbacks[event]) callbacks[event] = [];
        callbacks[event].push(callback);
    }

    function emit(event, data) {
        if (socket && connected) {
            socket.emit(event, data);
        }
    }

    function isConnected() {
        return connected;
    }

    function pause() { emit('pause'); }
    function resume() { emit('resume'); }
    function reset() { emit('reset'); }

    function setSpeed(speed) {
        emit('set_speed', { speed: speed });
    }

    return { init, on, emit, isConnected, pause, resume, reset, setSpeed };
})();
