/**
 * characters.js — Hunter & Prey 3D Characters + Power-Up Item
 *
 * Creates orange (hunter) and blue (prey) cubes with glow effects,
 * FOV cone visualization (120° normal, 360° when buffed),
 * smooth movement interpolation, and a glowing power-up item.
 */

const CharacterManager = (function () {
    const CUBE_SIZE = 0.8;
    const FOV_ANGLE = (120 * Math.PI) / 180; // 120 degrees in radians
    const FOV_RADIUS = 5.0;
    const TRAIL_MAX = 40;
    const LERP_SPEED = 0.15;

    // Colors
    const HUNTER_COLOR = 0xff6f3c;
    const HUNTER_EMISSIVE = 0xff4400;
    const PREY_COLOR = 0x4fc3f7;
    const PREY_EMISSIVE = 0x0088cc;
    const POWERUP_COLOR = 0xffdd00;
    const POWERUP_EMISSIVE = 0xffaa00;

    // Character objects
    let hunterGroup, preyGroup;
    let hunterMesh, preyMesh;
    let hunterFov, preyFov;
    let hunterFovBuffed, preyFovBuffed; // 360° FOV cones
    let hunterLight, preyLight;
    let hunterLabel, preyLabel;
    let hunterTrail = [], preyTrail = [];

    // Power-up objects
    let powerupGroup;
    let powerupMesh;
    let powerupLight;

    // Target state (for interpolation)
    let hunterTarget = { x: 0, z: 0, angle: 0 };
    let preyTarget = { x: 0, z: 0, angle: 0 };

    // Power-up state
    let powerupState = { x: 0, z: 0, active: false };
    let hunterHasBuff = false;
    let preyHasBuff = false;

    // Display toggling
    let showFov = true;
    let showTrails = true;

    let scene;

    function init(threeScene) {
        scene = threeScene;
        _createHunter();
        _createPrey();
        _createPowerup();
    }

    function _createCube(color, emissive) {
        const geo = new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
        const mat = new THREE.MeshStandardMaterial({
            color: color,
            emissive: emissive,
            emissiveIntensity: 0.4,
            roughness: 0.3,
            metalness: 0.5,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.position.y = CUBE_SIZE / 2;
        return mesh;
    }

    function _createFovCone(color) {
        // Create a 120-degree cone/sector shape
        const segments = 32;
        const halfAngle = FOV_ANGLE / 2;

        const shape = new THREE.Shape();
        shape.moveTo(0, 0);

        for (let i = 0; i <= segments; i++) {
            const angle = -halfAngle + (FOV_ANGLE * i) / segments;
            const x = Math.sin(angle) * FOV_RADIUS;
            const y = Math.cos(angle) * FOV_RADIUS;
            shape.lineTo(x, y);
        }
        shape.lineTo(0, 0);

        const geo = new THREE.ShapeGeometry(shape);
        const mat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = Math.PI / 2;
        mesh.position.y = 0.05;

        return mesh;
    }

    function _createFovCircle(color) {
        // Create a full 360-degree circle for buffed FOV
        const segments = 48;
        const shape = new THREE.Shape();

        for (let i = 0; i <= segments; i++) {
            const angle = (2 * Math.PI * i) / segments;
            const x = Math.sin(angle) * FOV_RADIUS;
            const y = Math.cos(angle) * FOV_RADIUS;
            if (i === 0) shape.moveTo(x, y);
            else shape.lineTo(x, y);
        }

        const geo = new THREE.ShapeGeometry(shape);
        const mat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = Math.PI / 2;
        mesh.position.y = 0.05;
        mesh.visible = false; // Hidden by default

        return mesh;
    }

    function _createLabel(text, color) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');

        // Draw background pill
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(40, 30, 176, 68, 34);
        } else {
            ctx.rect(40, 30, 176, 68); // fallback
        }
        ctx.fill();

        // Draw text
        ctx.font = 'bold 44px "Microsoft JhengHei", Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = color;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 4;
        ctx.fillText(text, 128, 64);

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(1.5, 0.75, 1);
        sprite.position.y = 1.8; // Position above character
        return sprite;
    }

    function _createHunter() {
        hunterGroup = new THREE.Group();

        // Main cube
        hunterMesh = _createCube(HUNTER_COLOR, HUNTER_EMISSIVE);
        hunterGroup.add(hunterMesh);

        // FOV cone (normal 120°)
        hunterFov = _createFovCone(HUNTER_COLOR);
        hunterGroup.add(hunterFov);

        // FOV circle (buffed 360°)
        hunterFovBuffed = _createFovCircle(HUNTER_COLOR);
        hunterGroup.add(hunterFovBuffed);

        // Point light glow
        hunterLight = new THREE.PointLight(HUNTER_COLOR, 0.6, 8);
        hunterLight.position.set(0, 1.5, 0);
        hunterGroup.add(hunterLight);

        // Eyes (two small white cubes on front face)
        _addEyes(hunterGroup);

        // Direction indicator (small arrow)
        _addDirectionArrow(hunterGroup, HUNTER_COLOR);

        // Label
        hunterLabel = _createLabel('鬼', '#ff6f3c');
        hunterGroup.add(hunterLabel);

        scene.add(hunterGroup);
    }

    function _createPrey() {
        preyGroup = new THREE.Group();

        // Main cube
        preyMesh = _createCube(PREY_COLOR, PREY_EMISSIVE);
        preyGroup.add(preyMesh);

        // FOV cone (normal 120°)
        preyFov = _createFovCone(PREY_COLOR);
        preyGroup.add(preyFov);

        // FOV circle (buffed 360°)
        preyFovBuffed = _createFovCircle(PREY_COLOR);
        preyGroup.add(preyFovBuffed);

        // Point light glow
        preyLight = new THREE.PointLight(PREY_COLOR, 0.6, 8);
        preyLight.position.set(0, 1.5, 0);
        preyGroup.add(preyLight);

        // Eyes
        _addEyes(preyGroup);

        // Direction indicator
        _addDirectionArrow(preyGroup, PREY_COLOR);

        // Label
        preyLabel = _createLabel('人', '#4fc3f7');
        preyGroup.add(preyLabel);

        scene.add(preyGroup);
    }

    function _createPowerup() {
        powerupGroup = new THREE.Group();

        // Octahedron (diamond shape)
        const geo = new THREE.OctahedronGeometry(0.4, 0);
        const mat = new THREE.MeshStandardMaterial({
            color: POWERUP_COLOR,
            emissive: POWERUP_EMISSIVE,
            emissiveIntensity: 0.8,
            roughness: 0.2,
            metalness: 0.7,
            transparent: true,
            opacity: 0.9,
        });
        powerupMesh = new THREE.Mesh(geo, mat);
        powerupMesh.position.y = 0.8;
        powerupMesh.castShadow = true;
        powerupGroup.add(powerupMesh);

        // Glow ring on ground
        const ringGeo = new THREE.RingGeometry(0.5, 0.7, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: POWERUP_COLOR,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.02;
        powerupGroup.add(ring);

        // Point light
        powerupLight = new THREE.PointLight(POWERUP_COLOR, 0.8, 6);
        powerupLight.position.set(0, 1.2, 0);
        powerupGroup.add(powerupLight);

        // Eye icon above (to indicate vision power-up)
        const eyeGeo = new THREE.SphereGeometry(0.12, 8, 8);
        const eyeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const eye = new THREE.Mesh(eyeGeo, eyeMat);
        eye.position.set(0, 1.4, 0);
        powerupGroup.add(eye);

        powerupGroup.visible = false;
        scene.add(powerupGroup);
    }

    function _addEyes(group) {
        const eyeGeo = new THREE.BoxGeometry(0.12, 0.12, 0.05);
        const eyeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

        const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
        eyeL.position.set(-0.18, CUBE_SIZE / 2 + 0.1, CUBE_SIZE / 2 + 0.02);
        group.add(eyeL);

        const eyeR = new THREE.Mesh(eyeGeo, eyeMat);
        eyeR.position.set(0.18, CUBE_SIZE / 2 + 0.1, CUBE_SIZE / 2 + 0.02);
        group.add(eyeR);

        // Pupils
        const pupilGeo = new THREE.BoxGeometry(0.06, 0.06, 0.03);
        const pupilMat = new THREE.MeshBasicMaterial({ color: 0x111111 });

        const pupilL = new THREE.Mesh(pupilGeo, pupilMat);
        pupilL.position.set(-0.18, CUBE_SIZE / 2 + 0.1, CUBE_SIZE / 2 + 0.05);
        group.add(pupilL);

        const pupilR = new THREE.Mesh(pupilGeo, pupilMat);
        pupilR.position.set(0.18, CUBE_SIZE / 2 + 0.1, CUBE_SIZE / 2 + 0.05);
        group.add(pupilR);
    }

    function _addDirectionArrow(group, color) {
        const arrowGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
        const arrowMat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.7,
        });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        arrow.rotation.x = Math.PI / 2;
        arrow.position.set(0, CUBE_SIZE / 2, CUBE_SIZE / 2 + 0.4);
        group.add(arrow);
    }

    function _addTrailDot(trail, position, color) {
        if (!showTrails) return;

        const geo = new THREE.SphereGeometry(0.08, 6, 6);
        const mat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.5,
        });
        const dot = new THREE.Mesh(geo, mat);
        dot.position.copy(position);
        dot.position.y = 0.1;
        scene.add(dot);
        trail.push({ mesh: dot, life: 1.0 });

        // Remove old dots
        while (trail.length > TRAIL_MAX) {
            const old = trail.shift();
            scene.remove(old.mesh);
            old.mesh.geometry.dispose();
            old.mesh.material.dispose();
        }
    }

    function setState(state) {
        hunterTarget.x = state.hunter_x;
        hunterTarget.z = state.hunter_z;
        hunterTarget.angle = state.hunter_angle;

        preyTarget.x = state.prey_x;
        preyTarget.z = state.prey_z;
        preyTarget.angle = state.prey_angle;

        // Power-up state
        if (state.powerup_active !== undefined) {
            powerupState.x = state.powerup_x;
            powerupState.z = state.powerup_z;
            powerupState.active = state.powerup_active;
        }

        if (state.hunter_fov_buff !== undefined) {
            hunterHasBuff = state.hunter_fov_buff;
        }
        if (state.prey_fov_buff !== undefined) {
            preyHasBuff = state.prey_fov_buff;
        }
    }

    function update(deltaTime) {
        // Smoothly interpolate positions
        hunterGroup.position.x += (hunterTarget.x - hunterGroup.position.x) * LERP_SPEED;
        hunterGroup.position.z += (-hunterTarget.z - hunterGroup.position.z) * LERP_SPEED;

        preyGroup.position.x += (preyTarget.x - preyGroup.position.x) * LERP_SPEED;
        preyGroup.position.z += (-preyTarget.z - preyGroup.position.z) * LERP_SPEED;

        // Smoothly interpolate rotations
        const hunterTargetRotY = hunterTarget.angle + Math.PI / 2;
        const preyTargetRotY = preyTarget.angle + Math.PI / 2;

        hunterGroup.rotation.y = _lerpAngle(hunterGroup.rotation.y, hunterTargetRotY, LERP_SPEED);
        preyGroup.rotation.y = _lerpAngle(preyGroup.rotation.y, preyTargetRotY, LERP_SPEED);

        // FOV visibility
        if (showFov) {
            hunterFov.visible = !hunterHasBuff;
            hunterFovBuffed.visible = hunterHasBuff;
            preyFov.visible = !preyHasBuff;
            preyFovBuffed.visible = preyHasBuff;
        } else {
            hunterFov.visible = false;
            hunterFovBuffed.visible = false;
            preyFov.visible = false;
            preyFovBuffed.visible = false;
        }

        // Power-up item
        if (powerupState.active) {
            powerupGroup.visible = true;
            powerupGroup.position.x = powerupState.x;
            powerupGroup.position.z = -powerupState.z;
            powerupMesh.rotation.y += 0.03;
            powerupMesh.position.y = 0.8 + Math.sin(Date.now() * 0.003) * 0.15;
        } else {
            powerupGroup.visible = false;
        }

        // Trail dots
        _addTrailDot(hunterTrail, hunterGroup.position, HUNTER_COLOR);
        _addTrailDot(preyTrail, preyGroup.position, PREY_COLOR);

        // Fade trails
        _updateTrails(hunterTrail);
        _updateTrails(preyTrail);

        // Keep normal labels
        _updateLabel(hunterLabel, '鬼', '#ff6f3c');
        _updateLabel(preyLabel, '人', '#4fc3f7');

        // Pulse glow effect
        const pulse = 0.4 + Math.sin(Date.now() * 0.003) * 0.15;
        hunterMesh.material.emissiveIntensity = hunterHasBuff ? 0.8 : pulse;
        preyMesh.material.emissiveIntensity = preyHasBuff ? 0.8 : pulse;
    }

    function _updateLabel(sprite, text, color) {
        if (sprite.lastText === text) return;
        
        const canvas = sprite.material.map.image;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw background pill
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(40, 30, 176, 68, 34);
        } else {
            ctx.rect(40, 30, 176, 68);
        }
        ctx.fill();

        // Draw text
        ctx.font = 'bold 44px "Microsoft JhengHei", "PingFang TC", "Source Han Sans TC", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = color;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
        ctx.shadowBlur = 6;
        ctx.fillText(text, 128, 64);
        
        sprite.material.map.needsUpdate = true;
        sprite.lastText = text;
    }

    function _updateTrails(trail) {
        for (let i = trail.length - 1; i >= 0; i--) {
            trail[i].life -= 0.02;
            trail[i].mesh.material.opacity = trail[i].life * 0.4;

            if (trail[i].life <= 0) {
                scene.remove(trail[i].mesh);
                trail[i].mesh.geometry.dispose();
                trail[i].mesh.material.dispose();
                trail.splice(i, 1);
            }
        }
    }

    function _lerpAngle(current, target, t) {
        let diff = target - current;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        return current + diff * t;
    }

    function resetPositions(state) {
        hunterGroup.position.x = state.hunter_x;
        hunterGroup.position.z = -state.hunter_z;
        preyGroup.position.x = state.prey_x;
        preyGroup.position.z = -state.prey_z;

        hunterTarget.x = state.hunter_x;
        hunterTarget.z = state.hunter_z;
        preyTarget.x = state.prey_x;
        preyTarget.z = state.prey_z;

        hunterHasBuff = false;
        preyHasBuff = false;

        _clearTrail(hunterTrail);
        _clearTrail(preyTrail);
    }

    function _clearTrail(trail) {
        trail.forEach((item) => {
            scene.remove(item.mesh);
            item.mesh.geometry.dispose();
            item.mesh.material.dispose();
        });
        trail.length = 0;
    }

    function toggleFov() {
        showFov = !showFov;
        return showFov;
    }

    function toggleTrails() {
        showTrails = !showTrails;
        if (!showTrails) {
            _clearTrail(hunterTrail);
            _clearTrail(preyTrail);
        }
        return showTrails;
    }

    function getDistance() {
        const dx = hunterGroup.position.x - preyGroup.position.x;
        const dz = hunterGroup.position.z - preyGroup.position.z;
        return Math.sqrt(dx * dx + dz * dz);
    }

    return {
        init,
        setState,
        update,
        resetPositions,
        toggleFov,
        toggleTrails,
        getDistance,
    };
})();
