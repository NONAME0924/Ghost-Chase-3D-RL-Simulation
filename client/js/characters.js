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
    const POINT_COLOR = 0xffdd00;
    const POINT_EMISSIVE = 0xffaa00;

    // Character objects
    let hunterGroup, preyGroup;
    let hunterMesh, preyMesh;
    let hunterFov, preyFov;
    let hunterLight, preyLight;
    let hunterLabel, preyLabel;
    let hunterTrail = [], preyTrail = [];

    // Points objects (3 dots to collect)
    let pointsGroups = [];

    // Target state (for interpolation)
    let hunterTarget = { x: 0, z: 0, angle: 0 };
    let preyTarget = { x: 0, z: 0, angle: 0 };

    // Points state
    let pointsState = [
        { x: 0, z: 0, active: false },
        { x: 0, z: 0, active: false },
        { x: 0, z: 0, active: false }
    ];

    // Display toggling
    let showFov = true;
    let showTrails = true;

    let scene;

    function init(threeScene) {
        scene = threeScene;
        _createHunter();
        _createPrey();
        _createPoints(3);
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
        const segments = 32;
        const geo = new THREE.BufferGeometry();
        // Initialize vertices with FOV_RADIUS to be visible from frame 1
        const vertices = new Float32Array((segments + 2) * 3);
        const halfAngle = FOV_ANGLE / 2;
        
        // Point 0: Center
        vertices[0] = 0; vertices[1] = 0; vertices[2] = 0;
        // Points 1..segments+1: Perimeter
        for (let i = 0; i <= segments; i++) {
            const angle = -halfAngle + (FOV_ANGLE * i) / segments;
            const idx = (i + 1) * 3;
            vertices[idx] = Math.sin(angle) * FOV_RADIUS;
            vertices[idx + 1] = Math.cos(angle) * FOV_RADIUS;
            vertices[idx + 2] = 0;
        }

        const indices = [];
        for (let i = 1; i <= segments; i++) {
            indices.push(0, i, i + 1);
        }
        
        geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3).setUsage(THREE.DynamicDrawUsage));
        geo.setIndex(indices);

        const mat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.35, 
            side: THREE.DoubleSide,
            depthWrite: false,
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = Math.PI / 2; // Fixed: Geometry +Y (points) now maps to Mesh +Z (forward)
        mesh.position.y = 0.1; // Elevate slightly to avoid floor fighting
        mesh.renderOrder = 10;
        mesh.frustumCulled = false;
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

    function _createPoints(count) {
        for (let i = 0; i < count; i++) {
            const group = new THREE.Group();

            // Octahedron (diamond shape)
            const geo = new THREE.OctahedronGeometry(0.3, 0);
            const mat = new THREE.MeshStandardMaterial({
                color: POINT_COLOR,
                emissive: POINT_EMISSIVE,
                emissiveIntensity: 0.8,
                roughness: 0.2,
                metalness: 0.7,
                transparent: true,
                opacity: 0.9,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.y = 0.6;
            mesh.castShadow = true;
            group.add(mesh);

            // Glow ring on ground
            const ringGeo = new THREE.RingGeometry(0.4, 0.5, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: POINT_COLOR,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide,
                depthWrite: false,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.02;
            group.add(ring);

            // Point light
            const light = new THREE.PointLight(POINT_COLOR, 0.6, 4);
            light.position.set(0, 0.8, 0);
            group.add(light);

            group.visible = false;
            scene.add(group);
            pointsGroups.push(group);
        }
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

        // Points state
        if (state.points_pos) {
            for (let i = 0; i < 3; i++) {
                pointsState[i] = {
                    x: state.points_pos[i][0],
                    z: state.points_pos[i][1],
                    active: state.points_active[i]
                };
            }
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
        hunterFov.visible = showFov;
        preyFov.visible = showFov;

        // Update points
        for (let i = 0; i < pointsGroups.length; i++) {
            const group = pointsGroups[i];
            const pState = pointsState[i];
            if (pState.active) {
                group.visible = true;
                group.position.x = pState.x;
                group.position.z = -pState.z;
                group.children[0].rotation.y += 0.03;
                group.children[0].position.y = 0.6 + Math.sin(Date.now() * 0.003 + i) * 0.1;
            } else {
                group.visible = false;
            }
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
        hunterMesh.material.emissiveIntensity = pulse;
        preyMesh.material.emissiveIntensity = pulse;

        // Update Dynamic FOVs (with existence check)
        if (showFov) {
            if (hunterFov) _updateFovMesh(hunterFov, FOV_ANGLE, 32);
            if (preyFov) _updateFovMesh(preyFov, FOV_ANGLE, 32);
        }
    }

    const raycaster = new THREE.Raycaster();
    const tempOrigin = new THREE.Vector3();

    function _updateFovMesh(mesh, fov, segments) {
        if (!mesh || !mesh.visible || !mesh.parent) return;

        try {
            const attr = mesh.geometry.attributes.position;
            const obstacles = SceneManager.getObstacles();
            const halfFov = fov / 2;

            mesh.parent.updateMatrixWorld();
            mesh.parent.getWorldPosition(tempOrigin);
            tempOrigin.y = 0.5;

            // Update Center
            attr.setXYZ(0, 0, 0, 0);

            for (let i = 0; i <= segments; i++) {
                const relAngle = -halfFov + (fov * i) / segments;
                const localDir = new THREE.Vector3(Math.sin(relAngle), 0, Math.cos(relAngle));
                const worldDir = localDir.applyQuaternion(mesh.parent.quaternion).normalize();

                raycaster.set(tempOrigin, worldDir);
                let dist = FOV_RADIUS;

                if (obstacles && obstacles.length > 0) {
                    const intersects = raycaster.intersectObjects(obstacles, true); // Recursive search
                    if (intersects.length > 0 && intersects[0].distance < FOV_RADIUS) {
                        dist = intersects[0].distance;
                    }
                }

                // Map results back to local triangle fan
                attr.setXYZ(i + 1, Math.sin(relAngle) * dist, Math.cos(relAngle) * dist, 0);
            }

            attr.needsUpdate = true;
            mesh.geometry.computeBoundingSphere();
        } catch (e) {
            // Silently skip frames if state is inconsistent
        }
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
