/**
 * scene.js — Three.js Scene Setup
 *
 * Creates the 3D scene, camera, lights, grid floor, and boundary walls.
 */

const SceneManager = (function () {
    const ARENA_SIZE = 20;
    const HALF = ARENA_SIZE / 2;

    let scene, camera, renderer, controls;
    let gridFloor, boundaryGroup;

    function init(container) {
        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        scene.fog = new THREE.FogExp2(0x1a1a2e, 0.012);

        // Camera
        camera = new THREE.PerspectiveCamera(
            55,
            window.innerWidth / window.innerHeight,
            0.1,
            200
        );
        camera.position.set(18, 22, 18);
        camera.lookAt(0, 0, 0);

        // Renderer
        renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;
        container.appendChild(renderer.domElement);

        // Orbit Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 8;
        controls.maxDistance = 50;
        controls.maxPolarAngle = Math.PI / 2.2;
        controls.target.set(0, 0, 0);

        // Lighting
        _createLights();

        // Floor
        _createFloor();

        // Boundary
        _createBoundary();

        // Resize
        window.addEventListener('resize', _onResize);

        return { scene, camera, renderer, controls };
    }

    function _createLights() {
        // Ambient
        const ambient = new THREE.AmbientLight(0x606080, 0.8);
        scene.add(ambient);

        // Hemisphere
        const hemi = new THREE.HemisphereLight(0x88aaff, 0x334455, 0.7);
        scene.add(hemi);

        // Main directional
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(15, 25, 10);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        dirLight.shadow.camera.near = 1;
        dirLight.shadow.camera.far = 60;
        dirLight.shadow.camera.left = -15;
        dirLight.shadow.camera.right = 15;
        dirLight.shadow.camera.top = 15;
        dirLight.shadow.camera.bottom = -15;
        scene.add(dirLight);

        // Accent light (hunter color)
        const accentA = new THREE.PointLight(0xff6f3c, 0.3, 30);
        accentA.position.set(-8, 8, -8);
        scene.add(accentA);

        // Accent light (prey color)
        const accentB = new THREE.PointLight(0x4fc3f7, 0.3, 30);
        accentB.position.set(8, 8, 8);
        scene.add(accentB);
    }

    function _createFloor() {
        // Grid helper
        const gridHelper = new THREE.GridHelper(ARENA_SIZE, 20, 0x2a2a40, 0x1a1a30);
        gridHelper.position.y = 0.01;
        scene.add(gridHelper);

        // Solid floor
        const floorGeo = new THREE.PlaneGeometry(ARENA_SIZE, ARENA_SIZE);
        const floorMat = new THREE.MeshStandardMaterial({
            color: 0x1e1e32,
            roughness: 0.9,
            metalness: 0.1,
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        scene.add(floor);

        // Subtle glow plane beneath floor
        const glowGeo = new THREE.PlaneGeometry(ARENA_SIZE + 4, ARENA_SIZE + 4);
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x1a1a3a,
            transparent: true,
            opacity: 0.3,
        });
        const glowPlane = new THREE.Mesh(glowGeo, glowMat);
        glowPlane.rotation.x = -Math.PI / 2;
        glowPlane.position.y = -0.05;
        scene.add(glowPlane);
    }

    function _createBoundary() {
        boundaryGroup = new THREE.Group();

        const wallHeight = 1.5;
        const wallThickness = 0.1;

        // Create 4 walls
        const wallData = [
            { pos: [0, wallHeight / 2, -HALF], size: [ARENA_SIZE, wallHeight, wallThickness] },
            { pos: [0, wallHeight / 2, HALF], size: [ARENA_SIZE, wallHeight, wallThickness] },
            { pos: [-HALF, wallHeight / 2, 0], size: [wallThickness, wallHeight, ARENA_SIZE] },
            { pos: [HALF, wallHeight / 2, 0], size: [wallThickness, wallHeight, ARENA_SIZE] },
        ];

        const wallMat = new THREE.MeshStandardMaterial({
            color: 0x3a3a5a,
            transparent: true,
            opacity: 0.15,
            roughness: 0.3,
            metalness: 0.6,
        });

        wallData.forEach((data) => {
            const geo = new THREE.BoxGeometry(...data.size);
            const mesh = new THREE.Mesh(geo, wallMat);
            mesh.position.set(...data.pos);
            boundaryGroup.add(mesh);
        });

        // Corner posts
        const postGeo = new THREE.CylinderGeometry(0.15, 0.15, wallHeight + 0.5, 8);
        const postMat = new THREE.MeshStandardMaterial({
            color: 0x6a6a8a,
            metalness: 0.8,
            roughness: 0.2,
        });

        const corners = [
            [-HALF, 0, -HALF], [HALF, 0, -HALF],
            [-HALF, 0, HALF], [HALF, 0, HALF],
        ];

        corners.forEach((pos) => {
            const post = new THREE.Mesh(postGeo, postMat);
            post.position.set(pos[0], (wallHeight + 0.5) / 2, pos[2]);
            post.castShadow = true;
            boundaryGroup.add(post);
        });

        // Edge glow lines
        const lineMat = new THREE.LineBasicMaterial({
            color: 0x4a4a7a,
            transparent: true,
            opacity: 0.4,
        });

        const lineCorners = [
            [new THREE.Vector3(-HALF, 0.05, -HALF), new THREE.Vector3(HALF, 0.05, -HALF)],
            [new THREE.Vector3(HALF, 0.05, -HALF), new THREE.Vector3(HALF, 0.05, HALF)],
            [new THREE.Vector3(HALF, 0.05, HALF), new THREE.Vector3(-HALF, 0.05, HALF)],
            [new THREE.Vector3(-HALF, 0.05, HALF), new THREE.Vector3(-HALF, 0.05, -HALF)],
        ];

        lineCorners.forEach((pts) => {
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const line = new THREE.Line(geo, lineMat);
            boundaryGroup.add(line);
        });

        scene.add(boundaryGroup);
    }

    function _onResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function render() {
        controls.update();
        renderer.render(scene, camera);
    }

    function getScene() { return scene; }
    function getCamera() { return camera; }

    return { init, render, getScene, getCamera, ARENA_SIZE, HALF };
})();
