/*
   ISL SignFlow 2.0 Client-Side Engine
   Webcam Ingestion -> Local MediaPipe Holistic -> 150-D Vector Stream -> WebSocket -> 3D Avatar Drive & Web Speech TTS
*/

let socket = null;
let camera = null;
let holistic = null;
let activeWebcam = false;
let autoTTS = true;
let socketUrl = `ws://${window.location.host}/ws`;

// HTML Elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('landmark-overlay');
const canvasCtx = canvasElement.getContext('2d');
const btnWebcam = document.getElementById('btn-webcam');
const btnTtsToggle = document.getElementById('btn-tts-toggle');
const wsStatusElement = document.getElementById('ws-status');
const fpsCounter = document.getElementById('fps-counter');
const payloadMetric = document.getElementById('payload-metric');
const predictedLetterBox = document.getElementById('predicted-letter');
const confidenceFill = document.getElementById('confidence-fill');
const confidencePercent = document.getElementById('confidence-percentage');
const sentenceOutput = document.getElementById('sentence-output');
const btnClearSentence = document.getElementById('btn-clear-sentence');
const btnSpeakSentence = document.getElementById('btn-speak-sentence');
const synthesisInput = document.getElementById('synthesis-input');
const btnSynthesize = document.getElementById('btn-synthesize');
const cameraPrompt = document.getElementById('camera-prompt');

// Telemetry & FPS Counters
let lastFrameTime = performance.now();
let frameCount = 0;
let bytesSent = 0;
let lastTelemetryTime = performance.now();

// Three.js 3D Rigged Humanoid Avatar Setup
const avatarContainer = document.getElementById('avatar-container');
let scene, camera3D, renderer;
let joints = {}; // Holds 3D mesh points for skeletal rigging
let bones = [];   // Holds line meshes representing body links

// Generate default clean T-pose coordinates vector to initialize avatar and handle active tracking drops
function getTPoseCoordinates() {
    const coords = new Array(150).fill(0.0);
    // Upper body pose
    coords[126] = 0.0; coords[127] = 0.35; coords[128] = 0.0;     // Nose
    coords[129] = -0.05; coords[130] = 0.37; coords[131] = 0.05;   // L Eye
    coords[132] = 0.05; coords[133] = 0.37; coords[134] = 0.05;    // R Eye
    coords[135] = -0.22; coords[136] = 0.15; coords[137] = 0.0;    // L Shoulder
    coords[138] = 0.22; coords[139] = 0.15; coords[140] = 0.0;     // R Shoulder
    coords[141] = -0.42; coords[142] = 0.15; coords[143] = 0.0;    // L Elbow
    coords[144] = 0.42; coords[145] = 0.15; coords[146] = 0.0;     // R Elbow
    coords[147] = 0.0; coords[148] = -0.45; coords[149] = 0.0;     // Pelvis (bottom body anchor)
    
    // Left Hand wrist
    coords[63] = -0.6; coords[64] = 0.15; coords[65] = 0.0;
    // Left Hand fingers extending horizontally
    for (let i = 1; i < 21; i++) {
        const start = 63 + i * 3;
        coords[start] = -0.6 - 0.04 * (i % 5) - 0.02 * Math.floor(i / 5);
        coords[start+1] = 0.15 + 0.01 * (i % 5);
        coords[start+2] = 0.0;
    }
    
    // Right Hand wrist
    coords[0] = 0.6; coords[1] = 0.15; coords[2] = 0.0;
    // Right Hand fingers extending horizontally
    for (let i = 1; i < 21; i++) {
        const start = i * 3;
        coords[start] = 0.6 + 0.04 * (i % 5) + 0.02 * Math.floor(i / 5);
        coords[start+1] = 0.15 + 0.01 * (i % 5);
        coords[start+2] = 0.0;
    }
    
    return coords;
}
const T_POSE_COORDS = getTPoseCoordinates();

function initThreeJS() {
    const width = avatarContainer.clientWidth;
    const height = avatarContainer.clientHeight;
    
    // Create WebGL Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070513); // Deep dark purple
    
    // Ambient and directional lighting for premium spatial atmosphere
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xa855f7, 0.8);
    dirLight.position.set(0, 5, 5);
    scene.add(dirLight);
    
    const blueLight = new THREE.DirectionalLight(0x6366f1, 0.6);
    blueLight.position.set(-5, 3, -2);
    scene.add(blueLight);
    
    // Camera
    camera3D = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera3D.position.set(0, 0, 1.8);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    avatarContainer.appendChild(renderer.domElement);
    
    // Build rigged joints (spheres) with sleek glowing neon materials
    const jointMaterial = new THREE.MeshStandardMaterial({
        color: 0xa855f7,
        roughness: 0.1,
        metalness: 0.8,
        emissive: 0x581c87,
        emissiveIntensity: 0.5
    });
    
    const jointGeom = new THREE.SphereGeometry(0.015, 16, 16);
    const handJointGeom = new THREE.SphereGeometry(0.008, 8, 8);
    const headGeom = new THREE.SphereGeometry(0.08, 32, 32);
    
    // Head / Face center
    joints['head'] = new THREE.Mesh(headGeom, new THREE.MeshStandardMaterial({
        color: 0x6366f1,
        roughness: 0.2,
        emissive: 0x312e81,
        emissiveIntensity: 0.3
    }));
    joints['head'].position.set(0, 0.35, 0);
    scene.add(joints['head']);
    
    // Initialize pose joints
    const poseJointNames = ['nose', 'l_eye', 'r_eye', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'pelvis'];
    poseJointNames.forEach(name => {
        joints[name] = new THREE.Mesh(jointGeom, jointMaterial);
        joints[name].position.set(0, 0, -10); // Hide initially
        scene.add(joints[name]);
    });
    
    // Initialize left and right hand joints
    for (let h of ['L', 'R']) {
        const material = new THREE.MeshStandardMaterial({
            color: h === 'R' ? 0x22c55e : 0xeab308,
            roughness: 0.2,
            metalness: 0.6,
            emissive: h === 'R' ? 0x14532d : 0x713f12,
            emissiveIntensity: 0.4
        });
        
        for (let i = 0; i < 21; i++) {
            const key = `${h}_${i}`;
            joints[key] = new THREE.Mesh(handJointGeom, material);
            joints[key].position.set(0, 0, -10); // Hide initially
            scene.add(joints[key]);
        }
    }
    
    // Create aesthetic connections (bones) using line segments
    const boneMaterial = new THREE.LineBasicMaterial({
        color: 0x4f46e5,
        linewidth: 2,
        transparent: true,
        opacity: 0.7
    });
    
    // Helper to draw bone lines
    const bonePairs = [
        ['l_shoulder', 'r_shoulder'],
        ['l_shoulder', 'l_elbow'],
        ['r_shoulder', 'r_elbow'],
        ['l_shoulder', 'pelvis'],
        ['r_shoulder', 'pelvis']
    ];
    
    // Hand skeletal links (0 is wrist, fingers branch out: 1-4, 5-8, 9-12, 13-16, 17-20)
    for (let h of ['L', 'R']) {
        const fingerBones = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],     // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
        ];
        
        fingerBones.forEach(pair => {
            bonePairs.push([`${h}_${pair[0]}`, `${h}_${pair[1]}`]);
        });
    }
    
    // Connect wrist to elbows
    bonePairs.push(['l_elbow', 'L_0']);
    bonePairs.push(['r_elbow', 'R_0']);
    
    // Instantiate line meshes
    bonePairs.forEach(pair => {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, boneMaterial);
        line.userData = { start: pair[0], end: pair[1] };
        scene.add(line);
        bones.push(line);
    });
    
    // Load and drive avatar directly into beautiful initial T-Pose
    driveAvatarWithCoordinates(T_POSE_COORDS);
    
    // Start Three.js Animation Loop
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    
    // Animate lines to follow the spherical joints dynamically
    bones.forEach(bone => {
        const startJoint = joints[bone.userData.start];
        const endJoint = joints[bone.userData.end];
        
        if (startJoint && endJoint && startJoint.position.z > -5 && endJoint.position.z > -5) {
            const positions = bone.geometry.attributes.position.array;
            positions[0] = startJoint.position.x;
            positions[1] = startJoint.position.y;
            positions[2] = startJoint.position.z;
            positions[3] = endJoint.position.x;
            positions[4] = endJoint.position.y;
            positions[5] = endJoint.position.z;
            bone.geometry.attributes.position.needsUpdate = true;
            bone.visible = true;
        } else {
            bone.visible = false;
        }
    });
    
    renderer.render(scene, camera3D);
}

// Window resizing for Three.js WebGL canvas
window.addEventListener('resize', () => {
    if (!renderer) return;
    const width = avatarContainer.clientWidth;
    const height = avatarContainer.clientHeight;
    camera3D.aspect = width / height;
    camera3D.updateProjectionMatrix();
    renderer.setSize(width, height);
});

// Map 150-D Coordinate Array to Rigged joints with smart T-pose fallbacks for untracked sections
let lerpWeight = 0.35; // Position smoothing factor (lerping reduces noise)
function driveAvatarWithCoordinates(coords) {
    // 150-D vector extraction matching our schema:
    // [0:63] -> R Hand, [63:126] -> L Hand, [126:150] -> Pose
    
    // Offset coordinates for aesthetic positioning within screen bounds
    const offsetZ = -0.5; // push avatar slightly back
    const scaleFactor = 1.6;
    
    // 1. Right Hand (21 points) - fallback to T-pose if right hand landmarks are active 0
    const hasRHand = coords.slice(0, 63).some(v => v !== 0);
    const rHandData = hasRHand ? coords : T_POSE_COORDS;
    
    for (let i = 0; i < 21; i++) {
        const key = `R_${i}`;
        const start = i * 3;
        
        // Scale and mirror coordinates
        const tx = rHandData[start] * scaleFactor;
        const ty = -rHandData[start+1] * scaleFactor + 0.3; // align relative to torso
        const tz = -rHandData[start+2] * scaleFactor + offsetZ;
        
        // Apply smoothing interpolation
        joints[key].position.lerp(new THREE.Vector3(tx, ty, tz), lerpWeight);
    }
    
    // 2. Left Hand (21 points) - fallback to T-pose if left hand landmarks are active 0
    const hasLHand = coords.slice(63, 126).some(v => v !== 0);
    const lHandData = hasLHand ? coords : T_POSE_COORDS;
    
    for (let i = 0; i < 21; i++) {
        const key = `L_${i}`;
        const start = 63 + i * 3;
        
        const tx = lHandData[start] * scaleFactor;
        const ty = -lHandData[start+1] * scaleFactor + 0.3;
        const tz = -lHandData[start+2] * scaleFactor + offsetZ;
        
        joints[key].position.lerp(new THREE.Vector3(tx, ty, tz), lerpWeight);
    }
    
    // 3. Upper Body Pose (8 joints: nose, l_eye, r_eye, l_shoulder, r_shoulder, l_elbow, r_elbow, pelvis)
    const hasPose = coords.slice(126, 150).some(v => v !== 0);
    const poseData = hasPose ? coords : T_POSE_COORDS;
    
    const poseNames = ['nose', 'l_eye', 'r_eye', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'pelvis'];
    poseNames.forEach((name, idx) => {
        const start = 126 + idx * 3;
        const tx = poseData[start] * scaleFactor;
        const ty = -poseData[start+1] * scaleFactor + 0.3;
        const tz = -poseData[start+2] * scaleFactor + offsetZ;
        
        joints[name].position.lerp(new THREE.Vector3(tx, ty, tz), lerpWeight);
    });
    
    // Move Head mesh based on Nose coordinate
    if (joints['nose'].position.z > -5) {
        joints['head'].position.copy(joints['nose'].position);
        joints['head'].position.y += 0.08; // offset head mesh upwards
    }
}

// Play Avatar Sign Gesture Animation sequence (Text-To-Pose playback)
let animationQueue = [];
let animIntervalId = null;

function playPoseSequence(sequence) {
    // Clear existing animation playback
    if (animIntervalId) {
        clearInterval(animIntervalId);
    }
    
    let frameIdx = 0;
    animIntervalId = setInterval(() => {
        if (frameIdx >= sequence.length) {
            clearInterval(animIntervalId);
            animIntervalId = null;
            // Play next word in queue if available
            if (animationQueue.length > 0) {
                const nextSeq = animationQueue.shift();
                playPoseSequence(nextSeq);
            }
            return;
        }
        
        const frameCoords = sequence[frameIdx];
        driveAvatarWithCoordinates(frameCoords);
        frameIdx++;
    }, 33); // Play at ~30 FPS
}

// Client-Side Web Speech API Text-to-Speech Engine (Ultra-Low Latency & Server Offloaded)
function triggerClientTTS(text) {
    if (!autoTTS) return;
    
    // Cancel active utterances
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    // Attempt to select a premium Indian voice if available
    const voices = window.speechSynthesis.getVoices();
    const indVoice = voices.find(voice => voice.lang.includes('IN') || voice.name.includes('India'));
    if (indVoice) {
        utterance.voice = indVoice;
    }
    
    window.speechSynthesis.speak(utterance);
}

// WebSocket Connection Setup
function connectWebSocket() {
    socket = new WebSocket(socketUrl);
    
    socket.onopen = () => {
        console.log("WebSocket connected.");
        wsStatusElement.innerHTML = '<span class="status-dot connected"></span> WebSocket: Connected';
    };
    
    socket.onclose = () => {
        console.log("WebSocket disconnected. Attempting reconnect in 4s...");
        wsStatusElement.innerHTML = '<span class="status-dot disconnected"></span> WebSocket: Disconnected';
        setTimeout(connectWebSocket, 4000);
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === "translation") {
            const letter = data.text.toUpperCase();
            const conf = Math.round(data.confidence * 100);
            
            // Display prediction and confidence
            predictedLetterBox.innerText = letter;
            confidencePercent.innerText = `${conf}%`;
            confidenceFill.style.width = `${conf}%`;
            
            // Update accumulated sentence logic
            if (sentenceOutput.innerText === "Waiting for gesture inputs...") {
                sentenceOutput.innerText = "";
            }
            
            // Append prediction (only append if it's different from the last letter, or handle space)
            const currentSent = sentenceOutput.innerText;
            if (currentSent === "" || currentSent[currentSent.length - 1] !== letter) {
                sentenceOutput.innerText += letter;
                // Speak predicted letter natively client-side
                triggerClientTTS(letter);
            }
        } else if (data.type === "avatar_trajectory") {
            console.log("Received WebSocket avatar trajectory payload.");
            animationQueue = [];
            for (let word in data.sequences) {
                animationQueue.push(data.sequences[word]);
            }
            if (animationQueue.length > 0) {
                playPoseSequence(animationQueue.shift());
            }
        }
    };
}

// Local Relative normalizer mirroring python normalizer for client-side processing
function localRelativeNormalization(coords) {
    const normalized = [...coords];
    
    // 8 upper body pose points starting at index 126
    // L Shoulder: 126 + 3*3 = 135
    // R Shoulder: 126 + 4*3 = 138
    const lx = coords[135], ly = coords[136], lz = coords[137];
    const rx = coords[138], ry = coords[139], rz = coords[140];
    
    // Calculate shoulder span
    const dx = lx - rx, dy = ly - ry, dz = lz - rz;
    let d_scale = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (d_scale < 1e-5) d_scale = 1.0;
    
    const midX = (lx + rx) / 2.0;
    const midY = (ly + ry) / 2.0;
    const midZ = (lz + rz) / 2.0;
    
    // 1. Right Hand relative anchoring to Right Wrist (index 0)
    const hasRHand = coords.slice(0, 63).some(v => v !== 0);
    if (hasRHand) {
        const wx = coords[0], wy = coords[1], wz = coords[2];
        for (let i = 0; i < 21; i++) {
            const idx = i * 3;
            normalized[idx] = (coords[idx] - wx) / d_scale;
            normalized[idx+1] = (coords[idx+1] - wy) / d_scale;
            normalized[idx+2] = (coords[idx+2] - wz) / d_scale;
        }
    }
    
    // 2. Left Hand relative anchoring to Left Wrist (index 63)
    const hasLHand = coords.slice(63, 126).some(v => v !== 0);
    if (hasLHand) {
        const wx = coords[63], wy = coords[64], wz = coords[65];
        for (let i = 0; i < 21; i++) {
            const idx = 63 + i * 3;
            normalized[idx] = (coords[idx] - wx) / d_scale;
            normalized[idx+1] = (coords[idx+1] - wy) / d_scale;
            normalized[idx+2] = (coords[idx+2] - wz) / d_scale;
        }
    }
    
    // 3. Upper Body Pose relative anchoring to mid-shoulder
    for (let i = 0; i < 8; i++) {
        const idx = 126 + i * 3;
        normalized[idx] = (coords[idx] - midX) / d_scale;
        normalized[idx+1] = (coords[idx+1] - midY) / d_scale;
        normalized[idx+2] = (coords[idx+2] - midZ) / d_scale;
    }
    
    return normalized;
}

// MediaPipe Holistic Ingestion & Webcam Pipeline
function onResults(results) {
    // Increment FPS counter
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
        fpsCounter.innerText = `FPS: ${frameCount}`;
        frameCount = 0;
        lastFrameTime = now;
    }
    
    // Clear landmark canvas overlay
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Build 150-D coordinate array
    const rawCoords = new Array(150).fill(0.0);
    
    // 1. Right Hand landmarks (0:63)
    if (results.rightHandLandmarks) {
        results.rightHandLandmarks.forEach((lm, idx) => {
            const start = idx * 3;
            rawCoords[start] = lm.x;
            rawCoords[start+1] = lm.y;
            rawCoords[start+2] = lm.z;
            
            // Draw visual hand circle on overlay canvas
            canvasCtx.beginPath();
            canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 3, 0, 2*Math.PI);
            canvasCtx.fillStyle = '#22c55e';
            canvasCtx.fill();
        });
    }
    
    // 2. Left Hand landmarks (63:126)
    if (results.leftHandLandmarks) {
        results.leftHandLandmarks.forEach((lm, idx) => {
            const start = 63 + idx * 3;
            rawCoords[start] = lm.x;
            rawCoords[start+1] = lm.y;
            rawCoords[start+2] = lm.z;
            
            canvasCtx.beginPath();
            canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 3, 0, 2*Math.PI);
            canvasCtx.fillStyle = '#eab308';
            canvasCtx.fill();
        });
    }
    
    // 3. Pose Upper Joints (126:150)
    // Map indices: 0:nose, 1:l_eye(2), 2:r_eye(5), 3:l_shoulder(11), 4:r_shoulder(12), 5:l_elbow(13), 6:r_elbow(14), 7:l_hip(23)
    if (results.poseLandmarks) {
        const targetPoseIdx = [0, 2, 5, 11, 12, 13, 14, 23];
        targetPoseIdx.forEach((mpIdx, idx) => {
            const lm = results.poseLandmarks[mpIdx];
            if (lm) {
                const start = 126 + idx * 3;
                rawCoords[start] = lm.x;
                rawCoords[start+1] = lm.y;
                rawCoords[start+2] = lm.z;
                
                // Draw pose marker
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 4, 0, 2*Math.PI);
                canvasCtx.fillStyle = '#a855f7';
                canvasCtx.fill();
            }
        });
    }
    
    // Apply client-side Relative Normalization
    const normalized = localRelativeNormalization(rawCoords);
    
    // Drive Three.js humanoid avatar with these local coordinates immediately (0 latency!)
    driveAvatarWithCoordinates(normalized);
    
    // Transmit coordinates over WebSocket to server
    if (socket && socket.readyState === WebSocket.OPEN) {
        const payloadStr = JSON.stringify(normalized);
        socket.send(payloadStr);
        
        // Track network payload size
        bytesSent += payloadStr.length;
        const timeNow = performance.now();
        if (timeNow - lastTelemetryTime >= 1000) {
            const kbRate = Math.round((bytesSent / 1024));
            payloadMetric.innerText = `Payload: ${kbRate} KB/s`;
            bytesSent = 0;
            lastTelemetryTime = timeNow;
        }
    }
}

// Initializing MediaPipe Holistic Ingest
function startWebcamInference() {
    cameraPrompt.style.opacity = '0';
    setTimeout(() => { cameraPrompt.style.display = 'none'; }, 500);
    
    holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    
    holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    holistic.onResults(onResults);
    
    camera = new Camera(webcamElement, {
        onFrame: async () => {
            canvasElement.width = webcamElement.videoWidth;
            canvasElement.height = webcamElement.videoHeight;
            await holistic.send({ image: webcamElement });
        },
        width: 640,
        height: 480
    });
    
    camera.start().then(() => {
        activeWebcam = true;
        btnWebcam.innerText = "Stop Webcam";
        btnWebcam.classList.replace('btn-primary', 'btn-secondary');
    });
}

function stopWebcamInference() {
    if (camera) {
        camera.stop();
    }
    if (holistic) {
        holistic.close();
    }
    
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    cameraPrompt.style.display = 'flex';
    setTimeout(() => { cameraPrompt.style.opacity = '1'; }, 50);
    
    activeWebcam = false;
    btnWebcam.innerText = "Start Webcam";
    btnWebcam.classList.replace('btn-secondary', 'btn-primary');
}

// Event Listeners
btnWebcam.addEventListener('click', () => {
    // Standard hack to register user interaction token and unlock SpeechSynthesis in Chrome/Safari
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    
    if (activeWebcam) {
        stopWebcamInference();
    } else {
        startWebcamInference();
    }
});

btnTtsToggle.addEventListener('click', () => {
    autoTTS = !autoTTS;
    btnTtsToggle.innerText = `Auto-TTS: ${autoTTS ? 'ON' : 'OFF'}`;
    btnTtsToggle.classList.toggle('btn-secondary');
    btnTtsToggle.classList.toggle('btn-primary');
});

btnClearSentence.addEventListener('click', () => {
    sentenceOutput.innerText = "Waiting for gesture inputs...";
    predictedLetterBox.innerText = "-";
    confidenceFill.style.width = "0%";
    confidencePercent.innerText = "0%";
});

btnSpeakSentence.addEventListener('click', () => {
    const text = sentenceOutput.innerText;
    if (text !== "" && text !== "Waiting for gesture inputs...") {
        // Trigger browser speech synthesis of the sentence
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
});

// Text-To-Pose Generative Synthesis Trigger
btnSynthesize.addEventListener('click', () => {
    const text = synthesisInput.value.trim();
    if (text === "") return;
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        // Send request via WebSocket
        socket.send(JSON.stringify({
            "type": "text_to_pose_request",
            "text": text
        }));
    } else {
        // Fallback to fetch API
        fetch(`/api/text-to-pose?text=${encodeURIComponent(text)}`)
            .then(res => res.json())
            .then(data => {
                console.log("Received fetch API text-to-pose response:", data);
                animationQueue = [];
                for (let word in data.sequences) {
                    animationQueue.push(data.sequences[word]);
                }
                if (animationQueue.length > 0) {
                    playPoseSequence(animationQueue.shift());
                }
            });
    }
});

synthesisInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        btnSynthesize.click();
    }
});

// Bootstrapping
window.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    connectWebSocket();
    // Warm up native speech synthesis voices array
    window.speechSynthesis.getVoices();
});
