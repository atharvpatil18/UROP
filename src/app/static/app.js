/*
   ISL SignFlow 2.0 Client-Side Engine
   WebGL & High-Performance Pure Canvas 3D Perspective Renderer
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
const simButtons = document.querySelectorAll('.btn-sim');

// Telemetry & FPS Counters
let lastFrameTime = performance.now();
let frameCount = 0;
let bytesSent = 0;
let lastTelemetryTime = performance.now();

// Fallback High-Performance 3D Perspective Canvas Engine
const fallbackCanvas = document.getElementById('fallback-canvas-3d');
const fallbackCtx = fallbackCanvas.getContext('2d');
let rotationAngleY = -0.3; // Default perspective rotation angle
let rotationAngleX = 0.1;
let scaleFactor3D = 1.0;
let isDragging = false;
let previousMouseX, previousMouseY;

// Holds current active joint coordinates (150-D format) for the renderer
let activeAvatarCoords = null;

// Initialize perspective canvas mouse rotations
function initFallbackRenderer() {
    const resizeCanvas = () => {
        fallbackCanvas.width = fallbackCanvas.parentElement.clientWidth;
        fallbackCanvas.height = fallbackCanvas.parentElement.clientHeight;
        if (activeAvatarCoords) drawFallbackAvatar(activeAvatarCoords);
    };
    
    // Bind mouse listeners for 3D orbital camera dragging
    fallbackCanvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        previousMouseX = e.clientX;
        previousMouseY = e.clientY;
    });
    
    window.addEventListener('mouseup', () => { isDragging = false; });
    
    fallbackCanvas.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const deltaX = e.clientX - previousMouseX;
        const deltaY = e.clientY - previousMouseY;
        
        rotationAngleY += deltaX * 0.007;
        rotationAngleX += deltaY * 0.007;
        
        previousMouseX = e.clientX;
        previousMouseY = e.clientY;
        
        if (activeAvatarCoords) drawFallbackAvatar(activeAvatarCoords);
    });
    
    // Zoom handling
    fallbackCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        scaleFactor3D += e.deltaY * -0.001;
        scaleFactor3D = Math.min(Math.max(0.5, scaleFactor3D), 2.5);
        if (activeAvatarCoords) drawFallbackAvatar(activeAvatarCoords);
    }, { passive: false });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // Start continuous rendering loop
    setInterval(() => {
        if (activeAvatarCoords) drawFallbackAvatar(activeAvatarCoords);
    }, 30);
}

// Generate default clean T-pose coordinates vector
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
    coords[147] = 0.0; coords[148] = -0.45; coords[149] = 0.0;     // Pelvis
    
    // Left Hand wrist
    coords[63] = -0.6; coords[64] = 0.15; coords[65] = 0.0;
    // Left Hand fingers extending horizontally
    for (let i = 1; i < 21; i++) {
        const start = 63 + i * 3;
        coords[start] = -0.6 - 0.04 * (i % 5) - 0.02 * Math.floor(i / 5);
        coords[start+1] = 0.15 + 0.005 * (i % 5);
        coords[start+2] = 0.0;
    }
    
    // Right Hand wrist
    coords[0] = 0.6; coords[1] = 0.15; coords[2] = 0.0;
    // Right Hand fingers extending horizontally
    for (let i = 1; i < 21; i++) {
        const start = i * 3;
        coords[start] = 0.6 + 0.04 * (i % 5) + 0.02 * Math.floor(i / 5);
        coords[start+1] = 0.15 + 0.005 * (i % 5);
        coords[start+2] = 0.0;
    }
    
    return coords;
}
const T_POSE_COORDS = getTPoseCoordinates();

// 3D Perspective Projection Mathematics
function project3D(x, y, z, width, height) {
    // 1. Orbital Camera rotation around Y-axis (Yaw)
    let x1 = x * Math.cos(rotationAngleY) - z * Math.sin(rotationAngleY);
    let z1 = x * Math.sin(rotationAngleY) + z * Math.cos(rotationAngleY);
    
    // 2. Camera rotation around X-axis (Pitch)
    let y2 = y * Math.cos(rotationAngleX) - z1 * Math.sin(rotationAngleX);
    let z2 = y * Math.sin(rotationAngleX) + z1 * Math.cos(rotationAngleX);
    
    // 3. Perspective Projection
    const distance = 2.0; // Camera distance
    const fov = 400 * scaleFactor3D;
    const scale = fov / (z2 + distance);
    
    const projX = width / 2 + x1 * scale;
    const projY = height / 2 + y2 * scale;
    
    return { x: projX, y: projY, z: z2 };
}

// Draw T-pose or animated sign language skeleton inside local HTML5 Canvas
function drawFallbackAvatar(coords) {
    const width = fallbackCanvas.width;
    const height = fallbackCanvas.height;
    
    // Clear and draw glowing futuristic space backdrop
    fallbackCtx.fillStyle = '#070513';
    fallbackCtx.fillRect(0, 0, width, height);
    
    // Ambient orbital gird grids for space depth
    fallbackCtx.strokeStyle = 'rgba(168, 85, 247, 0.03)';
    fallbackCtx.lineWidth = 1;
    for (let i = 0; i < width; i += 40) {
        fallbackCtx.beginPath();
        fallbackCtx.moveTo(i, 0);
        fallbackCtx.lineTo(i, height);
        fallbackCtx.stroke();
    }
    
    // Kinematic joint mappings
    const hasPose = coords.slice(126, 150).some(v => v !== 0);
    const poseData = hasPose ? coords : T_POSE_COORDS;
    
    const hasRHand = coords.slice(0, 63).some(v => v !== 0);
    const rHandData = hasRHand ? coords : T_POSE_COORDS;
    
    const hasLHand = coords.slice(63, 126).some(v => v !== 0);
    const lHandData = hasLHand ? coords : T_POSE_COORDS;
    
    // 3D coordinate map dictionary
    const points3D = {};
    
    // Project Upper Body Pose
    const poseNames = ['nose', 'l_eye', 'r_eye', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'pelvis'];
    poseNames.forEach((name, idx) => {
        const start = 126 + idx * 3;
        const x = poseData[start] * 0.9;
        const y = -poseData[start+1] * 0.9 + 0.15; // align relative to chest
        const z = -poseData[start+2] * 0.9;
        points3D[name] = project3D(x, y, z, width, height);
    });
    
    // Kinematically chain Right Wrist to Right Elbow
    const rElbow = poseData[144:147]; // elbow coordinates
    const rx = (rElbow[0] + (hasRHand ? rHandData[0] * 0.35 : 0.15)) * 0.9;
    const ry = -(rElbow[1] + (hasRHand ? -rHandData[1] * 0.35 : 0.0)) * 0.9 + 0.15;
    const rz = -(rElbow[2] + (hasRHand ? -rHandData[2] * 0.35 : 0.0)) * 0.9;
    points3D['R_0'] = project3D(rx, ry, rz, width, height);
    
    // Project Right fingers relative to Right Wrist
    for (let i = 1; i < 21; i++) {
        const start = i * 3;
        const fx = rx + rHandData[start] * 0.3;
        const fy = ry - rHandData[start+1] * 0.3;
        const fz = rz - rHandData[start+2] * 0.3;
        points3D[`R_${i}`] = project3D(fx, fy, fz, width, height);
    }
    
    // Kinematically chain Left Wrist to Left Elbow
    const lElbow = poseData[141:144];
    const lx = (lElbow[0] + (hasLHand ? lHandData[63] * 0.35 : -0.15)) * 0.9;
    const ly = -(lElbow[1] + (hasLHand ? -lHandData[64] * 0.35 : 0.0)) * 0.9 + 0.15;
    const lz = -(lElbow[2] + (hasLHand ? -lHandData[65] * 0.35 : 0.0)) * 0.9;
    points3D['L_0'] = project3D(lx, ly, lz, width, height);
    
    // Project Left fingers relative to Left Wrist
    for (let i = 1; i < 21; i++) {
        const start = 63 + i * 3;
        const fx = lx + lHandData[start] * 0.3;
        const fy = ly - lHandData[start+1] * 0.3;
        const fz = lz - lHandData[start+2] * 0.3;
        points3D[`L_${i}`] = project3D(fx, fy, fz, width, height);
    }
    
    // Head projection offset from nose
    points3D['head'] = { x: points3D['nose'].x, y: points3D['nose'].y - 15, z: points3D['nose'].z };
    
    // Render Connections (Bones)
    const bonePairs = [
        ['l_shoulder', 'r_shoulder', 'rgba(99, 102, 241, 0.85)'],
        ['l_shoulder', 'l_elbow', 'rgba(168, 85, 247, 0.85)'],
        ['r_shoulder', 'r_elbow', 'rgba(168, 85, 247, 0.85)'],
        ['l_shoulder', 'pelvis', 'rgba(99, 102, 241, 0.45)'],
        ['r_shoulder', 'pelvis', 'rgba(99, 102, 241, 0.45)'],
        ['l_elbow', 'L_0', 'rgba(168, 85, 247, 0.85)'],
        ['r_elbow', 'R_0', 'rgba(168, 85, 247, 0.85)'],
        ['nose', 'head', 'rgba(99, 102, 241, 0.85)']
    ];
    
    // Add finger links
    for (let h of ['L', 'R']) {
        const color = h === 'R' ? 'rgba(34, 197, 94, 0.8)' : 'rgba(234, 179, 8, 0.8)';
        const fingerBones = [
            [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],     // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
        ];
        fingerBones.forEach(pair => {
            bonePairs.push([`${h}_${pair[0]}`, `${h}_${pair[1]}`, color]);
        });
    }
    
    // Render lines (bones) with soft neon glow filters
    bonePairs.forEach(bone => {
        const p1 = points3D[bone[0]];
        const p2 = points3D[bone[1]];
        if (p1 && p2) {
            fallbackCtx.beginPath();
            fallbackCtx.moveTo(p1.x, p1.y);
            fallbackCtx.lineTo(p2.x, p2.y);
            fallbackCtx.strokeStyle = bone[2];
            fallbackCtx.lineWidth = bone[0].includes('R') || bone[0].includes('L') ? 2 : 4;
            fallbackCtx.lineCap = 'round';
            // Add subtle neon shadow glow
            fallbackCtx.shadowBlur = 8;
            fallbackCtx.shadowColor = bone[2];
            fallbackCtx.stroke();
            fallbackCtx.shadowBlur = 0; // Reset
        }
    });
    
    // Render spherical joints
    for (let name in points3D) {
        const pt = points3D[name];
        let color = '#a855f7'; // Purple pose joints
        let radius = 5;
        
        if (name.startsWith('R_')) {
            color = '#22c55e'; // Green right hand
            radius = 3;
        } else if (name.startsWith('L_')) {
            color = '#eab308'; // Yellow left hand
            radius = 3;
        } else if (name === 'head') {
            color = '#6366f1'; // Blue head
            radius = 18;
        }
        
        fallbackCtx.beginPath();
        fallbackCtx.arc(pt.x, pt.y, radius, 0, 2*Math.PI);
        fallbackCtx.fillStyle = color;
        fallbackCtx.shadowBlur = 10;
        fallbackCtx.shadowColor = color;
        fallbackCtx.fill();
        fallbackCtx.shadowBlur = 0;
    }
}

// Drive T-Pose coordinate array as default avatar model on load
activeAvatarCoords = T_POSE_COORDS;

// Text-to-Speech Engine
function triggerClientTTS(text) {
    if (!autoTTS) return;
    
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    
    const voices = window.speechSynthesis.getVoices();
    const indVoice = voices.find(voice => voice.lang.includes('IN') || voice.name.includes('India'));
    if (indVoice) utterance.voice = indVoice;
    
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
            
            predictedLetterBox.innerText = letter;
            confidencePercent.innerText = `${conf}%`;
            confidenceFill.style.width = `${conf}%`;
            
            if (sentenceOutput.innerText === "Waiting for gesture inputs...") {
                sentenceOutput.innerText = "";
            }
            
            const currentSent = sentenceOutput.innerText;
            if (currentSent === "" || currentSent[currentSent.length - 1] !== letter) {
                sentenceOutput.innerText += letter;
                triggerClientTTS(letter);
            }
        } else if (data.type === "sentence") {
            const phrase = data.text.toUpperCase();
            const conf = Math.round(data.confidence * 100);
            
            predictedLetterBox.innerText = phrase;
            confidencePercent.innerText = `${conf}%`;
            confidenceFill.style.width = `${conf}%`;
            
            if (sentenceOutput.innerText === "Waiting for gesture inputs..." || sentenceOutput.innerText === "") {
                sentenceOutput.innerText = phrase;
            } else {
                const currentSent = sentenceOutput.innerText;
                if (!currentSent.endsWith(phrase)) {
                    sentenceOutput.innerText += " " + phrase;
                }
            }
            
            triggerClientTTS(phrase);
            
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

// Play avatar pose sequences
let animationQueue = [];
let animIntervalId = null;

function playPoseSequence(sequence) {
    if (animIntervalId) clearInterval(animIntervalId);
    
    let frameIdx = 0;
    animIntervalId = setInterval(() => {
        if (frameIdx >= sequence.length) {
            clearInterval(animIntervalId);
            animIntervalId = null;
            if (animationQueue.length > 0) {
                const nextSeq = animationQueue.shift();
                playPoseSequence(nextSeq);
            } else {
                // Smoothly return to T-Pose on finish
                activeAvatarCoords = T_POSE_COORDS;
            }
            return;
        }
        
        activeAvatarCoords = sequence[frameIdx];
        frameIdx++;
    }, 33); // ~30 FPS
}

// Local Relative Normalizer
function localRelativeNormalization(coords) {
    const normalized = [...coords];
    
    const lx = coords[135], ly = coords[136], lz = coords[137];
    const rx = coords[138], ry = coords[139], rz = coords[140];
    
    const dx = lx - rx, dy = ly - ry, dz = lz - rz;
    let d_scale = Math.sqrt(dx*dx + dy*dy + dz*dz);
    if (d_scale < 1e-5) d_scale = 1.0;
    
    const midX = (lx + rx) / 2.0;
    const midY = (ly + ry) / 2.0;
    const midZ = (lz + rz) / 2.0;
    
    // Right Hand (index 0)
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
    
    // Left Hand (index 63)
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
    
    // Upper Body Pose
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
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
        fpsCounter.innerText = `FPS: ${frameCount}`;
        frameCount = 0;
        lastFrameTime = now;
    }
    
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const rawCoords = new Array(150).fill(0.0);
    
    // 1. Right Hand landmarks (0:63)
    if (results.rightHandLandmarks) {
        results.rightHandLandmarks.forEach((lm, idx) => {
            const start = idx * 3;
            rawCoords[start] = lm.x;
            rawCoords[start+1] = lm.y;
            rawCoords[start+2] = lm.z;
            
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
    if (results.poseLandmarks) {
        const targetPoseIdx = [0, 2, 5, 11, 12, 13, 14, 23];
        targetPoseIdx.forEach((mpIdx, idx) => {
            const lm = results.poseLandmarks[mpIdx];
            if (lm) {
                const start = 126 + idx * 3;
                rawCoords[start] = lm.x;
                rawCoords[start+1] = lm.y;
                rawCoords[start+2] = lm.z;
                
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 4, 0, 2*Math.PI);
                canvasCtx.fillStyle = '#a855f7';
                canvasCtx.fill();
            }
        });
    }
    
    const normalized = localRelativeNormalization(rawCoords);
    
    // Update active coordinates in rendering loop
    activeAvatarCoords = normalized;
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        const payloadStr = JSON.stringify(normalized);
        socket.send(payloadStr);
        
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
    
    try {
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
    } catch (e) {
        console.error("Camera/MediaPipe failed to start:", e);
        wsStatusElement.innerHTML = '<span class="status-dot disconnected"></span> MediaPipe failed to load.';
    }
}

function stopWebcamInference() {
    if (camera) camera.stop();
    if (holistic) holistic.close();
    
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    cameraPrompt.style.display = 'flex';
    setTimeout(() => { cameraPrompt.style.opacity = '1'; }, 50);
    
    activeWebcam = false;
    btnWebcam.innerText = "Start Webcam";
    btnWebcam.classList.replace('btn-secondary', 'btn-primary');
    activeAvatarCoords = T_POSE_COORDS;
}

// Dynamic high-fidelity simulated coordinate trajectory streams
// Maps specific continuous phrase concepts and streams their coordinate vectors
function simulateContinuousGesture(word) {
    console.log(`Streaming simulated gesture for phrase: '${word}'`);
    
    // Unlock client TTS on interaction
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    
    const seqLen = 45;
    const mockTrajectory = [];
    const base = np_zeros_coords();
    
    let freq = 0.12;
    for (let t = 0; t < seqLen; t++) {
        const frame = [...base];
        
        if (word === "hello") {
            // Wavy vertical hand coordinate movements
            frame[0] = 0.2 + 0.15 * Math.sin(0.3 * t);  // wrist x
            frame[1] = 0.1 + 0.22 * Math.sin(freq * t); // wrist y (large vertical waving)
            frame[2] = 0.0;
            // fingers extended
            for (let i = 1; i < 21; i++) {
                const start = i * 3;
                frame[start] = frame[0] + 0.02 * (i % 5);
                frame[start+1] = frame[1] + 0.05 + 0.01 * (i % 5);
                frame[start+2] = 0.0;
            }
        } else if (word === "thank you") {
            // Hand moving from chin (low y) out towards the screen (vertical drop)
            const progress = t / seqLen;
            frame[0] = 0.0;
            frame[1] = -0.1 - 0.25 * progress; // wrist y drops downwards
            frame[2] = 0.1 - 0.1 * progress;
            // flat hand
            for (let i = 1; i < 21; i++) {
                const start = i * 3;
                frame[start] = frame[0] + 0.02 * (i % 5);
                frame[start+1] = frame[1] + 0.04;
                frame[start+2] = 0.0;
            }
        } else if (word === "goodbye") {
            // Horizontal periodic wave
            frame[0] = 0.18 + 0.24 * Math.sin(0.4 * t); // wrist x sweeps horizontally
            frame[1] = 0.1;
            frame[2] = 0.0;
            for (let i = 1; i < 21; i++) {
                const start = i * 3;
                frame[start] = frame[0] + 0.03 * (i % 5);
                frame[start+1] = frame[1] + 0.04;
                frame[start+2] = 0.0;
            }
        } else if (word === "how are you") {
            // Circular movement profile (correlated orbital x and y)
            frame[0] = 0.15 * Math.cos(0.2 * t); // circular x
            frame[1] = 0.15 * Math.sin(0.2 * t); // circular y
            frame[2] = 0.0;
            for (let i = 1; i < 21; i++) {
                const start = i * 3;
                frame[start] = frame[0] + 0.02 * (i % 5);
                frame[start+1] = frame[1] + 0.04;
                frame[start+2] = 0.0;
            }
        }
        
        mockTrajectory.append = localRelativeNormalization(frame);
        mockTrajectory.push(localRelativeNormalization(frame));
    }
    
    // Play gesture animation on avatar
    playPoseSequence(mockTrajectory);
    
    // Stream coordinates over active WebSocket to show server sequence metrics
    if (socket && socket.readyState === WebSocket.OPEN) {
        let frameIdx = 0;
        const streamInterval = setInterval(() => {
            if (frameIdx >= mockTrajectory.length) {
                clearInterval(streamInterval);
                return;
            }
            socket.send(JSON.stringify(mockTrajectory[frameIdx]));
            frameIdx++;
        }, 33);
    } else {
        // Fallback local UI simulation output if socket offline
        setTimeout(() => {
            const phraseCaps = word.toUpperCase();
            predictedLetterBox.innerText = phraseCaps;
            confidencePercent.innerText = "95%";
            confidenceFill.style.width = "95%";
            
            if (sentenceOutput.innerText === "Waiting for gesture inputs...") {
                sentenceOutput.innerText = phraseCaps;
            } else {
                sentenceOutput.innerText += " " + phraseCaps;
            }
            triggerClientTTS(phraseCaps);
        }, 1200);
    }
}

function np_zeros_coords() {
    const coords = new Array(150).fill(0.0);
    coords[135] = -0.22; coords[136] = 0.15; coords[137] = 0.0;    // L Shoulder
    coords[138] = 0.22; coords[139] = 0.15; coords[140] = 0.0;     // R Shoulder
    coords[147] = 0.0; coords[148] = -0.45; coords[149] = 0.0;     // Pelvis
    return coords;
}

// Event Listeners
btnWebcam.addEventListener('click', () => {
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
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
    }
});

// Text-To-Pose Synthesis
btnSynthesize.addEventListener('click', () => {
    const text = synthesisInput.value.trim();
    if (text === "") return;
    
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            "type": "text_to_pose_request",
            "text": text
        }));
    } else {
        // Fallback fetch API
        fetch(`/api/text-to-pose?text=${encodeURIComponent(text)}`)
            .then(res => res.json())
            .then(data => {
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
    if (e.key === 'Enter') btnSynthesize.click();
});

// Bind simulated gesture triggers
simButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const word = btn.getAttribute('data-word');
        simulateContinuousGesture(word);
    });
});

// Bootstrapping
window.addEventListener('DOMContentLoaded', () => {
    initFallbackRenderer();
    connectWebSocket();
    window.speechSynthesis.getVoices();
});
