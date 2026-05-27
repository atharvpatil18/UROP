import os
import json
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from src.ml.models import ISLSeq2SeqLSTM, PoseEncodedSignformer
from src.vision.normalizer import normalize_coordinates

app = FastAPI(title="Real-Time Indian Sign Language Translation System")

# Mount static files for the premium frontend dashboard
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

# Load models if weights are available, otherwise initialize with random weights
# Hidden dimension: 128, Classes: 26 (alphabet)
lstm_model = ISLSeq2SeqLSTM(input_dim=150, hidden_dim=128, num_classes=26)
if os.path.exists("models/isl_lstm_classifier.pth"):
    lstm_model.load_state_dict(torch.load("models/isl_lstm_classifier.pth", map_location=torch.device('cpu')))
lstm_model.eval()

# Alphabet mapping
CLASSES = [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Text-to-pose database generating smooth spatial trajectories for driven avatars
def generate_trajectory_for_word(word: str, seq_len: int = 40) -> list:
    """
    Synthesizes smooth, continuous sinusoidal coordinate trajectories of dimension 150
    representing word concepts. This provides the avatar drive sequences.
    """
    word = word.lower().strip()
    trajectory = []
    
    # Base skeleton posture
    base = np.zeros(150, dtype=np.float32)
    base[135:138] = [-0.25, 0.0, 0.0]  # Left shoulder
    base[138:141] = [0.25, 0.0, 0.0]   # Right shoulder
    base[147:150] = [0.0, 0.5, 0.0]    # Pelvis center
    
    # Unique movement signatures for different vocabulary words
    if "hello" in word:
        # High wave gesture with Right Hand
        for t in range(seq_len):
            frame = base.copy()
            amp = 0.15 * np.sin(np.pi * t / seq_len)
            # Right wrist wave
            frame[0:3] = [0.2, 0.2 + 0.1 * np.sin(0.3 * t), 0.1]
            # Hand fingers waving
            for i in range(1, 21):
                start = i * 3
                frame[start:start+3] = frame[0:3] + [0.01 * i, 0.02 * np.sin(0.5 * t + i), 0.01]
            trajectory.append(normalize_coordinates(frame).tolist())
            
    elif "thank" in word:
        # Chin to chest touch gesture with Right Hand
        for t in range(seq_len):
            frame = base.copy()
            # Move hand from chin (0, 0) out towards the camera/user
            progress = t / seq_len
            x = 0.0
            y = -0.1 - 0.2 * progress
            z = 0.2 - 0.1 * progress
            frame[0:3] = [x, y, z]
            for i in range(1, 21):
                start = i * 3
                frame[start:start+3] = frame[0:3] + [0.01 * i, -0.01 * i, 0.01]
            trajectory.append(normalize_coordinates(frame).tolist())
            
    elif "goodbye" in word or "bye" in word:
        # Horizontal waving hand gesture
        for t in range(seq_len):
            frame = base.copy()
            frame[0:3] = [0.15 + 0.08 * np.sin(0.4 * t), -0.1, 0.1]
            for i in range(1, 21):
                start = i * 3
                frame[start:start+3] = frame[0:3] + [0.01 * i, 0.01 * np.sin(0.4 * t + i), 0.01]
            trajectory.append(normalize_coordinates(frame).tolist())
            
    else:
        # Default smooth circle movement
        for t in range(seq_len):
            frame = base.copy()
            frame[0:3] = [0.15 + 0.05 * np.cos(0.2 * t), -0.2 + 0.05 * np.sin(0.2 * t), 0.1]
            for i in range(1, 21):
                start = i * 3
                frame[start:start+3] = frame[0:3] + [0.01 * i, 0.005 * i, 0.01]
            trajectory.append(normalize_coordinates(frame).tolist())
            
    return trajectory


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Renders dashboard UI by loading the static html page.
    """
    index_path = "src/app/static/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h3>Index.html not found. Please assemble Phase 5.</h3>", status_code=404)


@app.get("/api/text-to-pose")
async def text_to_pose(text: str):
    """
    Generates 3D avatar pose coordinate sequence from text input.
    """
    words = text.strip().split()
    sequence_map = {}
    for word in words:
        sequence_map[word] = generate_trajectory_for_word(word)
    return {
        "text": text,
        "sequences": sequence_map
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time high-speed bidirectional WebSockets communicating
    lightweight normalized coordinate arrays (vector dimension 150).
    """
    await websocket.accept()
    print("WebSocket client connected.")
    
    # Store sliding window of frames for continuous translation
    frame_window = []
    window_max_len = 30
    
    try:
        while True:
            # Receive coordinate array from client browser
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            
            # Coordinate representation vector: list of 150 floats
            if isinstance(data, list) and len(data) == 150:
                # Apply normalization server-side as backup and validation
                normalized = normalize_coordinates(np.array(data, dtype=np.float32))
                
                # Append to sliding window
                frame_window.append(normalized)
                if len(frame_window) > window_max_len:
                    frame_window.pop(0)
                    
                # Run classification on sliding window sequence
                if len(frame_window) >= 10:
                    seq_tensor = torch.tensor([frame_window], dtype=torch.float32)
                    
                    with torch.no_grad():
                        logits = lstm_model(seq_tensor)
                        probs = torch.softmax(logits, dim=1)
                        confidence, pred_idx = torch.max(probs, 1)
                        
                        pred_char = CLASSES[pred_idx.item()]
                        conf_val = confidence.item()
                        
                        # Send prediction result if confidence passes threshold
                        if conf_val > 0.45:
                            await websocket.send_json({
                                "type": "translation",
                                "text": pred_char,
                                "confidence": conf_val
                            })
                            
            elif isinstance(data, dict) and data.get("type") == "text_to_pose_request":
                # Handle text-to-pose request via WebSockets
                text_input = data.get("text", "")
                words = text_input.strip().split()
                sequence_map = {}
                for word in words:
                    sequence_map[word] = generate_trajectory_for_word(word)
                    
                await websocket.send_json({
                    "type": "avatar_trajectory",
                    "text": text_input,
                    "sequences": sequence_map
                })
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket execution error: {e}")
