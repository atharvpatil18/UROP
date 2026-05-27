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


def classify_gesture_geometrically(coords: np.ndarray) -> tuple:
    """
    Heuristic geometric classifier analyzing finger extension vectors to recognize
    alphabet gestures in real-time. This provides high-fidelity, zero-lag tracking.
    Returns: (predicted_char, confidence)
    """
    # 1. Check if Right Hand is active (using Wrist position and coordinates)
    r_hand = coords[0:63]
    if not np.any(r_hand != 0.0):
        return None, 0.0
        
    # Extract joint keypoints for distance calculation
    # Right hand indices: 0: Wrist, 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
    # Knuckles: 2: Thumb base, 5: Index knuckle, 9: Middle knuckle, 13: Ring knuckle, 17: Pinky knuckle
    wrist = r_hand[0:3]
    
    def get_joint(idx):
        return r_hand[idx*3 : idx*3 + 3]
        
    # Tips
    t_tip = get_joint(4)
    i_tip = get_joint(8)
    m_tip = get_joint(12)
    r_tip = get_joint(16)
    p_tip = get_joint(20)
    
    # Knuckles / Bases
    t_base = get_joint(2)
    i_knuckle = get_joint(5)
    m_knuckle = get_joint(9)
    r_knuckle = get_joint(13)
    p_knuckle = get_joint(17)
    
    # Calculate Euclidean distances from wrist
    d_wrist_i_tip = np.linalg.norm(i_tip - wrist)
    d_wrist_i_knuckle = np.linalg.norm(i_knuckle - wrist)
    
    d_wrist_m_tip = np.linalg.norm(m_tip - wrist)
    d_wrist_m_knuckle = np.linalg.norm(m_knuckle - wrist)
    
    d_wrist_r_tip = np.linalg.norm(r_tip - wrist)
    d_wrist_r_knuckle = np.linalg.norm(r_knuckle - wrist)
    
    d_wrist_p_tip = np.linalg.norm(p_tip - wrist)
    d_wrist_p_knuckle = np.linalg.norm(p_knuckle - wrist)
    
    d_wrist_t_tip = np.linalg.norm(t_tip - wrist)
    d_wrist_t_base = np.linalg.norm(t_base - wrist)
    
    # Binarize finger extensions (extended = Tip is further from wrist than Knuckle)
    i_open = d_wrist_i_tip > d_wrist_i_knuckle + 0.05
    m_open = d_wrist_m_tip > d_wrist_m_knuckle + 0.05
    r_open = d_wrist_r_tip > d_wrist_r_knuckle + 0.05
    p_open = d_wrist_p_tip > d_wrist_p_knuckle + 0.05
    t_open = d_wrist_t_tip > d_wrist_t_base + 0.03
    
    # Count open fingers
    open_count = sum([i_open, m_open, r_open, p_open])
    
    # Heuristic matching mapping finger combinations to alphabet concepts
    if open_count == 0 and not t_open:
        return "A", 0.90  # Fist
    elif open_count == 4 and t_open:
        return "B", 0.95  # All fingers extended
    elif i_open and not m_open and not r_open and not p_open:
        if t_open:
            return "L", 0.90  # Index and Thumb (L shape)
        return "D", 0.95  # Only Index (Point/D)
    elif i_open and m_open and not r_open and not p_open:
        return "V", 0.95  # Peace / V Sign
    elif p_open and t_open and not i_open and not m_open and not r_open:
        return "Y", 0.95  # Shaka / Y Sign
    elif i_open and m_open and r_open and p_open and not t_open:
        return "F", 0.90  # B sign without Thumb
    elif i_open and m_open and r_open and not p_open:
        return "W", 0.90  # 3 / W Sign
    elif t_open and not i_open and not m_open and not r_open and not p_open:
        return "C", 0.75  # C shape
        
    return None, 0.0


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
                
                # First run heuristic geometric classifier
                geom_char, geom_conf = classify_gesture_geometrically(normalized)
                
                if geom_char:
                    await websocket.send_json({
                        "type": "translation",
                        "text": geom_char,
                        "confidence": geom_conf
                    })
                else:
                    # Fallback to LSTM sequence model
                    frame_window.append(normalized)
                    if len(frame_window) > window_max_len:
                        frame_window.pop(0)
                        
                    if len(frame_window) >= 10:
                        seq_tensor = torch.tensor([frame_window], dtype=torch.float32)
                        
                        with torch.no_grad():
                            logits = lstm_model(seq_tensor)
                            probs = torch.softmax(logits, dim=1)
                            confidence, pred_idx = torch.max(probs, 1)
                            
                            pred_char = CLASSES[pred_idx.item()]
                            conf_val = confidence.item()
                            
                            if conf_val > 0.40:  # Lower threshold slightly for better live sensitivity
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
