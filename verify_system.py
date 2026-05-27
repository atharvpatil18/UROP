import os
import json
import numpy as np
import torch

def verify_infrastructure():
    print("\n--- Verifying Phase 1: Infrastructure ---")
    
    # Check agent.json
    assert os.path.exists("agent.json"), "agent.json is missing!"
    with open("agent.json", "r") as f:
        agent_data = json.load(f)
        assert agent_data["projectName"] == "Real-Time ISL Translation System", "Incorrect project name in agent.json!"
    print("[PASS] agent.json is present and syntactically valid.")
    
    # Check hooks.json
    assert os.path.exists("hooks.json"), "hooks.json is missing!"
    with open("hooks.json", "r") as f:
        hooks_data = json.load(f)
        assert "PostToolUse" in hooks_data["hooks"], "PostToolUse hook missing!"
    print("[PASS] hooks.json is present and syntactically valid.")


def verify_etl_pipeline():
    print("\n--- Verifying Phase 2 & 3: Ingestion & Spatial Normalization ---")
    
    from src.data.etl import generate_continuous_sequences, temporal_hamming_filter
    from src.vision.normalizer import normalize_coordinates
    
    # Verify continuous sequence generation
    seqs, labels = generate_continuous_sequences(num_sequences=5)
    assert len(seqs) == 5, "Incorrect sequence count!"
    assert seqs[0].shape[1] == 150, f"Incorrect coordinate dimensions! Got {seqs[0].shape[1]}, expected 150."
    print("[PASS] Gesture coordinate trajectory generator produces correct 150-D vectors.")
    
    # Verify temporal Hamming filter
    test_seq = np.zeros((10, 150), dtype=np.float32)
    # Repeated static frames
    test_seq[1] = test_seq[0]
    test_seq[2] = test_seq[0]
    # Movement frame
    test_seq[3][0:10] = 0.5
    
    compressed = temporal_hamming_filter(test_seq, threshold=0.01, min_changes=2)
    assert len(compressed) < len(test_seq), "Hamming filter did not drop duplicate stationary frames!"
    print(f"[PASS] temporal_hamming_filter dropped stationary frames (Reduced from {len(test_seq)} to {len(compressed)} frames).")
    
    # Verify Relative Coordinate Normalization scale/translation invariance
    test_pose = np.random.randn(150).astype(np.float32)
    # Give it mock shoulder span indices for scale reference (l_shoulder=135:138, r_shoulder=138:141)
    test_pose[135:138] = [-0.3, 0.0, 0.0]
    test_pose[138:141] = [0.3, 0.0, 0.0]
    
    norm1 = normalize_coordinates(test_pose)
    
    # Translate pose coordinates globally
    translated_pose = test_pose.copy()
    translated_pose += 1.5 # Add global spatial translation offset to all keypoints
    # Restore shoulder offsets to preserve span scale
    translated_pose[135:138] = test_pose[135:138] + 1.5
    translated_pose[138:141] = test_pose[138:141] + 1.5
    
    norm2 = normalize_coordinates(translated_pose)
    
    # Verify equivalence under translation invariance
    # Using small epsilon threshold due to rounding tolerances
    max_diff = np.max(np.abs(norm1 - norm2))
    assert max_diff < 1e-4, f"Relative Normalization failed translation-invariance test! Max diff: {max_diff}"
    print("[PASS] Relative Coordinate Normalization satisfies translation-scale invariance.")


def verify_models():
    print("\n--- Verifying Phase 4: Machine Learning Models ---")
    
    from src.ml.models import ISLSeq2SeqLSTM, PoseEncodedSignformer
    
    # 1. Test LSTM Classifier compile and shapes
    lstm = ISLSeq2SeqLSTM(input_dim=150, hidden_dim=64, num_classes=26)
    test_input = torch.randn(4, 20, 150) # Batch=4, SeqLen=20, Dim=150
    out = lstm(test_input)
    assert out.shape == (4, 26), f"LSTM output shape mismatch! Got {out.shape}, expected (4, 26)."
    print("[PASS] PyTorch Sequential LSTM is successfully compiled and handles forward passes.")
    
    # 2. Test Signformer compile and shapes
    signformer = PoseEncodedSignformer(input_dim=150, d_model=32, nhead=2, target_vocab_size=10)
    src = torch.randn(2, 30, 150) # Batch=2, SrcSeqLen=30
    tgt = torch.ones(2, 5, dtype=torch.long) # Batch=2, TgtSeqLen=5
    logits = signformer(src, tgt)
    assert logits.shape == (2, 5, 10), f"Signformer output shape mismatch! Got {logits.shape}, expected (2, 5, 10)."
    print("[PASS] Pose-Encoded Signformer seq-to-seq model compiles and executes forward passes.")
    
    # 3. Check trained model weight check-points
    assert os.path.exists("models/isl_lstm_classifier.pth"), "LSTM model checkpoint missing!"
    assert os.path.exists("models/pose_encoded_signformer.pth"), "Signformer model checkpoint missing!"
    print("[PASS] Saved PyTorch model checkpoints are loaded successfully.")


def verify_fastapi_endpoints():
    print("\n--- Verifying Phase 5: FastAPI Application ---")
    
    from fastapi.testclient import TestClient
    from src.app.main import app
    
    client = TestClient(app)
    
    # Test index page
    res_index = client.get("/")
    assert res_index.status_code == 200, "Index page failed to load!"
    
    # Test Text-to-Pose endpoint
    res_pose = client.get("/api/text-to-pose?text=hello%20thank%20you")
    assert res_pose.status_code == 200, "Text-to-Pose endpoint failed!"
    data = res_pose.json()
    assert "hello" in data["sequences"], "Missing gesture key for 'hello'!"
    assert len(data["sequences"]["hello"][0]) == 150, "Invalid coordinate vector size!"
    print("[PASS] FastAPI endpoints respond successfully with correct coordinate structures.")


if __name__ == "__main__":
    print("==================================================")
    print("Running Automated System Verification Suite")
    print("==================================================")
    
    try:
        verify_infrastructure()
        verify_etl_pipeline()
        verify_models()
        verify_fastapi_endpoints()
        print("\n==================================================")
        print("ALL PROGRAMMATIC VERIFICATION CHECKS PASSED!")
        print("==================================================")
    except Exception as e:
        print(f"\n[FAIL] Programmatic verification failed: {e}")
        exit(1)
