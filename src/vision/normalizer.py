import numpy as np

def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Applies Relative Coordinate Normalization to a 150-dimensional coordinate vector.
    Schema:
      - [0:63]   : Right hand (21 points * 3D coordinates)
      - [63:126]  : Left hand (21 points * 3D coordinates)
      - [126:150] : Body pose (8 joints * 3D coordinates:
                     nose, left eye, right eye, left shoulder, right shoulder,
                     left elbow, right elbow, center-of-shoulders/hips)
    
    Normalization logic:
      - Right hand keypoints are anchored relative to the Right Wrist (first keypoint [0:3]).
      - Left hand keypoints are anchored relative to the Left Wrist (first keypoint [63:66]).
      - Pose keypoints are anchored relative to the mid-shoulder coordinate.
      - The entire coordinate universe is scaled by the dynamic shoulder-to-shoulder distance.
        Formula: P_norm = (P - P_anchor) / d_scale
    """
    normalized = coords.copy()
    
    # Extract shoulder coordinates for dynamic scaling
    # Left shoulder: joints[3] (index 126 + 3*3 = 135)
    # Right shoulder: joints[4] (index 126 + 4*3 = 138)
    l_shoulder = coords[135:138]
    r_shoulder = coords[138:141]
    
    # Calculate scale factor: Euclidean distance between shoulders (with epsilon safety)
    d_scale = np.linalg.norm(l_shoulder - r_shoulder)
    if d_scale < 1e-5:
        d_scale = 1.0  # Fallback to avoid division by zero
        
    # Calculate anchor: mid-shoulder coordinate
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    
    # 1. Normalize Right Hand (index 0 to 63)
    # Anchor: Right Wrist (index 0 to 3)
    r_wrist = coords[0:3]
    has_r_hand = np.any(coords[0:63] != 0.0)
    
    if has_r_hand:
        for i in range(21):
            start = i * 3
            end = start + 3
            # Apply: P_norm = (P - P_wrist) / d_scale
            normalized[start:end] = (coords[start:end] - r_wrist) / d_scale
            
    # 2. Normalize Left Hand (index 63 to 126)
    # Anchor: Left Wrist (index 63 to 66)
    l_wrist = coords[63:66]
    has_l_hand = np.any(coords[63:126] != 0.0)
    
    if has_l_hand:
        for i in range(21):
            start = 63 + i * 3
            end = start + 3
            # Apply: P_norm = (P - P_wrist) / d_scale
            normalized[start:end] = (coords[start:end] - l_wrist) / d_scale
            
    # 3. Normalize Pose landmarks (index 126 to 150)
    # Anchor: Mid-shoulder
    for i in range(8):
        start = 126 + i * 3
        end = start + 3
        # Apply: P_norm = (P - mid_shoulder) / d_scale
        normalized[start:end] = (coords[start:end] - mid_shoulder) / d_scale
        
    return normalized
