"""Landmark definitions and skeleton connections for pose estimation models."""

# ── Goliath 308 keypoints (Meta Sapiens, ECCV 2024) ──────────────────────
# Source: https://huggingface.co/spaces/facebook/sapiens-pose
# Order matches TorchScript model output channels (0–307).

GOLIATH_LANDMARK_NAMES = [
    # 0-14: Body (15)
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    # 15-20: Feet (6)
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    # 21-41: Right hand (21)
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    # 42-62: Left hand (21)
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    # 63-69: Additional body (7)
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
    # 70-77: Face structure (8)
    "center_of_glabella",
    "center_of_nose_root",
    "tip_of_nose_bridge",
    "midpoint_1_of_nose_bridge",
    "midpoint_2_of_nose_bridge",
    "midpoint_3_of_nose_bridge",
    "center_of_labiomental_groove",
    "tip_of_chin",
    # 78-86: Right eyebrow (9)
    "upper_startpoint_of_r_eyebrow",
    "lower_startpoint_of_r_eyebrow",
    "end_of_r_eyebrow",
    "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow",
    "upper_midpoint_2_of_r_eyebrow",
    "upper_midpoint_3_of_r_eyebrow",
    "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow",
    # 87-95: Left eyebrow (9)
    "upper_startpoint_of_l_eyebrow",
    "lower_startpoint_of_l_eyebrow",
    "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow",
    "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow",
    "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow",
    "lower_midpoint_3_of_l_eyebrow",
    # 96-119: Left eye (24)
    "l_inner_end_of_upper_lash_line",
    "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line",
    "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line",
    "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line",
    "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line",
    "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line",
    "l_midpoint_2_of_upper_eyelid_line",
    "l_midpoint_5_of_upper_eyelid_line",
    "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line",
    "l_midpoint_1_of_upper_eyelid_line",
    "l_midpoint_3_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line",
    "l_midpoint_5_of_upper_crease_line",
    "l_centerpoint_of_upper_crease_line",
    "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line",
    "l_midpoint_3_of_upper_crease_line",
    # 120-143: Right eye (24)
    "r_inner_end_of_upper_lash_line",
    "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line",
    "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line",
    "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line",
    "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line",
    "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line",
    "r_midpoint_1_of_upper_eyelid_line",
    "r_midpoint_4_of_upper_eyelid_line",
    "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line",
    "r_midpoint_2_of_upper_eyelid_line",
    "r_midpoint_6_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line",
    "r_midpoint_4_of_upper_crease_line",
    "r_centerpoint_of_upper_crease_line",
    "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line",
    "r_midpoint_6_of_upper_crease_line",
    # 144-160: Left lower eye (17)
    "l_inner_end_of_lower_lash_line",
    "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line",
    "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line",
    "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line",
    "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line",
    "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line",
    "l_midpoint_2_of_lower_eyelid_line",
    "l_midpoint_5_of_lower_eyelid_line",
    "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line",
    "l_midpoint_1_of_lower_eyelid_line",
    "l_midpoint_3_of_lower_eyelid_line",
    # 161-177: Right lower eye (17)
    "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line",
    "r_centerpoint_of_lower_lash_line",
    "r_midpoint_1_of_lower_lash_line",
    "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line",
    "r_midpoint_4_of_lower_lash_line",
    "r_midpoint_5_of_lower_lash_line",
    "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line",
    "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line",
    "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line",
    "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line",
    "r_midpoint_6_of_lower_eyelid_line",
    # 178-187: Nose detail (10)
    "tip_of_nose",
    "bottom_center_of_nose",
    "r_outer_corner_of_nose",
    "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril",
    "outer_corner_of_r_nostril",
    "upper_corner_of_r_nostril",
    "inner_corner_of_l_nostril",
    "outer_corner_of_l_nostril",
    "upper_corner_of_l_nostril",
    # 188-219: Mouth (32)
    "r_outer_corner_of_mouth",
    "l_outer_corner_of_mouth",
    "center_of_cupid_bow",
    "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip",
    "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip",
    "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip",
    "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip",
    "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip",
    "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip",
    "midpoint_6_of_lower_outer_lip",
    "r_inner_corner_of_mouth",
    "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip",
    "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip",
    "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip",
    "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip",
    "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip",
    "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip",
    "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip",
    "midpoint_6_of_lower_inner_lip",
    # 220-245: Left ear (26)
    "l_top_end_of_inferior_crus",
    "l_top_end_of_superior_crus",
    "l_start_of_antihelix",
    "l_end_of_antihelix",
    "l_midpoint_1_of_antihelix",
    "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix",
    "l_midpoint_3_of_antihelix",
    "l_point_1_of_inner_helix",
    "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix",
    "l_point_4_of_inner_helix",
    "l_point_5_of_inner_helix",
    "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix",
    "l_highest_point_of_antitragus",
    "l_bottom_point_of_tragus",
    "l_protruding_point_of_tragus",
    "l_top_point_of_tragus",
    "l_start_point_of_crus_of_helix",
    "l_deepest_point_of_concha",
    "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15",
    "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix",
    "l_point_8_of_inner_helix",
    # 246-271: Right ear (26)
    "r_top_end_of_inferior_crus",
    "r_top_end_of_superior_crus",
    "r_start_of_antihelix",
    "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix",
    "r_midpoint_1_of_inferior_crus",
    "r_midpoint_2_of_antihelix",
    "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix",
    "r_point_8_of_inner_helix",
    "r_point_3_of_inner_helix",
    "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix",
    "r_point_6_of_inner_helix",
    "r_point_7_of_inner_helix",
    "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus",
    "r_protruding_point_of_tragus",
    "r_top_point_of_tragus",
    "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha",
    "r_tip_of_ear_lobe",
    "r_midpoint_between_22_15",
    "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix",
    "r_point_2_of_inner_helix",
    # 272-289: Iris (18)
    "l_center_of_iris",
    "l_border_of_iris_3",
    "l_border_of_iris_midpoint_1",
    "l_border_of_iris_12",
    "l_border_of_iris_midpoint_4",
    "l_border_of_iris_9",
    "l_border_of_iris_midpoint_3",
    "l_border_of_iris_6",
    "l_border_of_iris_midpoint_2",
    "r_center_of_iris",
    "r_border_of_iris_3",
    "r_border_of_iris_midpoint_1",
    "r_border_of_iris_12",
    "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9",
    "r_border_of_iris_midpoint_3",
    "r_border_of_iris_6",
    "r_border_of_iris_midpoint_2",
    # 290-307: Pupils (18)
    "l_center_of_pupil",
    "l_border_of_pupil_3",
    "l_border_of_pupil_midpoint_1",
    "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4",
    "l_border_of_pupil_9",
    "l_border_of_pupil_midpoint_3",
    "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2",
    "r_center_of_pupil",
    "r_border_of_pupil_3",
    "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12",
    "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9",
    "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6",
    "r_border_of_pupil_midpoint_2",
]

GOLIATH_NAME_TO_INDEX = {name: i for i, name in enumerate(GOLIATH_LANDMARK_NAMES)}

# Skeleton connections for Goliath 308 (index pairs).
# Body + feet + hands + additional body points.  No face (70-307).
GOLIATH_SKELETON_CONNECTIONS = [
    # ── Head ──
    (0, 69),   # nose → neck
    (0, 1),    # nose → left_eye
    (0, 2),    # nose → right_eye
    (1, 3),    # left_eye → left_ear
    (2, 4),    # right_eye → right_ear
    # ── Neck → shoulders (via acromion) ──
    (69, 67),  # neck → left_acromion
    (69, 68),  # neck → right_acromion
    (67, 5),   # left_acromion → left_shoulder
    (68, 6),   # right_acromion → right_shoulder
    # ── Arms ──
    (5, 7),    # left_shoulder → left_elbow
    (6, 8),    # right_shoulder → right_elbow
    (7, 62),   # left_elbow → left_wrist
    (8, 41),   # right_elbow → right_wrist
    # ── Elbow detail ──
    (7, 63),   # left_elbow → left_olecranon
    (7, 65),   # left_elbow → left_cubital_fossa
    (8, 64),   # right_elbow → right_olecranon
    (8, 66),   # right_elbow → right_cubital_fossa
    # ── Torso ──
    (5, 9),    # left_shoulder → left_hip
    (6, 10),   # right_shoulder → right_hip
    (9, 10),   # left_hip → right_hip
    # ── Legs ──
    (9, 11),   # left_hip → left_knee
    (10, 12),  # right_hip → right_knee
    (11, 13),  # left_knee → left_ankle
    (12, 14),  # right_knee → right_ankle
    # ── Left foot ──
    (13, 17),  # left_ankle → left_heel
    (13, 15),  # left_ankle → left_big_toe
    (17, 15),  # left_heel → left_big_toe
    (15, 16),  # left_big_toe → left_small_toe
    # ── Right foot ──
    (14, 20),  # right_ankle → right_heel
    (14, 18),  # right_ankle → right_big_toe
    (20, 18),  # right_heel → right_big_toe
    (18, 19),  # right_big_toe → right_small_toe
    # ── Right hand (wrist = 41) ──
    (41, 24), (24, 23), (23, 22), (22, 21),  # thumb chain
    (41, 28), (28, 27), (27, 26), (26, 25),  # forefinger chain
    (41, 32), (32, 31), (31, 30), (30, 29),  # middle finger chain
    (41, 36), (36, 35), (35, 34), (34, 33),  # ring finger chain
    (41, 40), (40, 39), (39, 38), (38, 37),  # pinky chain
    (24, 28), (28, 32), (32, 36), (36, 40),  # palm (base joints)
    # ── Left hand (wrist = 62) ──
    (62, 45), (45, 44), (44, 43), (43, 42),  # thumb chain
    (62, 49), (49, 48), (48, 47), (47, 46),  # forefinger chain
    (62, 53), (53, 52), (52, 51), (51, 50),  # middle finger chain
    (62, 57), (57, 56), (56, 55), (55, 54),  # ring finger chain
    (62, 61), (61, 60), (60, 59), (59, 58),  # pinky chain
    (45, 49), (49, 53), (53, 57), (57, 61),  # palm (base joints)
]

# Indices 0–69 are body/feet/hands/additional.  70–307 are face detail.
GOLIATH_FACE_START = 70

# Direct mapping from Goliath 308 indices to MediaPipe 33 landmark names.
# Bypasses the lossy COCO 17 intermediate step, filling all 33 landmarks.
# FOOT_INDEX is not mapped here — computed as midpoint(big_toe, small_toe).
GOLIATH_TO_MP = {
    # Body core (same as COCO, but maps directly to MP names)
    0: "NOSE",
    1: "LEFT_EYE",
    2: "RIGHT_EYE",
    3: "LEFT_EAR",
    4: "RIGHT_EAR",
    5: "LEFT_SHOULDER",
    6: "RIGHT_SHOULDER",
    7: "LEFT_ELBOW",
    8: "RIGHT_ELBOW",
    9: "LEFT_HIP",
    10: "RIGHT_HIP",
    11: "LEFT_KNEE",
    12: "RIGHT_KNEE",
    13: "LEFT_ANKLE",
    14: "RIGHT_ANKLE",
    # Feet
    17: "LEFT_HEEL",
    20: "RIGHT_HEEL",
    # Wrists
    41: "RIGHT_WRIST",
    62: "LEFT_WRIST",
    # Hand fingertips (tip = "4" suffix in Goliath)
    21: "RIGHT_THUMB",          # right_thumb4
    25: "RIGHT_INDEX",          # right_forefinger4
    37: "RIGHT_PINKY",          # right_pinky_finger4
    42: "LEFT_THUMB",           # left_thumb4
    46: "LEFT_INDEX",           # left_forefinger4
    58: "LEFT_PINKY",           # left_pinky_finger4
    # Eye inner/outer corners (from lash line endpoints)
    96: "LEFT_EYE_INNER",       # l_inner_end_of_upper_lash_line
    97: "LEFT_EYE_OUTER",       # l_outer_end_of_upper_lash_line
    120: "RIGHT_EYE_INNER",     # r_inner_end_of_upper_lash_line
    121: "RIGHT_EYE_OUTER",     # r_outer_end_of_upper_lash_line
    # Mouth corners
    189: "MOUTH_LEFT",          # l_outer_corner_of_mouth
    188: "MOUTH_RIGHT",         # r_outer_corner_of_mouth
}

# Mapping from Goliath indices to COCO 17 indices.
# NOTE: Goliath ordering differs from COCO — wrists are at 41/62, not 9/10.
GOLIATH_TO_COCO = {
    0: 0,     # nose
    1: 1,     # left_eye
    2: 2,     # right_eye
    3: 3,     # left_ear
    4: 4,     # right_ear
    5: 5,     # left_shoulder
    6: 6,     # right_shoulder
    7: 7,     # left_elbow
    8: 8,     # right_elbow
    9: 11,    # left_hip
    10: 12,   # right_hip
    11: 13,   # left_knee
    12: 14,   # right_knee
    13: 15,   # left_ankle
    14: 16,   # right_ankle
    41: 10,   # right_wrist
    62: 9,    # left_wrist
}

# MediaPipe Pose landmarks (33 total)
MP_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

MP_NAME_TO_INDEX = {name: i for i, name in enumerate(MP_LANDMARK_NAMES)}

# COCO keypoint format (17 landmarks) - used by YOLO, HRNET, Sapiens, MMPose
COCO_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
]

COCO_NAME_TO_INDEX = {name: i for i, name in enumerate(COCO_LANDMARK_NAMES)}

# Mapping from COCO landmark names to MediaPipe landmark names
COCO_TO_MP = {
    'NOSE': 'NOSE',
    'LEFT_EYE': 'LEFT_EYE',
    'RIGHT_EYE': 'RIGHT_EYE',
    'LEFT_EAR': 'LEFT_EAR',
    'RIGHT_EAR': 'RIGHT_EAR',
    'LEFT_SHOULDER': 'LEFT_SHOULDER',
    'RIGHT_SHOULDER': 'RIGHT_SHOULDER',
    'LEFT_ELBOW': 'LEFT_ELBOW',
    'RIGHT_ELBOW': 'RIGHT_ELBOW',
    'LEFT_WRIST': 'LEFT_WRIST',
    'RIGHT_WRIST': 'RIGHT_WRIST',
    'LEFT_HIP': 'LEFT_HIP',
    'RIGHT_HIP': 'RIGHT_HIP',
    'LEFT_KNEE': 'LEFT_KNEE',
    'RIGHT_KNEE': 'RIGHT_KNEE',
    'LEFT_ANKLE': 'LEFT_ANKLE',
    'RIGHT_ANKLE': 'RIGHT_ANKLE',
}

# Skeleton connections for visualization (MediaPipe indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
]

# COCO skeleton connections
COCO_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Landmarks needed for gait analysis (minimum set)
GAIT_LANDMARKS = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
]

# ── Sapiens body-part segmentation classes (28) ───────────────────────
# Source: facebookresearch/sapiens — lite/demo/classes_and_palettes.py
GOLIATH_SEG_CLASSES = [
    "Background", "Apparel", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
    "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg",
    "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm",
    "Right_Upper_Leg", "Torso", "Upper_Clothing",
    "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
]

# Body-part indices (skin/body, excluding background and clothing)
GOLIATH_SEG_BODY_INDICES = [2, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 19, 20, 21]

# ── COCO-WholeBody 133 keypoints (RTMW) ──────────────────────────────
# Source: mmpose configs/_base_/datasets/coco_wholebody.py
WHOLEBODY_LANDMARK_NAMES = (
    # 0-16: Body (same as COCO 17)
    ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
     "left_wrist", "right_wrist", "left_hip", "right_hip",
     "left_knee", "right_knee", "left_ankle", "right_ankle"]
    # 17-22: Feet (6)
    + ["left_big_toe", "left_small_toe", "left_heel",
       "right_big_toe", "right_small_toe", "right_heel"]
    # 23-90: Face (68 — standard 68-point face landmark convention)
    + [f"face_{i}" for i in range(68)]
    # 91-111: Left hand (21)
    + ["left_hand_root",
       "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
       "left_forefinger1", "left_forefinger2", "left_forefinger3", "left_forefinger4",
       "left_middle_finger1", "left_middle_finger2", "left_middle_finger3", "left_middle_finger4",
       "left_ring_finger1", "left_ring_finger2", "left_ring_finger3", "left_ring_finger4",
       "left_pinky_finger1", "left_pinky_finger2", "left_pinky_finger3", "left_pinky_finger4"]
    # 112-132: Right hand (21)
    + ["right_hand_root",
       "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
       "right_forefinger1", "right_forefinger2", "right_forefinger3", "right_forefinger4",
       "right_middle_finger1", "right_middle_finger2", "right_middle_finger3", "right_middle_finger4",
       "right_ring_finger1", "right_ring_finger2", "right_ring_finger3", "right_ring_finger4",
       "right_pinky_finger1", "right_pinky_finger2", "right_pinky_finger3", "right_pinky_finger4"]
)

# First 17 of WholeBody are COCO body keypoints (identical indices)
WHOLEBODY_TO_COCO = {i: i for i in range(17)}

# ── Extended foot landmark names (detected from Sapiens / RTMW) ──────
EXTENDED_FOOT_LANDMARKS = [
    "LEFT_BIG_TOE", "LEFT_SMALL_TOE",
    "RIGHT_BIG_TOE", "RIGHT_SMALL_TOE",
]

# Goliath 308 foot indices → MediaPipe-style landmark names
GOLIATH_FOOT_INDICES = {
    15: "LEFT_BIG_TOE", 16: "LEFT_SMALL_TOE", 17: "LEFT_HEEL",
    18: "RIGHT_BIG_TOE", 19: "RIGHT_SMALL_TOE", 20: "RIGHT_HEEL",
}

# COCO-WholeBody 133 (RTMW) foot indices → MediaPipe-style landmark names
RTMW_FOOT_INDICES = {
    17: "LEFT_BIG_TOE", 18: "LEFT_SMALL_TOE", 19: "LEFT_HEEL",
    20: "RIGHT_BIG_TOE", 21: "RIGHT_SMALL_TOE", 22: "RIGHT_HEEL",
}

# ── Inverse mapping tables: MediaPipe 33 → other formats ─────────────

# MediaPipe landmark name → COCO 17 index
MP_TO_COCO_17 = {
    "NOSE": 0,
    "LEFT_EYE": 1,
    "RIGHT_EYE": 2,
    "LEFT_EAR": 3,
    "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,
    "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,
    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
}

BODY_25_LANDMARK_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]

# MediaPipe landmark name → BODY_25 index (None if no direct mapping)
MP_TO_BODY25 = {
    "NOSE": 0,
    "RIGHT_SHOULDER": 2, "RIGHT_ELBOW": 3, "RIGHT_WRIST": 4,
    "LEFT_SHOULDER": 5, "LEFT_ELBOW": 6, "LEFT_WRIST": 7,
    "RIGHT_HIP": 9, "RIGHT_KNEE": 10, "RIGHT_ANKLE": 11,
    "LEFT_HIP": 12, "LEFT_KNEE": 13, "LEFT_ANKLE": 14,
    "RIGHT_EYE": 15, "LEFT_EYE": 16, "RIGHT_EAR": 17, "LEFT_EAR": 18,
    "LEFT_FOOT_INDEX": 19, "LEFT_HEEL": 21,
    "RIGHT_FOOT_INDEX": 22, "RIGHT_HEEL": 24,
}

HALPE_26_LANDMARK_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe",
    "LHeel", "RHeel",
]

MP_TO_HALPE26 = {
    "NOSE": 0, "LEFT_EYE": 1, "RIGHT_EYE": 2,
    "LEFT_EAR": 3, "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10,
    "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16,
    "LEFT_FOOT_INDEX": 20, "RIGHT_FOOT_INDEX": 21,
    "LEFT_HEEL": 24, "RIGHT_HEEL": 25,
}

# Mapping myogait landmarks → OpenSim marker names for standard models
OPENSIM_MARKER_MAP = {
    "gait2392": {
        "RIGHT_SHOULDER": "R.Acromion",
        "LEFT_SHOULDER": "L.Acromion",
        "RIGHT_HIP": "R.ASIS",
        "LEFT_HIP": "L.ASIS",
        "RIGHT_KNEE": "R.Knee.Lat",
        "LEFT_KNEE": "L.Knee.Lat",
        "RIGHT_ANKLE": "R.Ankle.Lat",
        "LEFT_ANKLE": "L.Ankle.Lat",
        "RIGHT_HEEL": "R.Heel",
        "LEFT_HEEL": "L.Heel",
        "RIGHT_FOOT_INDEX": "R.MT5",
        "LEFT_FOOT_INDEX": "L.MT5",
    },
    "rajagopal2015": {
        "RIGHT_SHOULDER": "R.Acromion",
        "LEFT_SHOULDER": "L.Acromion",
        "RIGHT_HIP": "R.ASIS",
        "LEFT_HIP": "L.ASIS",
        "RIGHT_KNEE": "R.Knee.Lat",
        "LEFT_KNEE": "L.Knee.Lat",
        "RIGHT_ANKLE": "R.Ankle.Lat",
        "LEFT_ANKLE": "L.Ankle.Lat",
        "RIGHT_HEEL": "R.Heel",
        "LEFT_HEEL": "L.Heel",
        "RIGHT_FOOT_INDEX": "R.Toe.Tip",
        "LEFT_FOOT_INDEX": "L.Toe.Tip",
    },
}
