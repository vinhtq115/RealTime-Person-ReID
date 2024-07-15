import numpy as np


def get_pose_status(kpts_int: np.ndarray, kpts_score: np.ndarray) -> int:
    """Get pose status. Requires at least shoulder, hip and knee of one side to be visible.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        int: -1: unknown, 0: sitting, 1: standing
    """
    assert kpts_int.shape == (17, 2)

    # Check if shoulder, hip, knee are visible
    left_parts_visible = np.all(kpts_score[[5, 11, 13]] >= 0.5)
    right_parts_visible = np.all(kpts_score[[6, 12, 14]] >= 0.5)

    if not (left_parts_visible or right_parts_visible):
        return -1

    # Calculate angle(s) between body and leg
    if left_parts_visible:
        left_body_vector = kpts_int[5] - kpts_int[11]
        left_leg_vector = kpts_int[11] - kpts_int[13]
        left_angle = np.arccos(np.dot(left_body_vector, left_leg_vector) / (np.linalg.norm(left_body_vector) * np.linalg.norm(left_leg_vector)))
        left_angle = np.rad2deg(left_angle)

    if right_parts_visible:
        right_body_vector = kpts_int[6] - kpts_int[12]
        right_leg_vector = kpts_int[12] - kpts_int[14]
        right_angle = np.arccos(np.dot(right_body_vector, right_leg_vector) / (np.linalg.norm(right_body_vector) * np.linalg.norm(right_leg_vector)))
        right_angle = np.rad2deg(right_angle)

    # If both left and right parts are visible
    if left_parts_visible and right_parts_visible:
        return 1 if left_angle < 45 or right_angle < 45 else 0
    elif left_parts_visible:
        return 1 if left_angle < 45 else 0
    else:
        return 1 if right_angle < 45 else 0


def get_position_if_leg_visible_and_standing(kpts_int: np.ndarray, kpts_score: np.ndarray) -> np.ndarray | None:
    """Check if at least one leg is visible and pose is in standing position.
    If so, return the ankle coordinate.

    Args:
        kpts_int (np.ndarray): Keypoing coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        np.ndarray | None: Ankle coordinate if at least one leg is visible and pose is in standing position, None otherwise.
    """
    # Check which sides are visible
    left_leg_visible = np.all(kpts_score[[11, 13, 15]] >= 0.5)
    right_leg_visible = np.all(kpts_score[[12, 14, 16]] >= 0.5)

    if not (left_leg_visible or right_leg_visible):
        return None

    # Calculate angle
    if left_leg_visible:
        left_thigh = kpts_int[13] - kpts_int[11]
        left_calf = kpts_int[15] - kpts_int[13]
        left_leg_angle = np.arccos(np.dot(left_thigh, left_calf) / (np.linalg.norm(left_thigh) * np.linalg.norm(left_calf)))
        left_leg_angle = np.rad2deg(left_leg_angle)
        left_leg_angle_to_y_axis = np.arccos(np.dot(left_thigh, [0, 1]) / (np.linalg.norm(left_thigh) * np.linalg.norm([0, 1])))
        left_leg_angle_to_y_axis = np.rad2deg(left_leg_angle_to_y_axis)
    if right_leg_visible:
        right_thigh = kpts_int[14] - kpts_int[12]
        right_calf = kpts_int[16] - kpts_int[14]
        right_leg_angle = np.arccos(np.dot(right_thigh, right_calf) / (np.linalg.norm(right_thigh) * np.linalg.norm(right_calf)))
        right_leg_angle = np.rad2deg(right_leg_angle)
        right_leg_angle_to_y_axis = np.arccos(np.dot(right_thigh, [0, 1]) / (np.linalg.norm(right_thigh) * np.linalg.norm([0, 1])))
        right_leg_angle_to_y_axis = np.rad2deg(right_leg_angle_to_y_axis)

    # If at least one leg has angle less than 45 degrees, return True
    if (left_leg_visible and left_leg_angle < 45 and left_leg_angle_to_y_axis < 45) and \
        (right_leg_visible and right_leg_angle < 45 and right_leg_angle_to_y_axis < 45):
        return np.mean([kpts_int[15], kpts_int[16]], axis=0)
    elif left_leg_visible and left_leg_angle < 45 and left_leg_angle_to_y_axis < 45:
        return kpts_int[15]
    elif right_leg_visible and right_leg_angle < 45 and right_leg_angle_to_y_axis < 45:
        return kpts_int[16]
    return None


def get_ankle_from_body_and_knee_standing(kpts_int: np.ndarray, kpts_score: np.ndarray) -> np.ndarray:
    """Estimate ankle coordinate. Requires at least one side of both
    hip and knee to be visible and pose is in standing position.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        np.ndarray: Ankle coordinate. Shape: (2,)
    """
    # Check which sides are visible
    left_parts_visible = np.all(kpts_score[[5, 11, 13]] >= 0.5)
    right_parts_visible = np.all(kpts_score[[6, 12, 14]] >= 0.5)

    assert left_parts_visible or right_parts_visible, "At least one side of hip and knee must be visible"

    if left_parts_visible and right_parts_visible:
        # If both sides are visible, return the average
        left_hip_vector = kpts_int[11] - kpts_int[5]
        left_thigh_length = np.linalg.norm(kpts_int[13] - kpts_int[11])

        right_hip_vector = kpts_int[12] - kpts_int[6]
        right_thigh_length = np.linalg.norm(kpts_int[14] - kpts_int[12])

        avg_thigh_length = (left_thigh_length + right_thigh_length) / 2
        avg_hip_vector = (left_hip_vector + right_hip_vector) / 2
        avg_hip = np.mean([kpts_int[11], kpts_int[12]], axis=0)

        # Calculate coordinate of ankle
        ankle = avg_hip + avg_hip_vector / np.linalg.norm(avg_hip_vector) * avg_thigh_length * 2
    elif left_parts_visible:
        left_hip_vector = kpts_int[11] - kpts_int[5]
        left_thigh_length = np.linalg.norm(kpts_int[13] - kpts_int[11])
        ankle = kpts_int[11] + left_hip_vector / np.linalg.norm(left_hip_vector) * left_thigh_length * 2
    else:
        right_hip_vector = kpts_int[12] - kpts_int[6]
        right_thigh_length = np.linalg.norm(kpts_int[14] - kpts_int[12])
        ankle = kpts_int[12] + right_hip_vector / np.linalg.norm(right_hip_vector) * right_thigh_length * 2
    return ankle


def get_sitting_position(kpts_int: np.ndarray, kpts_score: np.ndarray) -> np.ndarray:
    """Estimate sitting position. Requires at least one side of hip and knee to be visible.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        np.ndarray: Sitting position. Shape: (2,)
    """
    # Check which sides are visible
    left_parts_visible = np.all(kpts_score[[5, 11, 13]] >= 0.5)
    right_parts_visible = np.all(kpts_score[[6, 12, 14]] >= 0.5)

    assert left_parts_visible or right_parts_visible, "At least one side of hip and knee must be visible"

    if left_parts_visible and right_parts_visible:
        # If both sides are visible, return the average
        left_hip_vector = kpts_int[11] - kpts_int[5]
        left_hip_length = np.linalg.norm(left_hip_vector)

        right_hip_vector = kpts_int[12] - kpts_int[6]
        right_hip_length = np.linalg.norm(right_hip_vector)

        avg_hip_length = (left_hip_length + right_hip_length) / 2
        avg_hip_vector = (left_hip_vector + right_hip_vector) / 2
        avg_hip = np.mean([kpts_int[11], kpts_int[12]], axis=0)

        # Calculate sitting position
        position = avg_hip + avg_hip_vector / np.linalg.norm(avg_hip_vector) * avg_hip_length
    elif left_parts_visible:
        left_hip_vector = kpts_int[11] - kpts_int[5]
        left_hip_length = np.linalg.norm(left_hip_vector)
        position = kpts_int[11] + left_hip_vector / np.linalg.norm(left_hip_vector) * left_hip_length
    else:
        right_hip_vector = kpts_int[12] - kpts_int[6]
        right_hip_length = np.linalg.norm(right_hip_vector)
        position = kpts_int[12] + right_hip_vector / np.linalg.norm(right_hip_vector) * right_hip_length
    return position


def head_visible(kpts_score: np.ndarray) -> bool:
    """Check if only head keypoints are visible and other keypoints are not.

    Args:
        kpts_score (np.ndarray): Keypoints score. Shape: (17,)

    Returns:
        bool: True if only head keypoints are visible and others are not, False otherwise.
    """
    return np.any(kpts_score[0:5] >= 0.5)


def head_shoulder_visible(kpts_score: np.ndarray) -> bool:
    """Check if head and shoulder keypoints are visible.

    Args:
        kpts_score (np.ndarray): Keypoints score. Shape: (17,)

    Returns:
        bool: True if only head and shoulder keypoints are visible, False otherwise.
    """
    return np.any(kpts_score[0:5] >= 0.5) and np.any(kpts_score[5:7] >= 0.5)


def estimate_position_from_shoulder_hip(kpts_int: np.ndarray, kpts_score: np.ndarray) -> np.ndarray:
    """Estimate position from shoulder and hip keypoints.
    Assuming that shoulder and hip keypoints are visible.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        np.ndarray: Estimated position. Shape: (2,)
    """
    if np.all(kpts_score[5:7] >= 0.5):
        shoulder = np.mean(kpts_int[5:7], axis=0)
    elif kpts_score[5] >= 0.5:
        shoulder = kpts_int[5]
    else:
        shoulder = kpts_int[6]

    if np.all(kpts_score[11:13] >= 0.5):
        hip = np.mean(kpts_int[11:13], axis=0)
    elif kpts_score[11] >= 0.5:
        hip = kpts_int[11]
    else:
        hip = kpts_int[12]

    vector = hip - shoulder
    body_length = np.linalg.norm(shoulder - hip)

    return hip + vector / np.linalg.norm(vector) * 1.25 * body_length


def estimate_position_from_shoulder_elbow(kpts_int: np.ndarray, kpts_score: np.ndarray) -> np.ndarray:
    """Estimate position from shoulder and elbow keypoints.
    Assuming that shoulder and elbow keypoints are visible.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)

    Returns:
        np.ndarray: Estimated position. Shape: (2,)
    """
    if np.all(kpts_score[5:7] >= 0.5):
        shoulder = np.mean(kpts_int[5:7], axis=0)
    elif kpts_score[5] >= 0.5:
        shoulder = kpts_int[5]
    else:
        shoulder = kpts_int[6]

    if np.all(kpts_score[7:9] >= 0.5):
        elbow = np.mean(kpts_int[7:9], axis=0)
    elif kpts_score[7] >= 0.5:
        elbow = kpts_int[7]
    else:
        elbow = kpts_int[9]

    vector = elbow - shoulder
    upper_arm_length = np.linalg.norm(shoulder - elbow)

    return elbow + vector / np.linalg.norm(vector) * 4 * upper_arm_length


def estimate_position_from_head_shoulder(kpts_int: np.ndarray, kpts_score: np.ndarray, x1: int, x2: int) -> np.ndarray:
    # Center of bbounging box is used as x-coordinate.
    avg_head = np.sum(kpts_int[0:5, 1], axis=0)
    head_visible_kpts_count = np.sum(kpts_score[0:5] >= 0.5)
    avg_head = avg_head / head_visible_kpts_count

    if np.all(kpts_score[5:7] >= 0.5):
        # Both shoulders are visible
        avg_shoulder = np.mean(kpts_int[5:7, 1], axis=0)
    elif kpts_score[5] >= 0.5:
        avg_shoulder = kpts_int[5, 1]
    else:
        avg_shoulder = kpts_int[6, 1]

    shoulder_vector = avg_shoulder - avg_head
    estimated_position = avg_head + shoulder_vector * 4.5
    x_mean = (x1 + x2) / 2
    return np.array([x_mean, estimated_position])


def estimate_position_from_head_bbox(kpts_int: np.ndarray, kpts_score: np.ndarray, x1: int, x2: int, y2: int) -> np.ndarray:
    # Center of bbounging box is used as x-coordinate.
    avg_head_y = np.sum(kpts_int[0:5, 1], axis=0)
    head_visible_kpts_count = np.sum(kpts_score[0:5] >= 0.5)
    avg_head_y = avg_head_y / head_visible_kpts_count
    x_mean = (x1 + x2) / 2
    neck_length = y2 - avg_head_y
    estimated_position_y = y2 + neck_length * 4.5
    return np.array([x_mean, estimated_position_y])
