"""
Mathematical utility functions for robotics and RL.

Provides common math operations for:
- Angle normalization
- Vector operations
- Distance calculations
- Coordinate transformations
"""

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in [-pi, pi]

    Example:
        >>> normalize_angle(3.5 * np.pi)  # Returns ~-0.5 * pi
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_relative_vector(
    from_x: float,
    from_y: float,
    from_heading: float,
    to_x: float,
    to_y: float,
) -> Tuple[float, float]:
    """
    Compute relative vector from one pose to another in local frame.

    Transforms a target position into the robot's local coordinate frame.
    This is useful for creating egocentric observations.

    Args:
        from_x: Source x position
        from_y: Source y position
        from_heading: Source heading in radians
        to_x: Target x position
        to_y: Target y position

    Returns:
        (dx_local, dy_local): Relative position in source's local frame
        dx_local: Forward distance (positive = ahead)
        dy_local: Lateral distance (positive = left)

    Example:
        >>> # Robot at (0, 0) heading east (0 rad), target at (5, 3)
        >>> dx, dy = compute_relative_vector(0, 0, 0, 5, 3)
        >>> # Returns (5, 3) - target is 5m ahead and 3m left
    """
    # Global frame relative vector
    dx_global = to_x - from_x
    dy_global = to_y - from_y

    # Rotate to local frame
    cos_h = np.cos(from_heading)
    sin_h = np.sin(from_heading)

    dx_local = cos_h * dx_global + sin_h * dy_global
    dy_local = -sin_h * dx_global + cos_h * dy_global

    return (dx_local, dy_local)


def euclidean_distance(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates

    Returns:
        Euclidean distance
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clip value to range [min_val, max_val].

    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))
