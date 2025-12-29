# Reviewer: 박용준
"""
A* pathfinding implementation for occupancy grid navigation.

Provides efficient path planning on 2D occupancy grids with 8-directional movement.
Used for realistic Nav2 simulation and event reachability validation.
"""

from typing import Tuple, List, Optional
import numpy as np
import heapq


class AStarPathfinder:
    """
    A* pathfinding algorithm for 2D occupancy grids.

    Attributes:
        grid: 2D numpy array where 0=free, 1=occupied
        resolution: Grid cell size in meters (default: 1.0)
    """

    def __init__(self, occupancy_grid: np.ndarray, resolution: float = 1.0):
        """
        Initialize pathfinder with occupancy grid.

        Args:
            occupancy_grid: 2D array (height x width), 0=free, 1=occupied
            resolution: Meters per grid cell (default: 1.0m)
        """
        self.grid = occupancy_grid.astype(np.uint8)
        self.height, self.width = self.grid.shape
        self.resolution = resolution

        # 8-directional movement (N, NE, E, SE, S, SW, W, NW)
        self.directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        # Cost multipliers (diagonal = sqrt(2), straight = 1.0)
        self.costs = [
            1.0, 1.414, 1.0, 1.414,
            1.0, 1.414, 1.0, 1.414
        ]

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices."""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return grid_y, grid_x  # Note: (row, col) = (y, x)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (meters)."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def is_valid(self, row: int, col: int) -> bool:
        """Check if grid cell is within bounds and free."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col] == 0
        return False

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Octile distance heuristic (supports diagonal movement)."""
        dy = abs(pos1[0] - pos2[0])
        dx = abs(pos1[1] - pos2[1])
        # Octile: min(dx,dy)*sqrt(2) + abs(dx-dy)
        return 1.414 * min(dx, dy) + abs(dx - dy)

    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Optional[Tuple[List[Tuple[float, float]], float]]:
        """
        Find shortest path from start to goal using A*.

        Args:
            start: Start position in world coordinates (x, y) in meters
            goal: Goal position in world coordinates (x, y) in meters

        Returns:
            (path, distance) if path found, None otherwise
            - path: List of waypoints [(x, y), ...] in world coordinates
            - distance: Total path length in meters
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        # Check validity
        if not self.is_valid(start_grid[0], start_grid[1]):
            return None
        if not self.is_valid(goal_grid[0], goal_grid[1]):
            return None

        # A* search
        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))

        came_from = {}
        g_score = {start_grid: 0.0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                # Reconstruct path
                path_grid = [current]
                while current in came_from:
                    current = came_from[current]
                    path_grid.append(current)
                path_grid.reverse()

                # Convert to world coordinates
                path_world = [self.grid_to_world(r, c) for r, c in path_grid]

                # Calculate total distance
                distance = g_score[goal_grid] * self.resolution

                return path_world, distance

            # Explore neighbors
            for i, (dr, dc) in enumerate(self.directions):
                neighbor = (current[0] + dr, current[1] + dc)

                if not self.is_valid(neighbor[0], neighbor[1]):
                    continue

                tentative_g = g_score[current] + self.costs[i]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def get_distance(self, start: Tuple[float, float], goal: Tuple[float, float]) -> float:
        """
        Get path distance from start to goal.

        Returns:
            Path length in meters, or np.inf if no path exists.
        """
        result = self.find_path(start, goal)
        if result is None:
            return np.inf
        _, distance = result
        return distance

    def path_exists(self, start: Tuple[float, float], goal: Tuple[float, float]) -> bool:
        """Check if a path exists between start and goal."""
        return self.find_path(start, goal) is not None


def create_occupancy_grid_from_walls(
    width: float,
    height: float,
    walls: List[List[Tuple[float, float]]],
    resolution: float = 1.0
) -> np.ndarray:
    """
    Create occupancy grid from wall polygons.

    Args:
        width: Map width in meters
        height: Map height in meters
        walls: List of wall polygons, each polygon is list of (x, y) vertices
        resolution: Grid resolution in meters per cell

    Returns:
        occupancy_grid: 2D array (height x width), 0=free, 1=occupied
    """
    grid_width = int(np.ceil(width / resolution))
    grid_height = int(np.ceil(height / resolution))

    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Add map boundaries
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # Rasterize wall polygons using simple fill algorithm
    for wall in walls:
        if len(wall) < 2:
            continue

        # Draw lines between consecutive vertices
        for i in range(len(wall)):
            p1 = wall[i]
            p2 = wall[(i + 1) % len(wall)]

            # Convert to grid coordinates
            x1, y1 = int(p1[0] / resolution), int(p1[1] / resolution)
            x2, y2 = int(p2[0] / resolution), int(p2[1] / resolution)

            # Bresenham's line algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                # Mark cell as occupied (with thickness=2 for walls)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = y1 + dr, x1 + dc
                        if 0 <= r < grid_height and 0 <= c < grid_width:
                            grid[r, c] = 1

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

    return grid
