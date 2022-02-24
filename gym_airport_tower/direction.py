from enum import Enum


class Direction(Enum):
    """
    This Enum class describes the direction the planes are moving to. The directions are the four Cardinal directions.
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3