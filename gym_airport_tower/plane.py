from typing import List, Tuple

import numpy as np

from gym_airport_tower.direction import Direction


class Plane:
    """
    The Plane is part of the Airspace. He moves around until he lands on a runway or crashes into another plane or runway.
    """

    def __init__(self, position: List[int], idx: int, max_shape: Tuple[int, int],
                 direction: Direction = Direction(np.random.randint(0, 4))) -> None:
        """
        :param position: Actual position of the plane
        :type position: List[int, int]
        :param idx: Id of the plane to identify it in the airspace
        :type idx: int
        :param max_shape: The size of the Airspace
        :type max_shape: Tuple[int, int]
        :param direction: The direction the plane currently moves to.
        :type direction: gym_airport_tower.direction.Direction
        """
        self.id = idx
        self.position = position
        self.direction: Direction = direction
        self.max_shape = max_shape
        self.landed = False

    def move(self, direction: Direction = None) -> None:
        """
        This method moves the train one tile forward. The plane continues to move into the direction it moved in the step
        before. If the plain hits the border of the Airspace he tries to move into a random position, so that he does
        not leaves the airspace.
        :param direction: Sets the move direction of this plane to the direction before the plane moves.
        :type direction: gym_airport_tower.direction.Direction
        :return: None
        :rtype: None
        """
        if direction:
            self.direction = direction
        if self.direction == Direction.SOUTH:
            if self.position[0] < (self.max_shape[0] - 1):
                self.position[0] += 1
            else:
                self.random_direction()
                self.move()
        elif self.direction == Direction.EAST:
            if self.position[1] < (self.max_shape[1] - 1):
                self.position[1] += 1
            else:
                self.random_direction()
                self.move()
        elif self.direction == Direction.NORTH:
            if self.position[0] >= 1:
                self.position[0] -= 1
            else:
                self.random_direction()
                self.move()
        elif self.direction == Direction.WEST:
            if self.position[1] >= 1:
                self.position[1] -= 1
            else:
                self.random_direction()
                self.move()

    def random_direction(self) -> None:
        """
        This methods sets the flight direction of this plane random to Direction.North, Direction.South, Direction.West
        or Direction.East
        :return: None
        :rtype: None
        """
        self.direction = Direction(np.random.randint(0, 4))
