from typing import List

import numpy as np

from gym_airport_tower.direction import Direction


class Plane:
    def __init__(self, position: List[int], idx, max_shape, direction=Direction(np.random.randint(0, 4))):
        self.id = idx
        self.position = position
        self.direction: Direction = direction
        self.max_shape = max_shape
        self.landed = False

    def move(self, direction: Direction = None):
        if direction:
            self.direction = direction
        if self.direction == Direction.SOUTH:
            if self.position[0] < (self.max_shape[0] - 1):
                self.position[0] += 1
            else:
                self.random_direction()
                self.move()
        elif self.direction == Direction.WEST:
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
        elif self.direction == Direction.EAST:
            if self.position[1] >= 1:
                self.position[1] -= 1
            else:
                self.random_direction()
                self.move()

    def random_direction(self):
        self.direction = Direction(np.random.randint(0, 4))
