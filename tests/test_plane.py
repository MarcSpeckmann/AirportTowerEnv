from unittest import TestCase

from gym_airport_tower.direction import Direction
from gym_airport_tower.plane import Plane


class TestPlane(TestCase):
    def test_move(self):
        # Test move east and free
        plane = Plane(position=[0, 0], idx=256, max_shape=(10, 9), direction=Direction.EAST)
        plane.move()
        self.assertEqual(plane.position, [0, 1])

        # Test move east and corner
        plane = Plane(position=[0, 8], idx=256, max_shape=(10, 9), direction=Direction.EAST)
        plane.move()
        self.assertIn(plane.direction, [Direction.WEST, Direction.SOUTH])
        if plane.direction == Direction.WEST:
            self.assertEqual(plane.position, [0, 7])
        elif plane.direction == Direction.SOUTH:
            self.assertEqual(plane.position, [1, 8])

        # Test move south and free
        plane = Plane(position=[0, 0], idx=256, max_shape=(10, 9), direction=Direction.SOUTH)
        plane.move()
        self.assertEqual(plane.position, [1, 0])

        # Test move south and corner
        plane = Plane(position=[9, 8], idx=256, max_shape=(10, 9), direction=Direction.SOUTH)
        plane.move()
        self.assertIn(plane.direction, [Direction.NORTH, Direction.WEST])
        if plane.direction == Direction.WEST:
            self.assertEqual(plane.position, [9, 7])
        elif plane.direction == Direction.NORTH:
            self.assertEqual(plane.position, [8, 8])

        # Test move west and free
        plane = Plane(position=[4, 6], idx=256, max_shape=(10, 9), direction=Direction.WEST)
        plane.move()
        self.assertEqual(plane.position, [4, 5])

        # Test move west and border
        plane = Plane(position=[4, 0], idx=256, max_shape=(10, 9), direction=Direction.WEST)
        plane.move()
        self.assertIn(plane.direction, [Direction.NORTH, Direction.EAST, Direction.SOUTH])
        if plane.direction == Direction.EAST:
            self.assertEqual(plane.position, [4, 1])
        elif plane.direction == Direction.NORTH:
            self.assertEqual(plane.position, [3, 0])
        elif plane.direction == Direction.SOUTH:
            self.assertEqual(plane.position, [5, 0])

        # Test move north and free
        plane = Plane(position=[4, 6], idx=256, max_shape=(10, 9), direction=Direction.NORTH)
        plane.move()
        self.assertEqual(plane.position, [3, 6])

        # Test move north and border
        plane = Plane(position=[0, 6], idx=256, max_shape=(10, 9), direction=Direction.NORTH)
        plane.move()
        self.assertIn(plane.direction, [Direction.WEST, Direction.EAST, Direction.SOUTH])
        if plane.direction == Direction.EAST:
            self.assertEqual(plane.position, [0, 7])
        elif plane.direction == Direction.WEST:
            self.assertEqual(plane.position, [0, 5])
        elif plane.direction == Direction.SOUTH:
            self.assertEqual(plane.position, [1, 6])

        # Test set discretion
        plane = Plane(position=[3, 7], idx=0, max_shape=(10, 9), direction=Direction.EAST)
        plane.move(direction=Direction.SOUTH)
        self.assertEqual(plane.direction, Direction.SOUTH)
        self.assertEqual(plane.position, [4, 7])

    def test_random_direction(self):
        south = 0
        north = 0
        west = 0
        east = 0
        plane = Plane(position=[4, 6], idx=256, max_shape=(10, 9), direction=Direction.NORTH)
        for i in range(1000000):
            plane._random_direction()
            if plane.direction == Direction.SOUTH:
                south += 1
            elif plane.direction == Direction.NORTH:
                north += 1
            elif plane.direction == Direction.EAST:
                east += 1
            elif plane.direction == Direction.WEST:
                west += 1
        self.assertEqual(north + south + west + east, 1000000)
        self.assertAlmostEqual(north / 1000000, 0.25, places=2)
        self.assertAlmostEqual(south / 1000000, 0.25, places=2)
        self.assertAlmostEqual(east / 1000000, 0.25, places=2)
        self.assertAlmostEqual(west / 1000000, 0.25, places=2)
