import itertools
from unittest import TestCase

import numpy as np

from gym_airport_tower.direction import Direction
from gym_airport_tower.plane import Plane
from gym_airport_tower.runway import Runway
from gym_airport_tower.runway_alignment import RunwayAlignment


class TestRunway(TestCase):
    def test_get_slicing(self):
        runway = Runway(position=(1, 1), idx=0, runway_len=3, alignment=RunwayAlignment.HORIZONTAL)
        self.assertEqual(np.s_[1, 1:4], runway.get_slicing(), )

        runway = Runway(position=(1, 1), idx=0, runway_len=3, alignment=RunwayAlignment.VERTICAL)
        self.assertEqual(np.s_[1:4, 1], runway.get_slicing())

        runway = Runway(position=(7, 1), idx=0, runway_len=8, alignment=RunwayAlignment.VERTICAL)
        self.assertEqual(np.s_[7:15, 1], runway.get_slicing())

        with self.assertRaises(AssertionError):
            Runway(position=(1, 1), idx=0, runway_len=0, alignment=RunwayAlignment.VERTICAL)

        with self.assertRaises(AssertionError):
            Runway(position=(1, 1), idx=0, runway_len=-3, alignment=RunwayAlignment.HORIZONTAL)

    def test_update_planes(self):
        # Check adding plane
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.HORIZONTAL)
        planes = [Plane(position=[2, 2], idx=1, direction=Direction.WEST, max_shape=(4, 4))]
        self.assertEqual(len(runway._curr_planes), 0)
        runway.update_planes(planes)
        self.assertIn(planes[0], runway._curr_planes)
        self.assertEqual(len(runway._curr_planes), 1)

        # check already existing plane
        [plane.move() for plane in planes]
        runway.update_planes(planes)
        self.assertIn(planes[0], runway._curr_planes)
        self.assertEqual(len(runway._curr_planes), 1)

        # check removed plane
        with self.assertRaises(ValueError):
            runway.update_planes([])

        # check with multiple planes
        # Check adding plane
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.HORIZONTAL)
        planes = [Plane(position=[0, 2], idx=1, direction=Direction.WEST, max_shape=(4, 4)),
                  Plane(position=[0, 1], idx=1, direction=Direction.WEST, max_shape=(4, 4))]
        self.assertEqual(len(runway._curr_planes), 0)
        runway.update_planes(planes)
        self.assertIn(planes[0], runway._curr_planes)
        self.assertIn(planes[1], runway._curr_planes)
        self.assertEqual(len(runway._curr_planes), 2)

        # check already existing plane
        [plane.move() for plane in planes]
        runway.update_planes(planes)
        self.assertIn(planes[0], runway._curr_planes)
        self.assertIn(planes[1], runway._curr_planes)
        self.assertEqual(len(runway._curr_planes), 2)

        # check removed plane
        with self.assertRaises(ValueError):
            runway.update_planes([])

    def test_update_plane(self):
        # Horizontal west approach
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.HORIZONTAL)
        plane = Plane(position=[2, 2], idx=1, direction=Direction.WEST, max_shape=(4, 4))
        runway.update_planes([plane])
        self.assertEqual(runway._curr_planes[plane], [0, 0, 1])
        plane.move()
        runway._update_plane(plane)
        self.assertEqual(runway._curr_planes[plane], [0, 2, 1])
        plane.move()
        runway._update_plane(plane)
        self.assertNotIn(plane, runway._curr_planes)

        # Horizontal east approach
        runway = Runway(position=(1, 1), idx=0, runway_len=3, alignment=RunwayAlignment.HORIZONTAL)
        plane = Plane(position=[1, 1], idx=1, direction=Direction.EAST, max_shape=(4, 4))
        runway.update_planes([plane])
        self.assertEqual(runway._curr_planes[plane], [1, 0, 0])
        plane.move()
        runway._update_plane(plane)
        self.assertEqual(runway._curr_planes[plane], [1, 2, 0])
        plane.move()
        runway._update_plane(plane)
        self.assertNotIn(plane, runway._curr_planes)

        # Vertical south approach
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.VERTICAL)
        plane = Plane(position=[0, 0], idx=1, direction=Direction.SOUTH, max_shape=(4, 4))
        runway.update_planes([plane])
        self.assertEqual(runway._curr_planes[plane], [1, 0, 0])
        plane.move()
        runway._update_plane(plane)
        self.assertEqual(runway._curr_planes[plane], [1, 2, 0])
        plane.move()
        runway._update_plane(plane)
        self.assertNotIn(plane, runway._curr_planes)

        # Vertical north approach
        runway = Runway(position=(0, 0), idx=0, runway_len=4, alignment=RunwayAlignment.VERTICAL)
        plane = Plane(position=[3, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))
        runway.update_planes([plane])
        self.assertEqual(runway._curr_planes[plane], [0, 0, 0, 1])
        plane.move()
        runway._update_plane(plane)
        self.assertEqual(runway._curr_planes[plane], [0, 0, 2, 1])
        plane.move()
        runway._update_plane(plane)
        self.assertEqual(runway._curr_planes[plane], [0, 3, 2, 1])
        plane.move()
        runway._update_plane(plane)
        self.assertNotIn(plane, runway._curr_planes)

        # 2 same time
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.VERTICAL)
        plane1 = Plane(position=[2, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))
        plane2 = Plane(position=[3, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))
        runway.update_planes([plane1])
        self.assertEqual(runway._curr_planes[plane1], [0, 0, 1])
        plane1.move()
        plane2.move()
        runway.update_planes([plane1, plane2])
        self.assertEqual(runway._curr_planes[plane1], [0, 2, 1])
        self.assertEqual(runway._curr_planes[plane2], [0, 0, 1])
        plane1.move()
        plane2.move()
        runway._update_plane(plane1)
        runway._update_plane(plane2)
        self.assertNotIn(plane1, runway._curr_planes)
        self.assertEqual(runway._curr_planes[plane2], [0, 2, 1])
        plane2.move()
        runway._update_plane(plane2)
        self.assertNotIn(plane2, runway._curr_planes)

    def test_check_landing_order(self):
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.VERTICAL)
        plane1 = Plane(position=[2, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))

        # test size = 3
        for perm in itertools.product([1, 2, 3, 0], repeat=3):
            runway._curr_planes[plane1] = list(perm)
            if list(perm) in [[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3], [0, 0, 1], [0, 2, 1], [3, 2, 1]]:
                self.assertEqual(runway._check_landing_order(plane1), True)
            else:
                with self.assertRaises(ValueError):
                    runway._check_landing_order(plane1)

    def test_check_landing_done(self):
        runway = Runway(position=(0, 0), idx=0, runway_len=3, alignment=RunwayAlignment.VERTICAL)
        plane1 = Plane(position=[2, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))
        plane2 = Plane(position=[3, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))

        # both direction done
        runway._curr_planes[plane1] = [1, 2, 3]
        runway._curr_planes[plane2] = [3, 2, 1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertNotIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, True)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertNotIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, True)

        # not done
        plane1.landed = False
        plane2.landed = False
        runway._curr_planes[plane1] = [1, 2, 0]
        runway._curr_planes[plane2] = [0, 2, 1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, False)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, False)

        runway._curr_planes[plane1] = [1, 0, 0]
        runway._curr_planes[plane2] = [0, 0, 1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, False)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, False)

        runway._curr_planes[plane1] = [0, 0, 0]
        runway._curr_planes[plane2] = [0, 0, 0]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, False)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, False)

        runway._curr_planes[plane1] = [1, -1, 4]
        runway._curr_planes[plane2] = [-3, -2, -1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, False)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, False)

        # more runway length
        runway = Runway(position=(0, 0), idx=0, runway_len=9, alignment=RunwayAlignment.VERTICAL)
        plane1 = Plane(position=[8, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))
        plane2 = Plane(position=[9, 0], idx=1, direction=Direction.NORTH, max_shape=(4, 4))

        runway._curr_planes[plane1] = [1, 2, 3, 4, 5, 6, 7, 8, 0]
        runway._curr_planes[plane2] = [0, 0, 0, 0, 5, 4, 3, 2, 1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, False)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, False)

        runway._curr_planes[plane1] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        runway._curr_planes[plane2] = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.assertEqual(plane1.landed, False)
        runway._check_landing_done(plane1)
        self.assertNotIn(plane1, runway._curr_planes)
        self.assertEqual(plane1.landed, True)

        self.assertEqual(plane2.landed, False)
        runway._check_landing_done(plane2)
        self.assertNotIn(plane2, runway._curr_planes)
        self.assertEqual(plane2.landed, True)
