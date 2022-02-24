from unittest import TestCase

import numpy as np

from gym_airport_tower.airspace import Airspace
from gym_airport_tower.direction import Direction
from gym_airport_tower.plane import Plane
from gym_airport_tower.runway import Runway
from gym_airport_tower.runway_alignment import RunwayAlignment


class TestAirspace(TestCase):
    def test_reset(self):
        num_start_planes = 3
        airspace = Airspace(airspace_size=(10, 10), max_planes=5, num_runways=2, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=num_start_planes)
        airspace.reset()
        self.assertEqual(num_start_planes, len(airspace.planes))

    def test_air_space(self):
        airspace = Airspace(airspace_size=(5, 5), max_planes=2, num_runways=1, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=2)
        airspace.planes = []
        airspace.runways = []
        airspace.runways.append(
            Runway(position=(0, 0), runway_len=3, alignment=RunwayAlignment.VERTICAL, idx=airspace.max_planes))
        airspace.runways.append(
            Runway(position=(4, 2), runway_len=3, alignment=RunwayAlignment.HORIZONTAL, idx=airspace.max_planes + 1))
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(5, 5), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[2, 2], idx=1, max_shape=(5, 5), direction=Direction(
            np.random.randint(0, 4))))

        self.assertTrue((np.array(
            [[0, -1, -1, -1, -1],
             [2, -1, -1, -1, -1],
             [2, -1, 1, -1, -1],
             [-1, -1, -1, -1, -1],
             [-1, -1, 3, 3, 3]], dtype=np.int32) == airspace.air_space).all())

    def test_set_air_space(self):
        airspace = Airspace()
        with self.assertRaises(NotImplementedError):
            airspace.air_space = 4

    def test_spawn_random_plane(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=100, num_runways=0, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=0)
        airspace.spawn_random_plane(1)
        self.assertEqual(1, len(airspace.planes))
        airspace.planes = []
        for i in range(100):
            airspace.spawn_random_plane(0.33)

        self.assertAlmostEquals(len(airspace.planes) / 100, 0.33, places=1)

    def test_move_planes(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=3, num_runways=0, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[1, 0], idx=0, max_shape=(10, 10), direction=Direction.SOUTH))
        airspace.planes.append(Plane(position=[2, 0], idx=1, max_shape=(10, 10), direction=Direction.NORTH))

        airspace.move_planes((0, Direction.EAST.value))

        self.assertEqual(airspace.planes[0].position, [1, 1])
        self.assertEqual(airspace.planes[0].direction, Direction.EAST)
        self.assertEqual(airspace.planes[1].position, [1, 0])

    def test_check_direct_collision(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=3, num_runways=0, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[1, 0], idx=1, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))

        self.assertEqual(None, airspace.check_direct_collision())
        airspace.planes.append(Plane(position=[1, 0], idx=2, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        with self.assertRaises(ValueError):
            airspace.check_direct_collision()

    def test_check_north_west_collision(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=3, num_runways=0, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[1, 0], idx=0, max_shape=(10, 10), direction=Direction.SOUTH))
        airspace.planes.append(Plane(position=[2, 0], idx=1, max_shape=(10, 10), direction=Direction.NORTH))
        for plane in airspace.planes:
            plane.move()
        with self.assertRaises(ValueError):
            airspace.check_north_west_collision()

        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[1, 0], idx=0, max_shape=(10, 10), direction=Direction.NORTH))
        airspace.planes.append(Plane(position=[2, 0], idx=1, max_shape=(10, 10), direction=Direction.SOUTH))
        for plane in airspace.planes:
            plane.move()
        self.assertEqual(None, airspace.check_north_west_collision())

    def test_check_east_west_collision(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=3, num_runways=0, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[0, 1], idx=0, max_shape=(10, 10), direction=Direction.EAST))
        airspace.planes.append(Plane(position=[0, 2], idx=1, max_shape=(10, 10), direction=Direction.WEST))
        for plane in airspace.planes:
            plane.move()
        with self.assertRaises(ValueError):
            airspace.check_east_west_collision()

        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[0, 1], idx=0, max_shape=(10, 10), direction=Direction.WEST))
        airspace.planes.append(Plane(position=[0, 2], idx=1, max_shape=(10, 10), direction=Direction.EAST))
        for plane in airspace.planes:
            plane.move()
        self.assertEqual(None, airspace.check_east_west_collision())

    def test_generate_runways(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=2, num_runways=2, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=5)
        airspace.runways = []
        airspace.generate_runways()
        self.assertEqual(2, len(airspace.runways))

        with self.assertRaises(RuntimeError):
            airspace = Airspace(airspace_size=(10, 10), max_planes=5, num_runways=100, runway_length=3,
                                plane_spawn_probability_per_step=0.3, num_start_planes=5)

    def test_check_runway(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=3, num_runways=1, runway_length=4,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction.EAST))
        airspace.runways.append(
            Runway(position=(0, 0), runway_len=4, alignment=RunwayAlignment.VERTICAL, idx=airspace.max_planes))

        with self.assertRaises(ValueError):
            airspace.check_runway()

        airspace.planes = []
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction.NORTH))
        self.assertEqual(None, airspace.check_runway())

        airspace.planes = []
        airspace.runways = []
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction.SOUTH))
        airspace.runways.append(
            Runway(position=(0, 0), runway_len=4, alignment=RunwayAlignment.HORIZONTAL, idx=airspace.max_planes))

        with self.assertRaises(ValueError):
            airspace.check_runway()

        airspace.planes = []
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction.WEST))
        self.assertEqual(None, airspace.check_runway())

    def test_check_done_planes(self):
        num_start_planes = 3
        airspace = Airspace(airspace_size=(10, 10), max_planes=5, num_runways=2, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=num_start_planes)
        airspace.check_done_planes()
        self.assertEqual(num_start_planes, len(airspace.planes))
        airspace.planes[0].landed = True
        airspace.check_done_planes()
        self.assertEqual(num_start_planes - 1, len(airspace.planes))
        airspace.planes[0].landed = True
        airspace.planes[1].landed = True
        airspace.check_done_planes()
        self.assertEqual(0, len(airspace.planes))

    def test_num_planes_on_runways(self):
        airspace = Airspace(airspace_size=(10, 10), max_planes=5, num_runways=2, runway_length=3,
                            plane_spawn_probability_per_step=0.3, num_start_planes=1)
        airspace.planes = []
        airspace.runways = []
        airspace.runways.append(
            Runway(position=(0, 0), runway_len=4, alignment=RunwayAlignment.VERTICAL, idx=airspace.max_planes))
        airspace.runways.append(
            Runway(position=(9, 8), runway_len=2, alignment=RunwayAlignment.HORIZONTAL, idx=airspace.max_planes + 1))
        airspace.planes.append(Plane(position=[0, 0], idx=0, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[1, 0], idx=1, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[2, 0], idx=2, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[4, 3], idx=3, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))
        airspace.planes.append(Plane(position=[1, 6], idx=4, max_shape=(10, 10), direction=Direction(
            np.random.randint(0, 4))))

        self.assertEqual(3, airspace.num_planes_on_runways())
