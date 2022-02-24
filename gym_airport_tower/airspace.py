from typing import Tuple, List

import numpy as np

from gym_airport_tower.direction import Direction
from gym_airport_tower.plane import Plane
from gym_airport_tower.runway import Runway
from gym_airport_tower.runway_alignment import RunwayAlignment


class Airspace:
    def __init__(self, airspace_size: Tuple[int, int] = (10, 10), max_planes: int = 5, num_runways: int = 1,
                 runway_length: int = 3, plane_spawn_probability_per_step: float = 0.3, num_start_planes: int = 1):
        """
        :param airspace_size: Array size of airspace
        :type airspace_size: Tuple[int, int]
        :param max_planes: Highest number of planes allowed in Airspace
        :type max_planes: int
        :param num_runways: Number of runways in Airspace
        :type num_runways: int
        :param runway_length: Length of runway in Airspace
        :type runway_length: int
        :param plane_spawn_probability_per_step: Probability of spawning an new plane each step.
        :type plane_spawn_probability_per_step: float
        :param num_start_planes: Number of planes in Airspace on creation
        :type num_start_planes: int
        """
        assert num_start_planes <= max_planes
        assert runway_length > 0
        assert 0 <= plane_spawn_probability_per_step <= 1

        self.max_planes: int = max_planes
        self.num_runways: int = num_runways
        self.runway_length: int = runway_length
        self.airspace_size: Tuple[int, int] = airspace_size
        self.plane_spawn_probability_per_step: float = plane_spawn_probability_per_step
        self.num_start_planes: int = num_start_planes

        self.planes: List[Plane] = []
        self.runways: List[Runway] = []
        self.generate_runways()
        for _ in range(self.num_start_planes):
            self.spawn_random_plane(spawn_prob=1)

    def reset(self) -> None:
        """
        Removes all current planes. And spawns new planes. Therefore the Airspace is in default state.
        :return: None
        :rtype: None
        """
        self.planes = []
        for _ in range(self.num_start_planes):
            self.spawn_random_plane(spawn_prob=1)

    @property
    def air_space(self) -> np.ndarray:
        """
        Generates array of airspace.
        :return: array of airspace
        :rtype: np.ndarray
        """
        airspace = np.zeros(self.airspace_size, dtype=np.int32) - 1
        for runway in self.runways:
            airspace[runway.get_slicing()] = runway.id
        for plane in self.planes:
            airspace[plane.position[0], plane.position[1]] = plane.id
        return airspace

    @air_space.setter
    def air_space(self, value):
        raise NotImplementedError

    def spawn_random_plane(self, spawn_prob: float = None):
        """
        Spawn new planes.
        :param spawn_prob: The probability how like it is that a new plane is spawned.
        :type spawn_prob: float
        :return: None
        :rtype: None
        """
        if not spawn_prob:
            spawn_prob = self.plane_spawn_probability_per_step
        airspace = self.air_space
        if len(self.planes) < self.max_planes and np.random.choice([0, 1], size=1, p=[1 - spawn_prob, spawn_prob]):
            avail_pos = np.where(airspace == -1)
            if len(avail_pos[0]) <= 0:
                raise RuntimeError("No space for more planes.")
            new_pos_ind = np.random.randint(low=0, high=len(avail_pos[0]) - 1)
            given_ids = [plane.id for plane in self.planes]
            free_ids = [i for i in range(0, self.max_planes) if i not in given_ids]
            self.planes.append(Plane([avail_pos[0][new_pos_ind], avail_pos[0][new_pos_ind]], max_shape=airspace.shape,
                                     idx=free_ids[0]))

    def move_planes(self, actions: Tuple[int, int]) -> None:
        """
        Moves all planes.
        :param actions: Move the plane with given id into direction
        :type actions: Tuple[int, int]
        :return: None
        :rtype: None
        """
        for plane in self.planes:
            if plane.id == actions[0]:
                plane.move(Direction(actions[1]))
            else:
                plane.move()

        self.check_direct_collision()

        self.check_north_west_collision()

        self.check_east_west_collision()

        self.check_runway()

        self.check_done_planes()

    def check_direct_collision(self) -> None:
        """
        Raises an Exception if two planes are one the same position.
        :return: None
        :rtype: None
        """
        if len(self.planes) > 1 and len(
                set([(int(plane.position[0]), int(plane.position[1])) for plane in self.planes])) < len(self.planes):
            raise ValueError("Plane collision detected")

    def check_north_west_collision(self) -> None:
        """
        Check if planes crashes, when they come from opposite direction(east, west). Because thereby they can crash
        without having the same internal position.
        [-1, 1, 0, -1, -1] ->  [-1, 0, 1, -1, -1]
        On crash an Exception gets thrown.
        :return: None
        :rtype: None
        """
        north = [plane.position.copy() for plane in self.planes if plane.direction == Direction.NORTH]
        south = [plane.position.copy() for plane in self.planes if plane.direction == Direction.SOUTH]
        for plane_s in south:
            for plane_n in north:
                plane_s[0] = plane_s[0] - 1
                if plane_n == plane_s:
                    raise ValueError("Plane collision by north/south crossing detected.")

    def check_east_west_collision(self):
        """
        Check if planes crashes, when they come from opposite direction(north, south). Because thereby they can crash
        without having the same internal position.
        [-1, 1, 0, -1, -1] ->  [-1, 0, 1, -1, -1]
        On crash an Exception gets thrown.
        :return: None
        :rtype: None
        """
        east = [plane.position.copy() for plane in self.planes if plane.direction == Direction.EAST]
        west = [plane.position.copy() for plane in self.planes if plane.direction == Direction.WEST]
        for plane_e in east:
            for plane_w in west:
                plane_e[1] = plane_e[1] - 1
                if plane_w == plane_e:
                    raise ValueError("Plane collision by west/east crossing detected.")

    def generate_runways(self) -> None:
        """
        Creates self.num_runways many runways. Position and alignment get chosen random.
        :return: None
        :rtype: None
        """
        for i in range(self.max_planes, self.max_planes + self.num_runways):
            airspace = self.air_space
            create_limit = 30
            create_counter = 0
            while create_counter < create_limit:
                create_counter += 1
                horr = np.random.randint(2, size=1)
                if horr:
                    idx = (np.random.choice(airspace.shape[0] - self.runway_length, size=1)[0],
                           np.random.choice(airspace.shape[1], size=1)[0])
                else:
                    idx = (np.random.choice(airspace.shape[0], size=1)[0],
                           np.random.choice(airspace.shape[1] - self.runway_length, size=1)[0])
                if horr and np.all(airspace[idx[0]: idx[0] + self.runway_length, idx[1]] == -1):
                    self.runways.append(
                        Runway(position=idx, runway_len=self.runway_length, alignment=RunwayAlignment.VERTICAL,
                               idx=i))
                    break
                elif not horr and np.all(airspace[idx[0], idx[1]: idx[1] + self.runway_length] == 0):
                    self.runways.append(
                        Runway(position=idx, runway_len=self.runway_length, alignment=RunwayAlignment.HORIZONTAL,
                               idx=i))
                    break
            else:
                # TODO: check if creation is im/possible instead of trying
                raise RuntimeError("No runway creation possible after {} tries.".format(create_limit))

    def check_runway(self) -> None:
        """
        Checks if a plane approaches a runway from the wrong direction. Only approachig from the beginning and end are
        allowed.
        On wrong approach an Exception gets thrown.
        The runways itself also run some checks internally.
        :return: None
        :rtype: None
        """
        airs = self.air_space
        for runway in self.runways:
            uniq = np.unique(airs[runway.get_slicing()])
            planes_ids = []
            for idx in uniq:
                if 0 <= idx < self.max_planes:
                    planes_ids.append(idx)
            planes = []
            if runway.alignment == RunwayAlignment.HORIZONTAL:
                planes = [plane for plane in self.planes if (plane.id in planes_ids and (
                        plane.direction == Direction.WEST or plane.direction == Direction.EAST))]
            elif runway.alignment == RunwayAlignment.VERTICAL:
                planes = [plane for plane in self.planes if (plane.id in planes_ids and (
                        plane.direction == Direction.NORTH or plane.direction == Direction.SOUTH))]

            if len(planes_ids) > len(planes):
                raise ValueError("Aircraft approached from the wrong direction and crashed.")

            runway.update_planes(planes)

    def check_done_planes(self) -> None:
        """
        Removes all planes which counts as landed.
        :return: None
        :rtype: None
        """
        dels = []
        for i in range(len(self.planes)):
            if self.planes[i].landed:
                dels.append(self.planes[i])
        for plane in dels:
            self.planes.remove(plane)

    def num_planes_on_runways(self) -> int:
        """
        Calculates the number of planes which are currently landing.
        :return: Number of planes which are currently on runways/landing.
        :rtype: int
        """
        airs = self.air_space
        planes_on_runway = 0
        for runway in self.runways:
            uniq = np.unique(airs[runway.get_slicing()])
            for idx in uniq:
                if idx < self.max_planes:
                    planes_on_runway += 1
        return planes_on_runway
