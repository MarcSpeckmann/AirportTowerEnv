from typing import Dict, List, Tuple

import numpy as np

from gym_airport_tower.plane import Plane
from gym_airport_tower.runway_alignment import RunwayAlignment


class Runway:
    """
    The Runway object represent a Runway in the Airspace. It tracks the Planes which are currently on it. It also checks
    if the planes are landed correctly. Otherwise Exceptions get raised if crashed occurs.
    """

    def __init__(self, position: Tuple[int, int], idx: int, runway_len: int = 3,
                 alignment: RunwayAlignment = RunwayAlignment.HORIZONTAL) -> None:
        """
        :param position: Top/left position of Runway.
        :type position: Tuple[int, int]
        :param idx: Id of the Runway to identify it in the airspace
        :type idx: int
        :param runway_len: Length of runway in tiles.
        :type runway_len: int
        :param alignment: Defines if the Runway is aligned vertical or horizontal.
        :type alignment: RunwayAlignment
        """
        assert runway_len > 0
        self._position = position
        self.id = idx
        self._runway_len = runway_len
        self.alignment = alignment

        self._curr_planes: Dict[Plane, List[int]] = {}

    def get_slicing(self) -> np.lib.index_tricks.IndexExpression:
        """
        Returns the position of this runway as numpy sclicing.
        :return: The position of the runway in the Airspace as Slicing
        :rtype: numpy.lib.index_tricks.IndexExpression
        """
        if self.alignment == RunwayAlignment.VERTICAL:
            return np.s_[self._position[0]: self._position[0] + self._runway_len, self._position[1]]
        elif self.alignment == RunwayAlignment.HORIZONTAL:
            return np.s_[self._position[0], self._position[1]: self._position[1] + self._runway_len]
        else:
            raise ValueError

    def update_planes(self, planes: List[Plane]) -> None:
        """
        This method updates the planes which are currently on this Runway.
        If a plane enters the runway it will get registered on the runway.
        Planes which are already on the Runway gets updated. This means it gets saved in which order the travel over the
        runway.
        If planes leave the Runway and are not count as landed. It will raise a exception because it is count as a crash.
        :param planes: A list of planes which are on the runway at the moment.
        :type planes: List[Plane]
        :return: None
        :rtype: None
        """
        planes = planes.copy()
        # TODO: check plane position in Runway slice
        for plane in list(self._curr_planes.keys()):
            if plane not in planes:
                del self._curr_planes[plane]
                raise ValueError("Aircraft has come off the runway")
            else:
                planes.remove(plane)
                self._update_plane(plane)
        for plane in planes:
            self._curr_planes[plane] = [0 for _ in range(self._runway_len)]
            self._update_plane(plane)

    def _update_plane(self, plane: Plane) -> None:
        """
        This method updates the single information about a plane, which is currently on the Runway.
        It also checks that the plane traverse the runway in the right order or that the plane landed.
        :param plane: Plane to update on the Runway
        :type plane: Plane
        :return: None
        :rtype: None
        """
        assert plane in self._curr_planes
        if self.alignment == RunwayAlignment.VERTICAL:
            self._curr_planes[plane][int(plane.position[0] - self._position[0])] = max(self._curr_planes[plane]) + 1

        elif self.alignment == RunwayAlignment.HORIZONTAL:
            self._curr_planes[plane][int(plane.position[1] - self._position[1])] = max(self._curr_planes[plane]) + 1

        # self._check_landing_order(plane)

        self._check_landing_done(plane)

    def _check_landing_order(self, plane: Plane) -> bool:
        """
        Checks that the plane traverse the Runway in the right order. Otherwise an exception is thrown.
        :param plane: The plane to check
        :type plane: Plane
        :return: True if in order, otherwise exception
        :rtype: bool
        """
        assert plane in self._curr_planes
        land_arr = self._curr_planes[plane]

        right_left = True
        right_order = True
        for i in range(self._runway_len):
            if right_order and not land_arr[i] == i + 1:
                right_order = False
            if not right_order and not land_arr[i] == 0:
                right_left = False

        left_right = True
        left_order = True
        for i in reversed(range(self._runway_len)):
            if left_order and not land_arr[i] == self._runway_len - i:
                left_order = False
            if not left_order and not land_arr[i] == 0:
                left_right = False

        if not right_left and not left_right:
            raise ValueError("Aircraft changed direction on landing ......")
        else:
            return True

    def _check_landing_done(self, plane: Plane) -> None:
        """
        Checks that the Plane is landed.
        :param plane: The plane to check.
        :type plane: Plane
        :return: None
        :rtype: None
        """
        landing_arr = self._curr_planes[plane]
        if landing_arr == list(range(1, self._runway_len + 1)) or landing_arr == list(reversed(range(1, self._runway_len + 1))):
            del self._curr_planes[plane]
            plane.landed = True
