from typing import Dict, List

import numpy as np

from gym_airport_tower.plane import Plane
from gym_airport_tower.runway_alignment import RunwayAlignment


class Runway:
    def __init__(self, position, idx, runway_len=3, alignment=RunwayAlignment.HORIZONTAL):
        self.position = position
        self.id = idx
        self.runway_len = runway_len
        self.alignment = alignment

        self.curr_planes: Dict[Plane, List[int]] = {}

    def get_slicing(self):
        if self.alignment == RunwayAlignment.VERTICAL:
            return np.s_[self.position[0]: self.position[0] + self.runway_len, self.position[1]]
        elif self.alignment == RunwayAlignment.HORIZONTAL:
            return np.s_[self.position[0], self.position[1]: self.position[1] + self.runway_len]
        else:
            raise ValueError

    def update_planes(self, planes: List[Plane]):
        for plane in list(self.curr_planes.keys()):
            if plane not in planes:
                del self.curr_planes[plane]
                raise ValueError("Aircraft has come off the runway")
            else:
                planes.remove(plane)
                self.update_plane(plane)
        for plane in planes:
            self.curr_planes[plane] = [0 for _ in range(self.runway_len)]
            self.update_plane(plane)

    def update_plane(self, plane):
        if self.alignment == RunwayAlignment.VERTICAL:
            self.curr_planes[plane][int(plane.position[0] - self.position[0])] = max(self.curr_planes[plane]) + 1

        elif self.alignment == RunwayAlignment.HORIZONTAL:
            self.curr_planes[plane][int(plane.position[1] - self.position[1])] = max(self.curr_planes[plane]) + 1

        # self.check_landing_order(plane)  #TODO: activate if working

        self.check_landing_done(plane)

    def check_landing_order(self, plane):
        right_left = True
        left_right = True
        land_arr = self.curr_planes[plane]
        right_order = True
        left_order = True
        for i in range(self.runway_len - 1):
            if right_order and not land_arr[i] != i + 1:
                right_order = False
            if not right_order and not land_arr[i] == 0:
                right_left = False
        for i in range(self.runway_len - 1, -1, -1):
            if left_order and not land_arr[i] != self.runway_len - i:
                left_order = False
            if not left_order and not land_arr[i] == 0:
                left_right = False
        if not right_left and not left_right:
            raise ValueError("Aircraft changed direction on landing ......")

    def check_landing_done(self, plane):
        landing_arr = self.curr_planes[plane]
        if landing_arr == list(range(1, self.runway_len + 1)) or landing_arr == list(reversed(range(1, 3 + 1))):
            #print("LANDED HURRAY")
            del self.curr_planes[plane]
            plane.landed = True
