import math
from typing import Tuple

import gym
import numpy as np

from gym_airport_tower.airspace import Airspace


class AirportTowerEnv(gym.Env):
    """
    AirportTowerEnv is an OpenAi gym environment that simulates the tower of an airport.
    On a grid of size X x Y, N aeroplanes are to be landed on M runways of length l.
    In each step, the agent has the option of flying an aircraft to the north, east, south or west.
    All other aircraft will continue to move in the direction they are currently flying.
    A plane is considered to have landed when it has crossed all the fields of a runway in the correct order.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_planes: int = 2,
                 num_runways: int = 1,
                 runway_length: int = 3,
                 airspace_size: Tuple[int, int] = (5, 5),
                 plane_spawn_probability_per_step=0.3,
                 num_start_planes: int = 1,
                 landing_reward: int = 100,
                 plane_in_air_penalty: int = -1,
                 plane_on_runway_reward: int = 5,
                 seed: int = 666,
                 render_env: bool = False):
        # Fixed parameter / configurations
        # ===================================

        self._seed: int = seed
        np.random.seed(self._seed)
        self.max_planes: int = max_planes
        self.num_runways: int = num_runways
        self.runway_length: int = runway_length
        self.airspace_size: Tuple[int, int] = airspace_size
        self.plane_spawn_probability_per_step = plane_spawn_probability_per_step
        self.num_start_planes: int = num_start_planes
        self.landing_reward: int = landing_reward
        self.plane_in_air_penalty: int = plane_in_air_penalty
        self.plane_on_runway_reward: int = plane_on_runway_reward
        self.render_env: bool = render_env

        # Active parameter
        # ===================================
        self.airspace: Airspace = Airspace(airspace_size=self.airspace_size, max_planes=self.max_planes,
                                           num_runways=self.num_runways, runway_length=self.runway_length,
                                           plane_spawn_probability_per_step=self.plane_spawn_probability_per_step,
                                           num_start_planes=self.num_start_planes)

        # GYM values
        # ===================================

        self.action_space = gym.spaces.Discrete(self.max_planes * 4)
        self.observation_space = gym.spaces.Box(low=-1, high=self.num_runways + self.max_planes,
                                                shape=self.airspace.air_space.shape, dtype=np.int32)
        self.reward_range = gym.spaces.Box(low=self.max_planes * self.plane_in_air_penalty,
                                           high=self.max_planes * self.landing_reward, shape=(1,), dtype=np.int64)

    def step(self, action: int):
        """
        Implement gym step method,
        :param action: The action from the agent (planeID, direction)
        :type action: Tuple[int, int]
        :return:
        :rtype:
        """
        done = False
        reward = 0

        num_before_planes = len(self.airspace.planes)
        # Move all planes/fulfill action from agent
        try:
            self.airspace.move_planes(actions=[(math.floor(action / 4), action % 4)])
        except ValueError:
            done = True

        # Check number of landed planes
        num_after_planes = len(self.airspace.planes)
        landed_planes = num_before_planes - num_after_planes

        # Reward for landed planes
        reward += landed_planes * self.landing_reward

        # Penalty for planes in the air
        reward -= num_after_planes * self.plane_in_air_penalty

        # Support reward for planes on runway
        reward += self.airspace.num_planes_on_runways() * self.plane_on_runway_reward

        if self.render_env:
            self.render()

        # random spawn planes
        self.airspace.spawn_random_plane()

        return self.airspace.air_space, reward, done, {}

    def reset(self) -> np.ndarray:
        """
        Implement Gym reset method
        :return: Observation of Airspace
        :rtype: np.ndarray
        """
        self.airspace.reset()
        if self.render_env:
            self.render()
        return self.airspace.air_space

    def render(self, mode="human") -> None:
        """
        Implement Gym render method.
        :param mode: Render mode
        :type mode: str
        :return: None
        :rtype: None
        """
        airspace = self.airspace.air_space
        if mode == 'human':
            max_val = np.max(airspace)
            decs = int(len(str(max_val)))
            print(
                int(airspace.shape[1] * 1.5) * "=" + "  {}  ".format(self) + int(
                    airspace.shape[1] * 1.5) * "=")

            display_airspace = np.array2string(airspace)
            display_airspace = display_airspace.replace('[[', '| ')
            display_airspace = display_airspace.replace(' [', '| ')
            display_airspace = display_airspace.replace(']]', ' |')
            display_airspace = display_airspace.replace(']', ' |')

            display_airspace = display_airspace.replace(' -1 ', '    ')
            display_airspace = display_airspace.replace(' -1 ', '    ')

            print('+' + (((decs + 2) * (airspace.shape[1])) + 1) * '-' + '+')
            print(display_airspace)
            print('+' + (((decs + 2) * (airspace.shape[1])) + 1) * '-' + '+\n')
        else:
            raise ValueError("Unknown render type")

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(self._seed)
