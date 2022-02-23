from typing import Tuple

import gym
import numpy as np
from gym import spaces

from gym_airport_tower.airspace import Airspace


class AirportTowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_planes: int = 2,
                 num_runways: int = 1,
                 runway_length: int = 3,
                 airspace_size: Tuple[int, int] = (5, 5),
                 airplane_move_tiles_per_step: int = 1,
                 plane_spawn_probability_per_step=0.3,
                 num_start_planes: int = 1,
                 landing_reward: int = 100,
                 plane_in_air_penalty: int = -1,
                 plane_on_runway_reward: int = 5,
                 render_env: bool = False):
        # Fixed parameter / configurations
        # ===================================

        # self.seed: int = 666
        # np.random.seed(self.seed)
        self.max_planes: int = max_planes
        self.num_runways: int = num_runways
        self.runway_length: int = runway_length
        self.airspace_size: Tuple[int, int] = airspace_size
        self.airplane_move_tiles_per_step: int = airplane_move_tiles_per_step
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
                                           airplane_move_tiles_per_step=self.airplane_move_tiles_per_step,
                                           plane_spawn_probability_per_step=self.plane_spawn_probability_per_step,
                                           num_start_planes=self.num_start_planes)

        # GYM values
        # ===================================

        self.action_space = spaces.MultiDiscrete([self.max_planes, 4])
        self.observation_space = spaces.Box(low=-1, high=self.num_runways + self.max_planes,
                                            shape=self.airspace.air_space.shape, dtype=np.int32)
        self.reward_range = gym.spaces.Box(low=self.max_planes * self.plane_in_air_penalty,
                                           high=self.max_planes * self.landing_reward, shape=(1,), dtype=np.int32)

    def step(self, action: Tuple[int, int]):
        done = False
        reward = 0

        num_before_planes = len(self.airspace.planes)

        try:
            self.airspace.move_planes(actions=action)
        except ValueError as e:
            # print(e)
            done = True

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

    def reset(self):
        self.airspace.reset()
        if self.render_env:
            self.render()
        return self.airspace.air_space

    def render(self, mode="human"):
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
