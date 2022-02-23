import ray
from ray import tune
from ray.rllib.agents.pg import pg
from ray.tune import register_env

from gym_airport_tower.airport_tower_env import AirportTowerEnv


def env_creator(env_config):
    return AirportTowerEnv(**env_config)


register_env("AirportTowerEnv", env_creator)

ray.init()
tune.run(
    pg.PGTrainer,
    stop={"episode_len_mean": 200, 'timesteps_total': 100000},
    checkpoint_at_end=True,
    config={
        # === Settings for Rollout Worker processes ===
        "num_workers": 2,
        "num_envs_per_worker": 1,
        # === Settings for the Trainer process ===
        "gamma": 0.99,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        #"train_batch_size": 200,
        #"model": {
        #    "fcnet_hiddens": [64, 64],
        #    "fcnet_activation": "relu",
        #},
        #"optimizer": {},
        # === Deep Learning Framework Settings ===
        "framework": "torch",
        # === Environment Settings ===
        "env": 'AirportTowerEnv',
        "horizon": 200,
        "env_config": {
            "max_planes": 1,
            "num_runways": 1,
            "runway_length": 3,
            "airspace_size": (5, 5),
            "airplane_move_tiles_per_step": 1,
            "plane_spawn_probability_per_step": 0.3,
            "num_start_planes":  1,
            "landing_reward":  100,
            "plane_in_air_penalty": -1,
            "plane_on_runway_reward":  5,
            "render_env":  False
        },
        # === Evaluation Settings ===
        "evaluation_num_workers": 1,
        "evaluation_interval": 10,
        # === Exploration Settings ===
        "explore": True,
        "exploration_config": {
            "type": "StochasticSampling",
        }
    },
)
