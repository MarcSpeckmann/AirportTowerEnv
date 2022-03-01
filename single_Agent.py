import numpy as np
import ray
from ray import tune
from ray.tune import register_env

from gym_airport_tower.airport_tower_env import AirportTowerEnv


def env_creator(env_config):
    return AirportTowerEnv(**env_config)

register_env("AirportTowerEnv", env_creator)

ray.init(include_dashboard=False)

config = {
    # === Settings for Rollout Worker processes ===
    "num_workers": 7,
    # "num_gpus":1,
    # "num_envs_per_worker": 1,
    "seed": tune.grid_search([42, 666, 123456789]),
    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": True,
    # Dense-layer setup for each the advantage branch and the value branch
    # in a dueling architecture.
    "hiddens": [256, 256],
    # Whether to use double dqn
    "double_q": True,
    # N-step Q learning
    "n_step": 1,

    # === Prioritized replay buffer ===
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Final value of beta (by default, we use constant beta=0.4).
    "final_prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    "prioritized_replay_beta_annealing_timesteps": 20000,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,

    # Callback to run before learning on a multi-agent batch of
    # experiences.
    "before_learn_on_batch": None,

    # The intensity with which to update the model (vs collecting samples
    # from the env). If None, uses the "natural" value of:
    # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
    # `num_envs_per_worker`).
    # If provided, will make sure that the ratio between ts inserted into
    # and sampled from the buffer matches the given value.
    # Example:
    #   training_intensity=1000.0
    #   train_batch_size=250 rollout_fragment_length=1
    #   num_workers=1 (or 0) num_envs_per_worker=1
    #   -> natural value = 250 / 1 = 250.0
    #   -> will make sure that replay+train op will be executed 4x as
    #      often as rollout+insert op (4 * 250 = 1000).
    # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further
    # details.
    "training_intensity": None,

    # === Parallelism ===
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # === Exploration
    "explore": True,
    "exploration_config": {
       # Exploration sub-class by name or full path to module+class
       # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
       "type": "EpsilonGreedy",
       # Parameters for the Exploration class' constructor:
       "initial_epsilon": 1.0,
       "final_epsilon": 0.02,
       "epsilon_timesteps": 950000,  # Timesteps over which to anneal epsilon.
    },
    # === Deep Learning Framework Settings ===
    "framework": "tf2",
    "eager_tracing": True,
    # === Environment Settings ===
    "env": 'AirportTowerEnv',
    "horizon": 200,
    "env_config": {
        "seed": 42, # seed gets set by ray
        "max_planes": tune.grid_search([1, 2]),
        "num_runways": tune.grid_search([1, 2]),
        "runway_length": tune.grid_search([3]),
        "airspace_size": tune.grid_search([(5, 5)]),
        "plane_spawn_probability_per_step": 0.3,
        "num_start_planes": 1,
        "landing_reward": 100,
        "plane_in_air_penalty": -1.0,
        "plane_on_runway_reward": 5,
        "render_env": False
    },
}

tune.run(
    "DQN",
    stop={"episode_len_mean": 200, 'timesteps_total': 1000000},
    checkpoint_at_end=True,
    num_samples=1,
    config=config,
    resume=False,
    local_dir="~/ray_results/basicDQNdifferentsingleENV256256"
)
