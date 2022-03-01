from time import sleep

from gym_airport_tower.airport_tower_env import AirportTowerEnv

env = AirportTowerEnv()
done = False
env.reset()
env.render()
while not done:
    _, _, done, _ = env.step(env.action_space.sample())
    env.render()
    sleep(1)
env.close()
