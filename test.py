from time import sleep

from gym_airport_tower.airport_tower_env import AirportTowerEnv

env = AirportTowerEnv()
done = False
env.render()
while not done:
    _, _, done, _ = env.step((23, 2))
    print(done)
    env.render()
    sleep(1)
