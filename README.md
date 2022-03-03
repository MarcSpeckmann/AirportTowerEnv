# AirportTowerEnv

AirportTowerEnv is an OpenAi gym environment that simulates the tower of an airport. 
On a grid of size X x Y, N aeroplanes are to be landed on M runways of length l. In each step, the agent has the option of flying an aircraft to the north, east, south or west. All other aircraft will continue to move in the direction they are currently flying.
A plane is considered to have landed when it has crossed all the fields of a runway in the correct order.

## Installation

1. Download the Repository

```bash
git clone https://github.com/XxHalbfettxX/AirportTowerEnv.git
```

2. Create the environment

```bash
cd AirportTowerEnv
conda env create -f "environment.yml"
conda activate AirportTowerEnv
```

3. Run the Experiments

```bash
python3 single_Agent.py
```

```bash
python3 multi_Agent.py
```

4. Inspect the results

```bash
tensorboard --logdir=~/ray_results/
```
