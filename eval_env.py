from sim.environment import CarlaGymEnv
from sim.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, LaneInvasionSensor
from sim.connection import ClientConnection

from sim.settings import *
import numpy as np

town = "Town03"

ckpt_freq = 10
continuous_action=True


try:
    client, world = ClientConnection(town).setup()
    print("Connection has been setup successfully.")
except:
    print("Connection has been refused by the server.")
    ConnectionRefusedError


env = CarlaGymEnv(client, world, town)
obs = env.reset()

act = np.array([-0.11248506,  0.85606664], dtype=np.float32)
observation, reward, done, info = env.step(act)
import pdb; pdb.set_trace()


