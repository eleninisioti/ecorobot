
import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State

from jax.experimental import host_callback as hcb
from xml.etree import ElementTree
import jax
import numpy as onp
from ecorobot.ecorobot.envs.foraging import Foraging
from ecorobot.ecorobot.envs.wall_following import WallFollowing
class DualTask:

    def __init__(self, robot_type, task_idx=0, num_rangefinders=6, **kwargs):

        self.envs = [
                     WallFollowing(robot_type=robot_type, **kwargs),Foraging(robot_type=robot_type,**kwargs),]

        self.env = self.envs[0]
        self.num_tasks = len(self.envs)



    def reset(self, key, task):

        self.env = self.envs[task]
        return self.env.reset(key)


    def step(self, state, action):


        return self.env.step(state, action)

