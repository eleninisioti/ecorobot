""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot.robots.base import RobotWrapper
import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State
from ecorobot.ecorobot.modules.food import Food
from ecorobot.ecorobot import robots
import brax
from ecorobot.ecorobot.sensors import Compass
import numpy as onp
class ForagingSensor(EcorobotEnv):

    def __init__(self, robot_type, episode_length=0, **kwargs):
        super().__init__()
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add food
        food = Food(loc="random")
        self.target = food
        self.add_module(food)

        sensor = Compass(food, self.robot.torso_size)
        self.add_module(sensor)
        self.compass = sensor

        self.init_sys()


    def reset(self, key):

        new_state = super().reset(key)
        metrics = {
            'reward_forward': 0.0,
            'reward_food': 0.0,
            'reward_survive': 0.0,
            'reward_ctrl': 0.0,
            'x_position': 0.0,
            'y_position': 0.0,
            'food_position_x':0.0,
            'food_position_y':0.0,
            'food_position_z': 0.0,

            'distance_from_origin': 0.0,
            'distance_to_target': 0.0,
            'x_velocity': 0.0,
            'y_velocity': 0.0,
        }

        obs = self._get_obs(new_state.pipeline_state)


        return new_state.replace(metrics=metrics, obs=obs)

    def step(self, state, action):
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)

        # calculate food reward
        robot_pos = pipeline_state.x.pos[0]

        target_pos = pipeline_state.x.pos[self.target.pos_idx]
        # target_pos = self.target.location

        #distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
        # distance_to_target = 10.0
        #reward_food = 1 - (distance_to_target / self.target.max_distance)


        #target_pos = pipeline_state.q[-self.target.info_size:]

        # reposition sensor
        sensor_pos = self.compass.reset(pipeline_state)
        new_q = pipeline_state.q.at[self.compass.q_idx:self.compass.q_idx+self.compass.info_size].set(sensor_pos)
        new_pos = pipeline_state.x.pos.at[self.compass.pos_idx].set(sensor_pos)
        x_new = pipeline_state.x.replace(pos=new_pos)
        pipeline_state = pipeline_state.replace(x=x_new, q=new_q)
        #pipeline_state = self.pipeline_init(new_q, pipeline_state.qd)

        target_pos = pipeline_state.x.pos[self.target.pos_idx]
        reward_food, distance_to_target = self.compass.get_reward(pipeline_state)

        state.metrics.update(
            reward_food=reward_food,
        food_position_x=target_pos[0],
        food_position_y=target_pos[1],
        food_position_z=target_pos[2],
        distance_to_target=distance_to_target)

        state = state.replace(obs=obs, reward=reward_food, pipeline_state=pipeline_state)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        # we need to remove the location of the food from the qpos and qvel
        modules_info = onp.sum([el.info_size for el in self.modules])
        qpos = qpos[:-modules_info]
        qvel = qvel[:-modules_info]


        ## get info from sensors
        #sensor_info = self.target_compass.get_obs(pipeline_state)

        #sensor_pos = self.target_compass.reset(pipeline_state)

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        robot_pos = pipeline_state.x.pos[0]
        to_food = self.compass.get_obs(pipeline_state)

        #target_pos = pipeline_state.q[self.target.q_idx:self.target.q_idx+self.target.info_size]
        #to_food = robot_pos-target_pos

        #distance = jnp.expand_dims(self.compass.get_distance(pipeline_state),axis=0)

        return jnp.concatenate([qpos] + [qvel] + [to_food] )


