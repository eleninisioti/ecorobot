""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State
from ecorobot.modules.food import Food
from ecorobot import robots
import brax


class Foraging(EcorobotEnv):

    def __init__(self, robot_type,episode_length, **kwargs):
        super().__init__(episode_length=episode_length, **kwargs)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add food
        food = Food(loc_type="radial", loc=[2.0, 0.0])
        self.target = food
        self.add_module(food)

        self.num_rangefinders = num_rangefinders
        for rangefinder in range(self.num_rangefinders):
            sensor = Rangefinder(self.modules, self.robot.torso_size,
                                 id=rangefinder,
                                 name="rangefinder_"+str(rangefinder),
                                 num_rangefinders=self.num_rangefinders,
                                 modules=self.modules)
            self.add_module(sensor)

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
        #target_pos = pipeline_state.q[-self.target.info_size:]

        target_pos = pipeline_state.x.pos[self.target.pos_idx]
        #target_pos = self.target.location

        distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
        #distance_to_target = 10.0
        reward_food = 1 - (distance_to_target / self.target.max_distance)

        reward = reward_food

        state.metrics.update(
            reward_food=reward_food,
        food_position_x=target_pos[0],
        food_position_y=target_pos[1],
        food_position_z=target_pos[2],
        distance_to_target=distance_to_target)

        state = state.replace(obs=obs, reward=reward)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        # we need to remove the location of the food from the qpos and qvel
        qpos = qpos[:-self.target.info_size]
        qvel = qvel[:-self.target.info_size]


        ## get info from sensors
        #sensor_info = self.target_compass.get_obs(pipeline_state)

        #sensor_pos = self.target_compass.reset(pipeline_state)

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        target_pos = pipeline_state.q[-self.target.info_size:]
        to_food = pipeline_state.x.pos[0]-target_pos

        return jnp.concatenate([qpos] + [qvel] + [to_food])


