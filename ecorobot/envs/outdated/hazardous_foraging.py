""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot.robots.base import RobotWrapper
from ecorobot.ecorobot.sensors import Rangefinder
from ecorobot.ecorobot.modules.hazard import Hazard

import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State
from ecorobot.ecorobot.modules.food import Food
from ecorobot.ecorobot import robots
import brax
import jax.numpy as jp
import numpy as onp
import random

class HazardousForaging(EcorobotEnv):

    def __init__(self,  robot_type, project_dir="projects/metaneat_benchmarking/metaneat_djo/hazardous",num_rangefinders=6,episode_length=0, reset_task=0,rollout_length=0, **kwargs):
        super().__init__(episode_length=episode_length, project_dir=project_dir)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add food
        self.num_hazards = 16
        offset = 4
        food = Food(loc_type="random_hazards", num_hazards=self.num_hazards)
        self.target = food
        self.add_module(food)

        # place hazards

        self.hazard_locs = []
        for hazard in range(self.num_hazards):
            x = hazard/(self.num_hazards/4)*2 + offset
            y = hazard%(self.num_hazards/4)*2 + offset

            loc = [x, y, 0]
            hazard = Hazard(xml_idx=hazard,
                            loc=loc,
                           name="hazard_"+str(hazard)
                            )
            self.add_module(hazard)
            self.hazard_locs.append(loc[:2])

        # add sensor


        self.num_rangefinders = num_rangefinders
        for rangefinder in range(self.num_rangefinders):

            sensor = Rangefinder(self.modules, self.robot.torso_size,
                                 id=rangefinder,
                                 name="rangefinder_" + str(rangefinder),
                                 num_rangefinders=self.num_rangefinders,
                                 modules=self.modules,
                                 wall_locs=self.hazard_locs)
            self.add_module(sensor)




        self.init_sys()



    def reset(self, key):

        new_state = super().reset(key)
        metrics = {
            'reward_forward': 0.0,
            'reward_hazard': 0.0,
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

    def get_hazard_penalty(self, pipeline_state):
        thres = 0.1
        distances = []
        for module in self.modules:
            if module.type == "rangefinder":
                distances.append(module.get_obs(pipeline_state))
        distances =jnp.array(distances)
        penalty = jnp.where(jnp.min(distances) < thres, -1.0, 0.0)
        return penalty


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

        # reposition rangefinders
        q_new =pipeline_state.q
        x_new =pipeline_state.x
        for module in self.modules:
            if module.type == "rangefinder":
                sensor_pos = module.reset(pipeline_state)
                q_new = q_new.at[module.q_idx:module.q_idx+module.info_size].set(sensor_pos)
                new_pos = x_new.pos.at[module.pos_idx].set(sensor_pos)
                x_new = pipeline_state.x.replace(pos=new_pos)
        pipeline_state = pipeline_state.replace(x=x_new, q=q_new)

        # add hazard collision penalty
        hazard_penalty = self.get_hazard_penalty(state.pipeline_state)
        reward = reward_food - hazard_penalty


        state.metrics.update(
            reward_hazard=-hazard_penalty,
            reward_food=reward_food,
        food_position_x=target_pos[0],
        food_position_y=target_pos[1],
        food_position_z=target_pos[2],
        distance_to_target=distance_to_target)

        state = state.replace(obs=obs, reward=reward, pipeline_state=pipeline_state)

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


        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        ## get info from rangefinders
        range_obs = []
        for module in self.modules:
            if module.type == "rangefinder":
                range_obs.append(module.get_obs(pipeline_state))


        return jnp.concatenate([qpos] + [qvel] + range_obs)


