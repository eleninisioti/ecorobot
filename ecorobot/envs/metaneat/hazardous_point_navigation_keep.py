""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
from ecorobot.sensors import Rangefinder
from ecorobot.modules.hazard import Hazard

import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State
from ecorobot.modules.metaneat.food import Food
from ecorobot import robots
import brax
import jax.numpy as jp
import numpy as onp
import random
import jax

class HazardousPointNavigation(EcorobotEnv):

    def __init__(self,  direction, genidx=0, robot_type="discrete_fish", project_dir="projects/metaneat_benchmarking/metaneat_djo/hazardous",num_rangefinders=12,episode_length=0, reset_task=0,rollout_length=0, **kwargs):
        super().__init__(episode_length=episode_length, project_dir=project_dir, **kwargs)
        #kwargs["backend"] = "spring"
        #direction=direction-1
        robot_type = "discrete_fish"

        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add hazards
        self.build_terrain()

        # add food
        food = Food(loc_type="random_hazards",
                    add_joints=True,
                    max_distance=3,
                    radius=3,
                    hazard_locs=self.hazard_locs,
                    direction=direction,
                    num_hazards=self.num_hazards
                    )
        self.target = food
        self.target_loc = jnp.array(food.loc)
        self.add_module(food)

        # add sensors
        self.sensors = []


        self.num_rangefinders = num_rangefinders
        hazards = [el for el in self.modules if el.type == "hazard"]
        self.sensors = []

        for rangefinder in range(self.num_rangefinders):

            sensor = Rangefinder(self.modules, self.robot.torso_size,
                                 id=rangefinder,
                                 name="rangefinder_" + str(rangefinder),
                                 num_rangefinders=self.num_rangefinders,
                                 modules=self.modules,
                                 max_angle=60,
                                 max_distance=3,
                                 walls=hazards)

            self.sensors.append(sensor)
            #self.add_module(sensor)



        self.init_sys()


    def build_terrain(self):
        self.num_hazards = 25

        distance = 3


        offset = 1
        width = int(self.num_hazards/10)
        self.hazard_locs = []
        for x_loc in range(-width*offset,(width+1)*offset, offset):
            for y_loc in range(-width*offset,(width+1)*offset, offset):
                loc = [x_loc, y_loc]
                self.hazard_locs.append([x_loc, y_loc])

        del self.hazard_locs[12]

        self.num_hazards = len(self.hazard_locs)




        # place hazards

        for hazard in range(self.num_hazards):
            # x = hazard/(self.num_hazards/4)*2 + offset -2
            # y = hazard%(self.num_hazards/4)*2 + offset

            loc = [self.hazard_locs[hazard][0], self.hazard_locs[hazard][1], 0.1]
            hazard = Hazard(xml_idx=hazard,
                            loc=loc,
                            name="hazard_" + str(hazard)
                            )
            self.add_module(hazard)
            #self.hazard_locs.append(loc[:2])


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
        sensor_locs = []
        for sensor in self.sensors:
            sensor_locs.append(sensor.reset(new_state.pipeline_state))

        new_state.info["sensor_locs"] = sensor_locs


        obs = self._get_obs(new_state)

        return new_state.replace(metrics=metrics, obs=obs, info=new_state.info)

    def get_hazard_penalty(self, state):
        thres = 0.25 + 0.1
        distances = []
        for sensor_idx, sensor_loc in enumerate(state.info["sensor_locs"]):
            distances.append(self.sensors[sensor_idx].get_obs(state, sensor_loc))
        distances = jnp.array(distances)
        penalty = jnp.where(jnp.min(distances) < thres, 1.0, 0.0)
        penalty = 0.0
        return penalty


    def step(self, state, action):
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(state)

        # calculate food reward

        robot_pos = pipeline_state.x.pos[0]
        #target_pos = pipeline_state.q[-self.target.info_size:]

        #target_pos = pipeline_state.x.pos[self.target.pos_idx]
        #target_pos = self.target_loc
        #target_pos = self.target.location
        target_pos = pipeline_state.x.pos[self.target.pos_idx]


        distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
        #distance_to_target = 10.0
        reward_food = 1 - (distance_to_target / self.target.max_distance)

        # reposition rangefinders
        """
        q_new =pipeline_state.q
        x_new =pipeline_state.x
        for module in self.modules:
            if module.type == "rangefinder":
                sensor_pos = module.reset(pipeline_state)
                q_new = q_new.at[module.q_idx:module.q_idx+module.info_size].set(sensor_pos)
                new_pos = x_new.pos.at[module.pos_idx].set(sensor_pos)
                x_new = pipeline_state.x.replace(pos=new_pos)
        pipeline_state = pipeline_state.replace(x=x_new, q=q_new)
        """

        sensor_locs = []
        for sensor in self.sensors:
            sensor_locs.append(sensor.reset(pipeline_state))

        state.info["sensor_locs"] = sensor_locs


        # add hazard collision penalty
        hazard_penalty = self.get_hazard_penalty(state)
        #hazard_penalty=0.0
        #hazard_penalty = 0.0
        #hazard_penalty =0.0
        reward = reward_food - hazard_penalty

        #done =jnp.where(distance_to_target < 0.2, 1.0, state.done)
        done = 0.0

        state.metrics.update(
            reward_hazard=-hazard_penalty,
            reward_food=reward_food,
        food_position_x=target_pos[0],
        food_position_y=target_pos[1],
        food_position_z=target_pos[2],
        distance_to_target=distance_to_target)

        state = state.replace(obs=obs, reward=reward, pipeline_state=pipeline_state, info=state.info, done=0.0)

        return state

    def _get_obs(self, state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = state.pipeline_state.q
        qvel = state.pipeline_state.qd

        # we need to remove the location of the food from the qpos and qvel
        #modules_info = onp.sum([el.info_size for el in self.modules])
        #qpos = qpos[:-modules_info]
        #qvel = qvel[:-modules_info]


        #if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
        #    qpos = qpos[2:]

        ## get info from rangefinders
        qpos = qpos[:-self.target.info_size]
        qvel = qvel[:-self.target.info_size]


        ## get info from sensors
        #sensor_info = self.target_compass.get_obs(pipeline_state)

        #sensor_pos = self.target_compass.reset(pipeline_state)

        #if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
        #    qpos = qpos[2:]

        #to_food = state.pipeline_state.x.pos[0]-target_pos

        range_obs = []



        for sensor_idx, sensor_loc in enumerate(state.info["sensor_locs"]):
            range_obs.append(self.sensors[sensor_idx].get_obs(state, sensor_loc))


        #target_pos = self.target_loc[:2]
        target_pos = state.pipeline_state.q[-self.target.info_size:]

        to_food = state.pipeline_state.x.pos[0]-target_pos


        return jnp.concatenate( [qpos] + [qvel] )
        #return jnp.concatenate( [qvel] + [to_food])


