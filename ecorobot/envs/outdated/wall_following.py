""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot.robots.base import RobotWrapper
import gym
from brax import envs
import math
import jax.numpy as jnp
import math as omath

from brax import math
from brax.envs.base import State
from ecorobot.ecorobot.modules.food import Food
from ecorobot.ecorobot import robots
import brax
from ecorobot.ecorobot.sensors import Rangefinder
from ecorobot.ecorobot.modules.wall import Wall
from jax.experimental import host_callback as hcb
from xml.etree import ElementTree
import jax
import numpy as onp
import random



class WallFollowing(EcorobotEnv):

    def __init__(self, robot_type, num_rangefinders=6, episode_length=1000, **kwargs):
        super().__init__(episode_length=episode_length,**kwargs)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add food
        self.food_radius = 10
        self.target_loc = [8.0,4]
        #food = Food(loc_type="radial", loc=self.target_loc, radius=self.food_radius)
        #self.target = food
        #self.add_module(food)

        self.build_wall()

        self.num_rangefinders = num_rangefinders
        for rangefinder in range(self.num_rangefinders):
            sensor = Rangefinder(self.modules, self.robot.torso_size,
                                 id=rangefinder,
                                 name="rangefinder_"+str(rangefinder),
                                 num_rangefinders=self.num_rangefinders,
                                 modules=self.modules)
            self.add_module(sensor)


        self.init_sys()


    def build_wall(self):
        # we just create the walls without caring about their location


        robot_loc = jnp.array([0, 0])
        target_loc = jnp.array([8.66,5]) # dummy value
        turn_point = jnp.array([robot_loc[0], target_loc[1]-robot_loc[1]])
        self.height = 3
        self.corridor_width = 0.5
        self.first_corridor_length = jnp.abs(turn_point[1]).astype(float)
        self.second_corridor_length = jnp.abs(target_loc[0]).astype(float)

        self.wall_locs = [[turn_point[1].astype(float)/2, -self.corridor_width],
                          [turn_point[1].astype(float) / 2, self.corridor_width],
                          [turn_point[1].astype(float) - 0.5, -self.corridor_width],
                          [turn_point[1].astype(float),
                           self.corridor_width + 0.6],
                          [turn_point[1].astype(float) - 1.2,
                           self.corridor_width + 1.1]
                          ]



        random_angle =0

        self.wall_specification = {0: {"xml_idx": 0,
                                        "width": 0.2,
                                       "length": self.first_corridor_length,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float)/2, -self.corridor_width],
                                       "offset": [0, 0],
                                       "euler": "90 90 " + str(random_angle),
                                       "quart": ""

                                       },
                                   1: {"xml_idx": 1,
                                       "width": 0.2,
                                       "length": self.first_corridor_length,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float) / 2, self.corridor_width],
                                       "offset": [0, 0],
                                       "euler": "90 90 " + str(random_angle),
                                       "quart": ""

                                       },
                                   2: {"xml_idx": 2,
                                       "width": 0.2,
                                       "length": 1.7,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float)-0.5, -self.corridor_width],
                                       "offset": [-0.5, 0],

                                       "euler": "90 90 " + str(random_angle),
                                       "quart": ""

                                       },
                                   3: {"xml_idx": 3,
                                       "width": 0.2,
                                       "length": self.second_corridor_length*0.7,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float),
                                               self.corridor_width+0.6 ],
                                       "offset": [0, 0.6],

                                       "euler": "90 0 " + str(random_angle),
                                       "quart": ""

                                       },
                                   4: {"xml_idx": 4,
                                       "width": 0.2,
                                       "length": self.second_corridor_length*0.7-2,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float)-1.2,
                                               self.corridor_width+1.1],
                                       "offset": [-1.2, 1.1],

                                       "euler": "90 0 " + str(random_angle),
                                       "quart": ""

                                       },
                                   5: {"xml_idx": 5,
                                       "width": 0.2,
                                       "length": 1.7,
                                       "height": self.height,
                                       "loc": [turn_point[1].astype(float)-0.5, 2.8],

                                       "offset": [-1.2, 1.1],

                                       "euler": "90 90 " + str(random_angle),
                                       "quart": ""

                                       },

        }



        for wall_idx, wall_features in self.wall_specification.items():
            wall = Wall(
                xml_idx=wall_features["xml_idx"],
                name="wall_" + str(wall_features["xml_idx"]),
                offset=wall_features["offset"],
                        width=wall_features["width"],
                        length=wall_features["length"],
                        height=wall_features["height"],
                        loc=[wall_features["loc"][0], wall_features["loc"][1]],
                        euler= wall_features["euler"],
                        has_pos=True
            )
            self.add_module(wall)
            if wall_features["xml_idx"]== len(self.wall_locs)-1:
                self.target = wall
                self.target.max_distance = jnp.sqrt(jnp.sum((wall_features["loc"][0]) ** 2+ (wall_features["loc"][1]) ** 2))



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

        # set wall positions based on food
        # move the walls by a constant offset
        pipeline_state = new_state.pipeline_state


        offset_x = jax.random.uniform(key,minval=0.2,maxval=2)
        offset_y = jax.random.uniform(key,minval=0.2,maxval=2)

        for module in self.modules:
            if module.type == "wall":

                #new_wall_pos = pipeline_state.x.pos[module.pos_idx]

                new_wall_pos = pipeline_state.q[module.q_idx: module.q_idx+module.info_size]

                temp =  jnp.array([offset_x, offset_y, 0])
                new_wall_pos = new_wall_pos + temp

                pos_new = pipeline_state.x.pos.at[module.pos_idx].set(new_wall_pos)
                #x_new = pipeline_state.x.replace(pos=pos_new)
                q_new = pipeline_state.q.at[module.q_idx:module.q_idx+module.info_size].set(new_wall_pos)

                pipeline_state = pipeline_state.replace(q=q_new)

        """

        target_pos = new_state.pipeline_state.q[self.target.q_idx:self.target.q_idx+self.target.info_size]
        new_q = new_state.pipeline_state.q
        for module in self.modules:
            if module.type == "wall":

                wall_pos = module.reset_wall(target_pos, new_state)
                new_q = new_q.at[module.q_idx:module.q_idx+module.info_size].set(wall_pos)
        """



        #pipeline_state = self.pipeline_init(new_q, new_state.pipeline_state.qd)
        obs = self._get_obs(pipeline_state)

        return new_state.replace(metrics=metrics, obs=obs, pipeline_state=pipeline_state)

    def step(self, state, action):
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)

        # calculate food reward
        robot_pos = pipeline_state.x.pos[0]
        target_pos = pipeline_state.x.pos[self.target.pos_idx]


        distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
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


        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        ## get info from rangefinders
        range_obs = []
        for module in self.modules:
            if module.type == "rangefinder":
                range_obs.append(module.get_obs(pipeline_state))

        return jnp.concatenate([qpos] + [qvel] + range_obs)


