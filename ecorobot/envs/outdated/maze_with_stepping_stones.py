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
from ecorobot.ecorobot.sensors import Rangefinder
from ecorobot.ecorobot.modules.wall import Wall
import numpy as onp
import math as omath


class SteppingStoneMaze(EcorobotEnv):

    def __init__(self, project_dir, robot_type, episode_length, num_rangefinders=4, num_compasses=2, **kwargs):
        super().__init__(episode_length=episode_length, project_dir=project_dir)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.robot_loc = [-2, 0]
        self.add_robot(robot, self.robot_loc)

        # add food
        self.maze_length = 10

        self.food_radius = 10
        self.food_loc = [-self.maze_length / 2 + 1, self.maze_length - 1.5]
        food = Food(loc=self.food_loc, add_joints=False, loc_type="fixed")

        self.target = food
        self.add_module(food)

        self.add_stepping_stones()

        self.build_maze()

        self.num_rangefinders = num_rangefinders
        for rangefinder in range(self.num_rangefinders):
            sensor = Rangefinder(self.modules, self.robot.torso_size,
                                 id=rangefinder,
                                 name="rangefinder_" + str(rangefinder),
                                 num_rangefinders=self.num_rangefinders,
                                 modules=self.modules,
                                 wall_locs=self.wall_locs)
            self.add_module(sensor)

        self.num_compasses = num_compasses
        for compass in range(self.num_compasses):
            compass = Compass(food, self.robot.torso_size,
                              idx=compass,
                              name="compass_" + str(compass),
                              num_compasses=num_compasses,
                              target_loc=self.food_loc)
            self.add_module(compass)
        self.episode_length = episode_length

        self.init_sys()

    def add_stepping_stones(self):
        self.stepping_stones = []
        self.stone_locs = [[self.robot_loc[0], self.robot_loc[1]+1.5],
                           [self.maze_length/4, self.robot_loc[1]+1.5],
                           [self.maze_length/4, self.robot_loc[1]+4.5],
                           [self.maze_length/4-1.5, self.robot_loc[1]+4.5],
                           [self.maze_length/4-1.5, self.food_loc[1]]]
        #self.stone_locs = [[0, 1.5], [2.2, 0.5], [2.2, 1.8], [-0.2, 3], [0.5, 3.7], [1.2, 4.8], [-1, 5]]
        for stone_idx, stone_loc in enumerate(self.stone_locs):
            name = "step_" + str(stone_idx)
            stone = Food(loc=stone_loc, name=name, idx=stone_idx, size=0.2, add_joints=False)
            self.add_module(stone)
            self.stepping_stones.append(stone)

        self.max_stone_distance = 3.6055  # find most distant stone

    def build_maze(self):
        # todo: i do not understand why but the locations in the visualization appear at twice the actual location. so scaleing for now
        scale_debug = 0.5

        length = 2
        height = 1
        self.wall_locs = [[0 * scale_debug, 0 * scale_debug],
                          [2 * scale_debug, (1.8 + length) * scale_debug],
                          [-3 * scale_debug, (1 + length) * scale_debug],
                          [(-2.5 + length) * scale_debug, (3 + length) * scale_debug],
                          [(-1) * scale_debug, (2 + length) * scale_debug],
                          [(0.5) * scale_debug, (1 + length) * scale_debug],
                          [0 * scale_debug, -0.5 * scale_debug],
                          [0 * scale_debug, 5.5 * scale_debug],
                          [-3.5 * scale_debug, 2.5 * scale_debug],
                          [3.5 * scale_debug, 2.5 * scale_debug]]

        self.wall_locs = [[self.robot_loc[0] + 2, self.robot_loc[1] ],
                          [self.maze_length / 2-length*1.5/2, self.maze_length * 0.5],
                          [- self.maze_length / 2+length, self.maze_length * 0.7],
                          [- self.maze_length / 2+length*2, self.maze_length * 0.5],
                          [- self.maze_length / 2+length+1, self.maze_length * 0.4],
                          [0, -0.5],  # wall behind the agent
                          [0, self.maze_length - 0.5],
                          [- self.maze_length / 2, self.maze_length / 2 - 0.5],
                          [self.maze_length / 2, self.maze_length / 2 - 0.5]]

        """
                    0: {"xml_idx": 0,
                "width": 0.2,
                "length": length * 0.8,
                "height": height,
                "loc": self.wall_locs[0],
                "offset": [0, 0],
                "quart": (0.6532814824381883, 0.6532814824381882, -0.27059805007309845, -0.2705980500730985),
                },

            1: {"xml_idx": 1,
                "width": 0.2,
                "length": length * 2.5,
                "height": height,
                "loc": self.wall_locs[1],
                "offset": [0, 0],
                "quart": (0.5963678105290182, 0.5963678105290181, 0.37992819659091526, 0.3799281965909153)},

            2: {"xml_idx": 2,
                "width": 0.2,
                "length": length,
                "height": height,
                "quart": (0.6532814824381883, 0.6532814824381882, 0.27059805007309845, 0.2705980500730985),
                "loc": self.wall_locs[2],
                "offset": [0, 0]},

            3: {"xml_idx": 3,
                "width": 0.2,
                "length": length * 3.5,
                "height": height,
                "quart": (0.560985526796931, 0.5609855267969309, -0.4304593345768794, -0.43045933457687946),
                "loc": self.wall_locs[3],
                "offset": [0, 0]},
                
             4: {"xml_idx": 4,
                "width": 0.2,
                "length": length * 2,
                "height": height,
                "quart": (0.7010573846499779, 0.7010573846499778, 0.09229595564125724, 0.09229595564125725),
                "loc": self.wall_locs[4],
                "offset": [0, 0]},
        """
        self.wall_specification = {

            0: {"xml_idx": 0,
                "width": 0.2,
                "length": length * 0.8,
                "height": height,
                "loc": self.wall_locs[0],
                "offset": [0, 0],
                "quart": (0.6532814824381883, 0.6532814824381882, -0.27059805007309845, -0.2705980500730985),
                },

            1: {"xml_idx": 1,
                "width": 0.2,
                "length": length * 1.5,
                "height": height,
                "loc": self.wall_locs[1],
                "offset": [0, 0],
                "quart": (0.5963678105290182, 0.5963678105290181, 0.37992819659091526, 0.3799281965909153)},

            2: {"xml_idx": 2,
                "width": 0.2,
                "length": length * 2,
                "height": height,
                "quart": (0.560985526796931, 0.5609855267969309, -0.4304593345768794, -0.43045933457687946),
                "loc": self.wall_locs[2],
                "offset": [0, 0]},

            3: {"xml_idx": 3,
                "width": 0.2,
                "length": length*3,
                "height": height,
                "quart": (0.6532814824381883, 0.6532814824381882, 0.27059805007309845, 0.2705980500730985),
                "loc": self.wall_locs[3],
                "offset": [0, 0]},


            5: {"xml_idx": 5,
                "width": 0.2,
                "length": self.maze_length,
                "height": height,
                "quart": (0.5000000000000001, 0.5, 0.4999999999999999, 0.5),
                "loc": self.wall_locs[5],
                "offset": [0, 0]},
            6: {"xml_idx": 6,
                "width": 0.2,
                "length": self.maze_length,
                "height": height,
                "quart": (0.5000000000000001, 0.5, 0.4999999999999999, 0.5),
                "loc": self.wall_locs[6],
                "offset": [0, 0]},
            7: {"xml_idx": 7,
                "width": 0.2,
                "length": self.maze_length,
                "height": height,
                "loc": self.wall_locs[7],
                "offset": [0, 0],
                "quart": (0.5000000000000001, -0.4999999999999999, 0.5, 0.5)},
            8: {"xml_idx": 8,
                "width": 0.2,
                "length": self.maze_length,
                "height": height,
                "loc": self.wall_locs[8],
                "offset": [0, 0],
                "quart": (0.5000000000000001, -0.4999999999999999, 0.5, 0.5),
                },


        }

        for wall_idx, wall_features in self.wall_specification.items():
            print(wall_idx)
            wall = Wall(
                xml_idx=wall_features["xml_idx"],
                name="wall_" + str(wall_features["xml_idx"]),
                offset=wall_features["offset"],
                width=wall_features["width"],
                length=wall_features["length"],
                height=wall_features["height"],
                loc=wall_features["loc"],
                quart=wall_features["quart"],
                has_pos=False)
            self.add_module(wall)

    def reset(self, key):

        new_state = super().reset(key)
        metrics = {
            'reward_forward': 0.0,
            'reward_food': 0.0,
            'reward_survive': 0.0,
            'reward_ctrl': 0.0,
            'x_position': 0.0,
            'y_position': 0.0,
            'distance_from_origin': 0.0,
            'distance_to_target': 0.0,
            'x_velocity': 0.0,
            'y_velocity': 0.0,
        }

        # reposition rangefinders
        pipeline_state = new_state.pipeline_state
        q_new = pipeline_state.q
        x_new = pipeline_state.x

        for module in self.modules:
            if module.type == "rangefinder" or module.type == "compass":
                sensor_pos = module.reset(pipeline_state)
                q_new = q_new.at[module.q_idx:module.q_idx + module.info_size].set(sensor_pos)
                new_pos = x_new.pos.at[module.pos_idx].set(sensor_pos)
                x_new = x_new.replace(pos=new_pos)

        pipeline_state = pipeline_state.replace(x=x_new, q=q_new)

        obs = self._get_obs(pipeline_state)

        new_state.info["current_step"] = 0
        new_state.info["key"] = key
        new_state.info["n_stones"] = 0  # how many waypoints the agent has passed

        return new_state.replace(metrics=metrics, obs=obs, info=new_state.info, pipeline_state=pipeline_state)


    def step(self, state, action):
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state

        # reposition rangefinders
        q_new = pipeline_state.q
        x_new = pipeline_state.x


        for module in self.modules:
            if module.type == "rangefinder" or module.type == "compass":
                sensor_pos = module.reset(pipeline_state)
                q_new = q_new.at[module.q_idx:module.q_idx + module.info_size].set(sensor_pos)
                new_pos = x_new.pos.at[module.pos_idx].set(sensor_pos)
                x_new = x_new.replace(pos=new_pos)

        pipeline_state = pipeline_state.replace(x=x_new, q=q_new)

        obs = self._get_obs(pipeline_state)

        # calculate target reward

        robot_pos = state.pipeline_state.x.pos[0]
        target_pos = jnp.array(self.food_loc)
        distance_to_target = jnp.sqrt(jnp.sum((robot_pos[:2] - target_pos) ** 2))

        # detect if robot reached next stepping stone
        current_stone = state.info["n_stones"]
        stone_pos = jnp.array(self.stone_locs)[current_stone]
        distance_stone = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(stone_pos)[:2]) ** 2))

        stone_pos_next = jnp.array(self.stone_locs)[current_stone + 1]

        current_stone = jnp.where(distance_stone < (1.0), jnp.minimum(current_stone + 1, len(self.stone_locs)), current_stone)
        state.info["n_stones"] = current_stone
        distance_stone_next = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(stone_pos_next)[:2]) ** 2))
        reward_food = state.info["n_stones"] + (1 - distance_stone_next / self.max_stone_distance)

        print(reward_food, distance_stone, distance_stone_next, state.info["n_stones"])

        distance_target = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(self.food_loc)) ** 2))
        at_target = jnp.less(distance_target, 0.1)
        reward_food = jnp.where(at_target, 10.0, reward_food)

        reward_food = jnp.where(jnp.equal(state.info["current_step"], self.episode_length - 1), reward_food, 0.0)

        state.metrics.update(
            reward_food=reward_food,
            distance_to_target=0.0
            #
        )

        state = state.replace(obs=obs, reward=reward_food, pipeline_state=pipeline_state, info=state.info)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        ## get info from rangefinders
        range_obs = []
        compass_obs = []
        for module in self.modules:
            if module.type == "rangefinder":
                range_obs.append(module.get_obs(pipeline_state))

            elif module.type == "compass":
                compass_obs.append(jnp.expand_dims(module.get_distance(pipeline_state), axis=0))

        # we need to remove the location of the food from the qpos and qvel

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]
        modules_info = onp.sum([el.info_size for el in self.modules if el.has_pos])
        qpos = qpos[:-modules_info]
        qvel = qvel[:-modules_info]

        return jnp.concatenate([qpos] + [qvel] + range_obs + compass_obs)
