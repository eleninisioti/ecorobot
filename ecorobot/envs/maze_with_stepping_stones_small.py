""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
import jax.numpy as jnp
from brax.envs.base import State
from ecorobot.modules.food import Food
from ecorobot import robots
from ecorobot.modules.wall import Wall


class SteppingStoneMazeSmall(EcorobotEnv):

    def __init__(self, robot_type, project_dir="temp", n_rangefinder_sensors=5, **kwargs):
        self.episode_length = 1000
        #self.reward_for_solved = 10 + 7 # reached target through all stepping stones (reward only at the end)
        self.reward_for_solved = 13000 # approximate, you took 2/3 of th episode to reach the end

        self.num_tasks = 1
        self.max_reward = self.reward_for_solved

        super().__init__(episode_length=self.episode_length, project_dir=project_dir, **kwargs)
        self.robot_loc = [-3, 1]

        robot = robots.get_environment(env_name=robot_type,
                                       **kwargs)

        self.n_rangefinder_sensors = n_rangefinder_sensors
        robot = RobotWrapper(env=robot,
                             robot_type=robot_type,
                             n_rangefinder_sensors=n_rangefinder_sensors,
                             n_pieslice_sensors=0)

        self.add_robot(robot)

        # add food
        self.maze_length = 4
        self.food_radius = 10
        self.food_loc = [-3, 6]
        self.max_food_distance = self.maze_length
        food = Food(loc_type="fixed",
                    max_distance=self.max_food_distance,
                    radius=10,
                    init_loc=self.food_loc,
                    idx=0,
                    color="0 1 0 0"  # make food visible to pieslice sensors
                    )

        self.target = food
        self.add_module(food)
        self.max_sensor_distance = self.max_food_distance

        self.add_stepping_stones()

        self.build_maze()

        self.init_sys()


    def add_stepping_stones(self):
        self.stepping_stones = []
        self.stone_locs = [[self.robot_loc[0]+2, self.robot_loc[1] + 1.5],
                           [self.maze_length-0.8, self.robot_loc[1]],
                           [self.maze_length - 0.8, self.robot_loc[1]+1.5],
                           [self.food_loc[0]+ 3, self.food_loc[1]-1.5],
                           [self.food_loc[0]+4, self.food_loc[1]],
                           [self.food_loc[0] + 1.0, self.food_loc[1]]

                           ]
        for stone_idx, stone_loc in enumerate(self.stone_locs):
            name = "step_" + str(stone_idx)
            stone = Food(loc_type="fixed",
                         max_distance=self.max_food_distance,
                         init_loc=stone_loc,
                         name=name,
                         color="1 0 0 0",  # make stepping stones invisible
                         idx=stone_idx,
                         size=0.2)
            self.add_module(stone)
            self.stepping_stones.append(stone)
        self.stone_locs.append(self.food_loc)

        # self.max_stone_distance = 3.6055  # find most distant stone
        self.max_stone_distance = jnp.sqrt((self.maze_length*2)**2+ (self.maze_length*2)**2)

    def get_wall_quat(self, angle_with_z):
        L = (jnp.cos(jnp.deg2rad((90) / 2)), 0, jnp.sin(jnp.deg2rad(90 / 2)), 0)  # rotate around y-axis
        R = (jnp.cos(jnp.deg2rad((-90 + angle_with_z) / 2)), jnp.sin(jnp.deg2rad((-90 + angle_with_z) / 2)), 0,
             0)  # rotate around z axis
        # quat = str(quat_y[0] + quat_z[0]) + " 0 " + str(quat_z[1]) + " " + str(quat_y[1])
        # quat = self.quaternion_multiply(quat_y, quat_z)
        quat_0 = float(L[0] * R[0] - L[1] * R[1] - L[2] * R[2] - L[3] * R[3])
        quat_1 = float(L[0] * R[1] + L[1] * R[0] + L[2] * R[3] - L[3] * R[2])
        quat_2 = float(L[0] * R[2] - L[1] * R[3] + L[2] * R[0] + L[3] * R[1])
        quat_3 = float(L[0] * R[3] + L[1] * R[2] - L[2] * R[1] + L[3] * R[0])
        quat = [quat_0, quat_1, quat_2, quat_3]
        return quat

    def build_maze(self):
        # todo: i do not understand why but the locations in the visualization appear at twice the actual location. so scaleing for now
        length = self.maze_length
        height = 0.7
        wall_width = 0.1


        self.wall_specification = {0: {"xml_idx": 0,
                                       "size": [  height,wall_width, self.maze_length],
                                       "loc": [0, 0],
                                       "offset": [0, 0],
                                       "quat": [jnp.cos(jnp.deg2rad((90) / 2)), 0, jnp.sin(jnp.deg2rad(90 / 2)), 0]
                                       }, # southern wall
                                   1: {"xml_idx": 1,
                                       "size": [  height,wall_width, self.maze_length],

                                       "loc": [0, 7],
                                       "offset": [0, 0],
                                       "quat": [jnp.cos(jnp.deg2rad((90) / 2)), 0, jnp.sin(jnp.deg2rad(90 / 2)), 0]
                                       }, # northern wall
                                   2: {"xml_idx": 2,
                                       "size": [height, wall_width, self.maze_length],

                                       "loc": [-self.maze_length, self.maze_length-0.5],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(0)
                                       }, # west wall
                                   3: {"xml_idx": 3,
                                       "size": [  height,wall_width, self.maze_length],

                                       "loc": [self.maze_length , self.maze_length-0.5 ],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(0)
                                       },  # west wall
                                   4: {"xml_idx": 4,
                                       "size": [height, wall_width, 2],

                                       "loc": [self.food_loc[0] + 1, self.food_loc[1] - 1],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(80)
                                       },
                                   5: {"xml_idx": 5,
                                       "size": [  height,wall_width, 1.5],

                                       "loc": [self.food_loc[0] + 1, self.food_loc[1] - 2.5],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(190)
                                       },
                                   6: {"xml_idx": 6,
                                       "size": [  height,wall_width, 2.5],

                                       "loc": [self.food_loc[0] + 3.5, self.food_loc[1] - 2.5],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(120)
                                       },
                                   7: {"xml_idx": 7,
                                       "size": [  height,wall_width, 0.5],

                                       "loc": [0, 0.5],
                                       "offset": [0, 0],
                                       "quat": self.get_wall_quat(60)
                                       },
                                   8: {"xml_idx": 8,
                                       "size": [  height,wall_width, 1.5],

                                       "loc": [self.maze_length-1.5, self.maze_length],
                                       "offset": [0,0],
                                       "quat": self.get_wall_quat(-60)
                                       },
                                   }

        for wall_idx, wall_features in self.wall_specification.items():
            wall = Wall(
                xml_idx=wall_features["xml_idx"],
                name="wall_" + str(wall_features["xml_idx"]),
                offset=wall_features["offset"],
                size=wall_features["size"],

                loc=[wall_features["loc"][0], wall_features["loc"][1],0],
                quat=wall_features["quat"]
            )
            self.add_module(wall)

    def reset(self, key, env_params=None):

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

        obs = self._get_obs(new_state.pipeline_state)

        new_state.info["current_step"] = 0
        new_state.info["key"] = key
        new_state.info["n_stones"] = 0  # how many waypoints the agent has passed

        # move robot to init loc


        new_qpos = new_state.pipeline_state.qpos.at[0:2].set(self.robot_loc)
        new_pipeline = new_state.pipeline_state.replace(qpos=new_qpos)
        return new_state.replace(metrics=metrics, obs=obs, info=new_state.info, done=new_state.done.astype(jnp.bool_), pipeline_state=new_pipeline)

    def step_debug(self, state, action, debug_robot_pos):
        state = super().step(state, action)
        robot_pos = debug_robot_pos[state.info["current_step"]]

        new_x = state.pipeline_state.x
        new_x = new_x.replace(pos=new_x.pos.at[0,:2].set(robot_pos))
        state = state.replace(pipeline_state=state.pipeline_state.replace(x=new_x))

        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)

        # calculate target reward



        target_pos = jnp.array(self.food_loc)

        # detect if robot reached next stepping stone

        current_stone = state.info["n_stones"]
        stone_pos = jnp.array(self.stone_locs)[current_stone]
        distance_stone = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(stone_pos)[:2]) ** 2))
        stone_pos_next = jnp.array(self.stone_locs)[current_stone + 1]
        current_stone = jnp.where(distance_stone < (0.5), jnp.minimum(current_stone + 1, len(self.stone_locs)),
                                  current_stone)
        state.info["n_stones"] = current_stone
        distance_stone_next = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(stone_pos_next)[:2]) ** 2))
        reward_food = state.info["n_stones"] + (1 - distance_stone_next / self.max_stone_distance)

        # WARNING: this is for ppo only
        #reward_food = jnp.where(jnp.equal(state.info["current_step"], self.episode_length - 1), reward_food, 0.0)

        distance_target = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(self.food_loc)) ** 2))
        at_target = jnp.less(distance_target, (self.target.size + 0.5))
        reward_food = jnp.where(jnp.logical_and(at_target, state.info["n_stones"] == len(self.stone_locs)), 10.0 + reward_food,
                                reward_food)
        done = jnp.where(at_target, 1.0, state.done)
        done = done.astype(jnp.bool_)
        state.metrics.update(
            reward_food=reward_food,
            distance_to_target=0.0
            #
        )

        state = state.replace(obs=obs, done=done, reward=reward_food, pipeline_state=pipeline_state, info=state.info)

        # state = state.replace(obs=obs, pipeline_state=pipeline_state)
        return state
    def step(self, state, action):
        state = super().step(state, action)

        pipeline_state = state.pipeline_state

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
        current_stone = jnp.where(distance_stone < (0.1), jnp.minimum(current_stone + 1, len(self.stone_locs)), current_stone)
        state.info["n_stones"] = current_stone
        distance_stone_next = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(stone_pos_next)[:2]) ** 2))
        reward_food = state.info["n_stones"] + (1 - distance_stone_next / self.max_stone_distance)
        # WARNING: this is for ppo only
        reward_food = jnp.where(jnp.equal(state.info["current_step"], self.episode_length-1), reward_food, 0.0)

        distance_target = jnp.sqrt(jnp.sum((robot_pos[:2] - jnp.array(self.food_loc)) ** 2))
        at_target = jnp.less(distance_target, (self.target.size+0.5))
        reward_food = jnp.where(jnp.logical_and(at_target,  state.info["n_stones"] == len(self.stone_locs )) , 10000.0+reward_food, reward_food)
        done = jnp.where(at_target, 1.0, state.done)
        done = done.astype(jnp.bool_)
        state.metrics.update(
            reward_food=reward_food,
            distance_to_target=0.0
            #
        )

        state = state.replace(obs=obs, done=done, reward=reward_food, pipeline_state=pipeline_state, info=state.info)

        # state = state.replace(obs=obs, pipeline_state=pipeline_state)
        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q[self.robot.idx:self.robot.info_size]
        qvel = pipeline_state.qd[self.robot.idx:self.robot.info_size][:2]

        # if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
        #    qpos = qpos[self.robot.info_size:]

        # rangefinder sensor shows the distance to the nearest item. if nothing is detected we give 1
        sensor_data = pipeline_state.sensordata
        rangefinder_data = sensor_data[:self.n_rangefinder_sensors]
        rangefinder_sensor_info = jnp.where(rangefinder_data == -1, 1.0, rangefinder_data)

        food_dir = qpos[:2] - jnp.array(self.food_loc)
        stepping_stone_locs = jnp.ravel(jnp.array(self.stone_locs))
        return jnp.concatenate([qpos] + [qvel] + [rangefinder_sensor_info] )
