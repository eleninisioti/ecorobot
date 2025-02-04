""" An environment where a single robot needs to maximize its velocity on the x-direction
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
from ecorobot import robots
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import PipelineEnv, State
from ecorobot.modules.wall import Wall
import copy

class LocomotionWithObstacles(EcorobotEnv):
    def __init__(self, robot_type,backend="mjx", only_forward=False, project_dir="temp", **kwargs):
        self.episode_length = 2000
        self.num_tasks = 1
        self.current_task = 0

        if robot_type == "discrete_fish":
            self.reward_for_solved = 350
        elif robot_type == "ant":
            self.reward_for_solved = 4000
        elif robot_type == "halfcheetah":
            self.reward_for_solved = 4000

        self.max_reward = self.reward_for_solved

        super().__init__(project_dir=project_dir,episode_length=self.episode_length, **kwargs)

        robot = robots.get_environment(env_name=robot_type,backend=backend)
        robot = RobotWrapper(env=robot, robot_type=robot_type, only_forward=only_forward)
        self.add_robot(robot)

        self.add_obstacles()

        self.init_sys()

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

    def add_obstacles(self):

        height = 0.5
        wall_width = 0.3
        offset = 2 # distance between obstacles
        init_diff = 0.1 # initial difficulty, the height of the obstacle


        # first the corridor walls
        obstacles = {"0" : {
                    "xml_idx": 0,
                    "width": wall_width,
                    "length": 2,
                    "offset": [0, 0],
            "quat": self.get_wall_quat(90),
            "color": "0 0 0 1",
                    "size": [1, 0.2, 50],
            "loc":[0, 3,0],
                    },

        "1": {
            "xml_idx": 1,
            "width": wall_width,
            "length": 2,
            "offset": [0, 0],
            "quat": self.get_wall_quat(90),
            "color": "0 0 0 1",
            "size": [1, 0.2, 50],
            "loc": [0, -3, 0],
        }}

        # first-level
        wall_length = 3
        start_x = 5
        level = 0
        n_obstacles = 5
        obstacle = {
                    "width": wall_width,
                    "length": wall_length,
                    "offset": [0, 0],
                    "quat": self.get_wall_quat(0),
                    "color": "0 0 1 1",
                    "size": [init_diff, 0.2, wall_length],
                    }
        for i in range(2,n_obstacles+2):
            obstacles[i] = copy.deepcopy(obstacle)
            obstacles[i]["loc"] =  [start_x + i*offset, 0, level]
            obstacles[i]["xml_idx"] = i

        # second-level
        start_x = (n_obstacles-1)*offset + 1
        level = 0
        n_obstacles = 5
        obstacle = {
                    "width": wall_width,
                    "length": wall_length,
                    "offset": [0, 0],
                    "quat": self.get_wall_quat(0),
                    "color": "0 1 0 1",
                    "size": [init_diff*2, 0.2, wall_length],

        }

        for i in range(2+n_obstacles,n_obstacles*2+2):
            obstacles[i] = copy.deepcopy(obstacle)
            obstacles[i]["loc"] =  [start_x + i*offset, 0, level]
            obstacles[i]["xml_idx"] = i



        # third-level
        start_x = (n_obstacles*2 - 1) * offset + 1
        level = 0
        n_obstacles = 5
        obstacle = {
            "width": wall_width,
            "length": wall_length,
            "offset": [0, 0],
            "quat": self.get_wall_quat(0),
            "color": "1 0 0 1",
            "size": [init_diff * 3, 0.2, wall_length],

        }

        for i in range(2 + n_obstacles*2, n_obstacles * 3 + 2):
            obstacles[i] = copy.deepcopy(obstacle)
            obstacles[i]["loc"] = [start_x + i * offset, 0, level]
            obstacles[i]["xml_idx"] = i
        for wall_idx, wall_features in obstacles.items():
            wall = Wall(
                xml_idx=wall_features["xml_idx"],
                name="wall_" + str(wall_features["xml_idx"]),
                offset=wall_features["offset"],
                size=wall_features["size"],
                loc=wall_features["loc"],
                quat=wall_features["quat"],
                color=wall_features["color"]
            )
            self.add_module(wall)

    def get_obs_size(self, task):
        return 26

    def get_action_size(self, task):
        8

    def reset(self, key, env_params=None):

        new_state = super().reset(key)
        metrics = {
            'reward_forward': 0.0,
            'reward_survive': 0.0,
            'reward_ctrl': 0.0,
            'x_position': 0.0,
            'y_position': 0.0,
            'distance_from_origin': 0.0,
            'x_velocity': 0.0,
            'y_velocity': 0.0,
        }
        reward, done, zero = jnp.zeros(3)

        metrics = {
            'reward_forward': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'reward_contact': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'forward_reward': zero,
        }

        obs = self._get_obs(new_state.pipeline_state)

        return new_state.replace(metrics=metrics,
                                 obs=obs,
                                 done=new_state.done.astype(jnp.bool_))
        # return self.robot.env.reset(key)

    def step(self, state, action):
        # return super().step(state, action)
        state = super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)
        # return self.robot.env.step(state, action)

        return state.replace(obs=obs,done=state.done.astype(jnp.bool_))

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        #if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
        #    qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel])
