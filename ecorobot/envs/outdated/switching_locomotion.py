""" An environment where a single robot needs to maximize its velocity on the x-direction
"""

from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot.robots.base import RobotWrapper
from ecorobot.ecorobot import robots
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import PipelineEnv, State


class SwitchingLocomotion(EcorobotEnv):
    def __init__(self, robot_type,project_dir, direction=1, **kwargs):
        super().__init__(project_dir=project_dir)

        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        self.direction = direction

        self.init_sys()

    def reset(self, key):

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

        obs = self._get_obs(new_state.pipeline_state)

        return new_state.replace(metrics=metrics, obs=obs)

    def step(self, state, action):
        state, reward = super().step(state, action)
        reward = reward*self.direction
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)


        state = state.replace(obs=obs, reward =reward
        )

        #state = self.robot.env.step(state, action)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel])

