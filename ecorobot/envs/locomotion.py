""" An environment where a single robot needs to maximize its velocity on the x-direction
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
from ecorobot import robots
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import PipelineEnv, State


class Locomotion(EcorobotEnv):
    def __init__(self,  robot_type, backend="mjx", only_forward=False,project_dir="temp", **kwargs):
        self.episode_length = 300
        self.num_tasks = 1
        self.current_task = 0

        if robot_type == "discrete_fish":
            self.reward_for_solved = 350
        elif robot_type == "ant":
            self.reward_for_solved = 4000
        else :
            self.reward_for_solved = 4000

        self.max_reward = self.reward_for_solved
        super().__init__(project_dir=project_dir, backend=backend, episode_length=self.episode_length, **kwargs)

        robot = robots.get_environment(env_name=robot_type,backend=backend)
        robot = RobotWrapper(env=robot, robot_type=robot_type, only_forward=only_forward)
        self.add_robot(robot)

        self.init_sys()


    def get_obs_size(self, task):
        return 26

    def get_action_size(self, task):
        8
    def reset(self, key, env_params=None):

        new_state = super().reset(key)

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

        return new_state.replace(metrics=metrics, obs=obs,done=new_state.done.astype(jnp.bool_))
        #return self.robot.env.reset(key)

    def step(self, state, action):
        #return super().step(state, action)
        state= super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)
        return self.robot.env.step(state, action)

        #return state.replace(obs=obs, done=state.done.astype(jnp.bool_))

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel])

