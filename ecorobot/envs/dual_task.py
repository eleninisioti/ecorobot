
from ecorobot.envs.base import EcorobotEnv
from ecorobot.envs.foraging_with_sensors import ForagingWithSensors
from ecorobot.envs.wall_following import WallFollowing
import jax
import jax.numpy as jnp
class DualTask(EcorobotEnv):

    def __init__(self, robot_type, project_dir, n_rangefinder_sensors=5, n_pieslice_sensors=4, **kwargs):

        self.episode_length = 2000

        super().__init__(episode_length=self.episode_length, project_dir=project_dir, **kwargs)
        foraging_env = ForagingWithSensors(n_pieslice_sensors=n_pieslice_sensors, robot_type=robot_type, project_dir=project_dir,n_rangefinder_sensors=n_rangefinder_sensors)
        wallfollowing_env = ForagingWithSensors(n_pieslice_sensors=n_pieslice_sensors, robot_type=robot_type, project_dir=project_dir,n_rangefinder_sensors=n_rangefinder_sensors)
        self.envs_reset = [foraging_env.reset, wallfollowing_env.reset]
        self.envs_step = [foraging_env.step, wallfollowing_env.step]

    def reset(self, key):

        current_env = 0
        state = jax.lax.switch(current_env, self.envs_reset, key)
        state.info["current_env"] = current_env
        state.info["current_step"] = 0

        state.info["key"] = key
        return state

    def step(self, state, action):
        state = jax.lax.switch(state.info["current_env"], self.envs_step, state, action)
        temp_state = self.envs_reset[1](state.info["key"])
        state = jax.lax.cond(state.info["current_step"] == (self.episode_length/2), lambda x: self.envs_reset[1](x), lambda x: state, state.info["key"])
        state.info["current_step"] = state.info["current_step"]+1

        state.info["current_env"] = jnp.where(state.info["steps"] == (self.episode_length/2), 1,state.info["steps"] )
        return state




