from brax.envs.half_cheetah import Halfcheetah
from brax import math
from brax.envs.base import State
import jax.numpy as jp

class ModdedHalfCheetah(Halfcheetah):
    def __init__(self, reset_noise_scale=0.1, direction=1, genidx=0, **kwargs):

        super().__init__(**kwargs)
        self.direction = int(direction)
        self._reset_noise_scale = float(reset_noise_scale)
        assert self.direction in (1, -1), "Invalid Ant direction"

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = self.direction * velocity[0]


        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state)
        reward = forward_reward # + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost,
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=velocity[0],
            y_position=pipeline_state.x.pos[0, 1],

        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward
        )

