from brax.envs.ant import Ant
from brax import math
from brax.envs.base import State
import jax.numpy as jp

class ModdedAnt(Ant):
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

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(
            pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
        )
        if self._terminate_when_unhealthy:
          healthy_reward = self._healthy_reward
        else:
          healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        reward = forward_reward # + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

