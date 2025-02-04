# from rllab.spaces import Box
import collections
from gym.spaces import Box
from brax.envs.base import Env, State
from brax import math, base
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

def logical_or(x, y):
    return jp.logical_not(jp.logical_and(jp.logical_not(x), jp.logical_not(y)))

def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)


def sample_goal_from_circle(seed: int):
    key = seed
    key1, key2 = jax.random.split(key)
    length = 1
    angle = jp.pi * jax.random.uniform(key=key2, minval=0.0, maxval=2.0)
    #angle = jp.deg2rad([45, 135,225, 315][seed-1])
    #angle = jp.pi * angle
    x = length * jp.cos(angle)
    y = length * jp.sin(angle)
    return jp.array([x, y])


def sample_goal_from_two(goalidx: int):
    id = goalidx % 2
    sample = [jp.array((jp.cos(0.785398), jp.sin(0.785398))), jp.array((jp.cos(3.92699), jp.sin(3.92699)))]
    return sample[id]


def sample_goals(num_goals, key: jax.random.PRNGKey):
    return jax.random.uniform(-0.5, 0.5, size=(num_goals, 2,), key=key)


class PointEnvRandGoal(Env):
    def __init__(self, goalidx, genidx, predefined_goals=False, backend=None, **kwargs):  # Can set goal to test adaptation.
        super().__init__(**kwargs)
        seed = (int(genidx) << 16) + int(goalidx)
        #seed = int(goalidx)
        key = jax.random.PRNGKey(seed=seed)
        self._goal = sample_goal_from_two(goalidx=goalidx) if predefined_goals else sample_goal_from_circle(seed=key)
        self.goal_idx = goalidx
        self._action_space = Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def observation_space(self):
        return Box(low=-jp.inf, high=jp.inf, shape=(2,))

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_size(self):
        return 2
    
    @property
    def action_size(self):
        return 2
    
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return "generalized"


    def render(self):

        fig, ax = plt.subplots()
        # plot food
        ax.scatter(x=self._goal[0], y=self._goal[1], color="green")



        # plot agent
        ax.scatter(x=0, y=0, color="orange")


        plt.savefig("2d_" + str(self.goal_idx) + ".png")
        plt.clf()


    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'forward_reward': zero,
        }
        pipeline_state = base.State(
            q=None, #jp.zeros(1),
            qd=None, #jp.zeros(1),
            x=base.Transform.create(pos=jp.zeros(2)),
            xd=base.Motion.create(vel=jp.zeros(1)),
            contact=None
        )
        obs = jp.zeros(2)
        return State(pipeline_state, obs, reward, done, metrics, info={'goal': self._goal})

    def step(self, state: State, action: jp.ndarray) -> State:
        # action = jp.clip(action, self._action_space.low, self._action_space.high)
        new_position = state.pipeline_state.x.pos + action
        delta_position = new_position - self._goal
        distance = jp.sqrt(jp.sum((new_position - self._goal) ** 2))
        done = jp.where(distance < 0.01, 1.0, 0.0)
        reward = -distance
        #jax.debug.print("Δ position = {d}, new_position = {p}, goal = {g}, distance={dst}",
        #                d=delta_position, p=new_position, g=self._goal, dst=distance)
        next_observation = new_position
        state.metrics.update(
            x_position=new_position[0],
            y_position=new_position[1],
            distance_from_origin=math.safe_norm(new_position),
            forward_reward=reward,
        )
        new_pipeline_state = state.pipeline_state.replace(
            x=state.pipeline_state.x.replace(pos=new_position)
        )
        return State(new_pipeline_state, next_observation, reward=reward, done=done, metrics=state.metrics, info=state.info)

