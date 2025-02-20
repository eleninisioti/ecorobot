# from rllab.spaces import Box
import collections
from gym.spaces import Box
from brax.envs.base import Env, State
from brax import math, base
import jumpy as jp
import jax
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches

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


def sample_goal_from_circle(key: jax.random.PRNGKey):
    key1, key2 = jax.random.split(key)
    length = jax.random.uniform(key=key1, minval=1.0, maxval=1.0)
    angle = jp.pi * jax.random.uniform(key=key2, minval=0.0, maxval=2.0)

    x = length * jp.cos(angle)
    y = length * jp.sin(angle)
    return jp.array([x, y])


def sample_goal_from_two(goalidx: int):
    id = goalidx % 2
    sample = [jp.array((jp.cos(0.785398), jp.sin(0.785398))), jp.array((jp.cos(3.92699), jp.sin(3.92699)))]
    return sample[id]


def sample_goal_from_four(goalidx: int):
    length = 1
    angle = jp.array([45, 135, 225, 315])[goalidx]
    x = length * jax.numpy.cos(jax.numpy.deg2rad(angle))
    y = length * jax.numpy.sin(jax.numpy.deg2rad(angle))
    return jp.array([x, y])


def sample_goals(num_goals, key: jax.random.PRNGKey):
    return jax.random.uniform(-0.5, 0.5, size=(num_goals, 2,), key=key)


class PointEnvRandGoal(Env):
    def __init__(self, goalidx=0, genidx=0, predefined_goals=False, backend=None,
                 **kwargs):  # Can set goal to test adaptation.
        #super().__init__(**kwargs)
        seed = (int(genidx) << 16) + int(goalidx)
        key = jax.random.PRNGKey(seed=seed)
        self._goal = sample_goal_from_four(goalidx=goalidx) if predefined_goals else sample_goal_from_circle(key=key)
        self.goal_idx = goalidx
        self._action_space = Box(low=-0.1, high=0.1, shape=(2,))
        self.episode_length = 100
        self.reward_for_solved = 1
        self.num_tasks = 1


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

    def reset(self, rng: jp.ndarray, env_params) -> State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'forward_reward': zero,
        }
        pipeline_state = base.State(
            q=None,  # jp.zeros(1),
            qd=None,  # jp.zeros(1),
            x=base.Transform.create(pos=jp.zeros(2)),
            xd=base.Motion.create(vel=jp.zeros(1)),
            contact=None
        )
        obs = jp.zeros(2)
        return State(pipeline_state, obs, reward, done, metrics, info={'goal': self._goal})

    def step(self, state: State, action: jp.ndarray) -> State:
        # action = jp.clip(action, self._action_space.low, self._action_space.high)
        old_position = state.pipeline_state.x.pos

        new_position = state.pipeline_state.x.pos + action * 0.1

        old_distance = jp.sqrt(jp.sum((old_position - self._goal) ** 2))
        new_distance = jp.sqrt(jp.sum((new_position - self._goal) ** 2))

        reward = old_distance - new_distance
        # reward = -new_distance

        reward = jp.where(new_distance > 0.8, 0, reward)

        done = jp.where(new_distance < 0.01, x=1.0, y=0.0)

        # jax.debug.print("Δ position = {d}, new_position = {p}, goal = {g}, distance={dst}",
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
        return State(new_pipeline_state, next_observation, reward=reward, done=done, metrics=state.metrics,
                     info=state.info)

    def viz_trajectory_old(self, states, env, save_file):
        """ Visualize the path """
        plt.figure()
        axis = plt.gca()
        # Plot the exploration paths
        xs, ys = states[:, 0], states[:, 1]
        axis.plot(xs, ys, color="g", marker="", alpha=0.5)

        # viz poison
        axis.scatter(env.goal_rec[0][0], env.goal_rec[0][1], color="green", s=100)

        # viz poison

        # axis.set_xlim(-2, 2)
        # axis.set_ylim(-2, 2)
        if save_file is None:
            plt.show()
        else:
            save_dir = os.path.join(save_file)
            plt.savefig(save_dir)
            plt.close()

    def viz_trajectory(self, states, env, reward, save_file):
        """Visualize the path with color-changing triangles."""
        plt.figure()
        axis = plt.gca()
        axis.set_xlim(-1.1, 1.1)
        axis.set_ylim(-1.1, 1.1)

        # Convert states to numpy array
        states = np.array(states)

        # Color palette
        colors = ["#7400b8", "#6930c3", "#5e60ce", "#5390d9", "#4ea8de",
                  "#48bfe3", "#56cfe1", "#64dfdf", "#72efdd", "#80ffdb"][::-1]

        num_steps = len(states) - 1  # Number of trajectory steps
        color_indices = np.linspace(0, len(colors) - 1, num_steps).astype(int)

        # Plot trajectory as color-changing triangles
        for i in range(num_steps):
            x= states[i][0]
            y= states[i][1]

            triangle = patches.RegularPolygon((x, y), numVertices=3, radius=0.05,
                                              color=colors[color_indices[i]])
            axis.add_patch(triangle)

        # Add white triangle at (0,0)
        # center_triangle = patches.RegularPolygon((0, 0), numVertices=3, radius=0.05,
        #                                         facecolor="white", edgecolor="black", linewidth=2)
        # axis.add_patch(center_triangle)

        # Visualize food
        axis.scatter(env._goal[0], env._goal[1],
                     color="green", s=100)



        axis.set_title("Reward " + str(reward))

        # Save or show figure
        if save_file is None:
            plt.show()
        else:
            save_dir = os.path.join(save_file)
            plt.savefig(save_dir)
            plt.close()