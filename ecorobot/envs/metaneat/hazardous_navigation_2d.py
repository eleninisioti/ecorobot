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


def sample_goals(num_goals, key: jax.random.PRNGKey):
    return jax.random.uniform(-0.5, 0.5, size=(num_goals, 2,), key=key)


class HazardousPointEnvRandGoal(Env):
    def __init__(self, goalidx, genidx, predefined_goals=False, backend=None, **kwargs):  # Can set goal to test adaptation.
        super().__init__(**kwargs)
        seed = (int(genidx) << 16) + int(goalidx)
        key = jax.random.PRNGKey(seed=seed)
        self._goal = sample_goal_from_two(goalidx=goalidx) if predefined_goals else sample_goal_from_circle(key=key)
        self.goal_idx = goalidx
        self._action_space = Box(low=-0.1, high=0.1, shape=(2,))

        self.build_terrain()

        num_lidars = 12

        lidar_depth = 0.5

        lidar_angles = jp.linspace(0, 360, num_lidars, endpoint=False)

        # List to store lidar points
        lidar_points = []

        # Calculate the points on the circle
        for angle in lidar_angles:
            x = jp.cos(jp.radians(angle)) * lidar_depth
            y = jp.sin(jp.radians(angle)) * lidar_depth
            lidar_points.append(jp.array([x, y]))

        self.lidar_points = jp.array(lidar_points)



    def render(self):

        fig, ax = plt.subplots()
        # plot food
        ax.scatter(x=self._goal[0], y=self._goal[1], color="green")

        # plot hazards
        for hazard in self.hazard_locs:
            # Add a circle to the plot
            circle = Circle(hazard, self.hazard_radius, color='red', fill=False, label="Circle")
            ax.add_patch(circle)



        # plot agent
        ax.scatter(x=0, y=0, color="orange")

        # plot rangefinder points
        for lidar in self.lidar_points:
            ax.scatter(x=lidar[0], y=lidar[1], color="orange")

        plt.savefig("2dhazards.png")
        plt.clf()




    def build_terrain(self):
        self.num_hazards = 25
        self.hazard_radius = 0.1

        offset = 0.5
        width = int(self.num_hazards/10)
        self.hazard_locs = []
        for x_loc in jp.arange(-width*offset,(width+1)*offset, offset):
            for y_loc in jp.arange(-width*offset,(width+1)*offset, offset):
                loc = [x_loc, y_loc]
                self.hazard_locs.append([x_loc, y_loc])

        del self.hazard_locs[12]
        self.num_hazards = len(self.hazard_locs)
        self.hazard_locs = jp.array(self.hazard_locs)

    @property
    def observation_space(self):
        return Box(low=-jp.inf, high=jp.inf, shape=(2,))

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_size(self):
        return 14
    
    @property
    def action_size(self):
        return 2
    
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return "generalized"

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
        obs = jp.zeros(14)
        return State(pipeline_state, obs, reward, done, metrics, info={'goal': self._goal})

    def get_distance(self, robot_pos, hazard_loc):
        distance = jp.sqrt(jp.sum((robot_pos - hazard_loc) ** 2))
        return distance

    def get_hazard_penalty(self, robot_pos):
        distances = []
        thres = 0.1
        distances = jax.vmap(self.get_distance, in_axes=(None,0))(robot_pos, self.hazard_locs)
        penalty = jp.where(jp.min(distances) < thres, 10.0, 0.0)
        #penalty = 0.0
        return penalty



    def does_line_intersect_circle(self, p1, p2, center, radius):
        # Unpack points and circle data
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = center

        # Calculate the coefficients of the quadratic equation
        dx = x2 - x1
        dy = y2 - y1

        # Quadratic coefficients (At^2 + Bt + C = 0)
        A = dx ** 2 + dy ** 2
        B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        C = (x1 - cx) ** 2 + (y1 - cy) ** 2 - radius ** 2

        # Discriminant
        discriminant = B ** 2 - 4 * A * C

        # If the discriminant is non-negative, the line intersects the circle
        return discriminant >= 0


    def get_lidar_obs(self, robot_pos, lidar_pos):
        p1 = robot_pos
        p2 = p1+lidar_pos

        intersections = jax.vmap(self.does_line_intersect_circle, in_axes=(None,None,0,None))(p1,p2, self.hazard_locs, self.hazard_radius)
        distances = jax.vmap(self.get_distance, in_axes=(None,0))(robot_pos, self.hazard_locs)
        obs = jp.where(intersections, distances, 1)
        obs = jp.min(obs)
        return obs

    def step(self, state: State, action: jp.ndarray) -> State:
        # action = jp.clip(action, self._action_space.low, self._action_space.high)
        new_position = state.pipeline_state.x.pos + action
        delta_position = new_position - self._goal
        distance = jp.sqrt(jp.sum((new_position - self._goal) ** 2))
        done = jp.where(distance < 0.01, 1.0, 0.0)
        reward = -distance
        #jax.debug.print("Δ position = {d}, new_position = {p}, goal = {g}, distance={dst}",
        #                d=delta_position, p=new_position, g=self._goal, dst=distance

        # check if agent is inside a hazard zone
        penalty = self.get_hazard_penalty(new_position)
        reward = reward-penalty

        next_observation = new_position

        # add lidar info to observation
        lidar_obs = jax.vmap(self.get_lidar_obs, in_axes=(None, 0))(new_position, self.lidar_points)
        next_observation = jp.concatenate([next_observation, lidar_obs], axis=0)

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

