""" An environment where a single robot needs to reach a food item
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
import jax.numpy as jnp
from brax.envs.base import State
from ecorobot.modules.food import Food
from ecorobot import robots
import numpy as onp
from mujoco import mjx
import brax
from brax import base
import mujoco
from brax.io.image import render_array
class ForagingWithCamera(EcorobotEnv):
    """
    ### Description

    The robot needs to get close to a green food item and stay there until the end of the episode. It is equipped with multiple pie-slice sensors that activate
    when they point to the food item. The environment also supports rangefinder sensors that show the distances to obstacles.

    ### Action Space

    The robot's action space

    ### Observation space

    Observations include the x-y velocity of the robot and the observations of the sensors. A sensor returns 1 if the food item is within its
     sight and 0 otherwise. You can also opt in to include the x-y position of the robot's torso.


    ### Rewards

    The reward at each step is the normalized distance to the food item. We consider as maximum distance the distance at initialization, so the reward can become negative if the agent moves further away from the food
    """

    def __init__(self, robot_type, project_dir, **kwargs):

        self.episode_length = 1000

        super().__init__(episode_length=self.episode_length, project_dir=project_dir, **kwargs)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type, n_rangefinder_sensors=0, n_pieslice_sensors=0, camera=True)

        self.add_robot(robot)


        # add food
        max_food_distance = 3.0
        food = Food(loc_type="radial",
                    max_distance=max_food_distance,
                    z_loc=robot.loc[2],
                    radius=10,
                    idx=0,
                    )
        self.target = food
        self.add_module(food)
        self.max_sensor_distance = max_food_distance

        self.distance_reached = 0.45 # minimum distance from target for solving the task
        self.init_sys()


    def reset(self, key):

        new_state = super().reset(key)
        metrics = {
            'reward_forward': 0.0,
            'reward_food': 0.0,
            'reward_survive': 0.0,
            'reward_ctrl': 0.0,
            'x_position': 0.0,
            'y_position': 0.0,
            'food_position_x':0.0,
            'food_position_y':0.0,
            'food_position_z': 0.0,
            'distance_from_origin': 0.0,
            'distance_to_target': 0.0,
            'x_velocity': 0.0,
            'y_velocity': 0.0,
        }

        obs = self._get_obs(new_state.pipeline_state)

        return new_state.replace(metrics=metrics, obs=obs)


    def move_food(self, state, loc):
        new_qpos = state.pipeline_state.qpos.at[self.target.q_idx:self.target.q_idx+self.target.info_size].set(loc)

        new_pipeline = state.pipeline_state.replace(qpos=new_qpos)
        return state.replace(pipeline_state=new_pipeline)



    def step(self, state, action):
        state = super().step(state, action)
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)

        # calculate food reward
        robot_pos = pipeline_state.x.pos[0]
        target_pos = pipeline_state.x.pos[self.target.pos_idx]

        distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
        reward_food = 1 - (distance_to_target / self.target.max_distance)

        reward = reward_food

        state.metrics.update(
            reward_food=reward_food,
        food_position_x=target_pos[0],
        food_position_y=target_pos[1],
        food_position_z=target_pos[2],
        distance_to_target=distance_to_target)


        #done = jnp.where(distance_to_target < self.distance_reached, 1.0,0.0)
        #done = jnp.where(state.info["current_step"] == self.episode_length, 1.0, 0.0)
        #reward = jnp.where(done, reward, 0.0)

        state = state.replace(obs=obs, reward=reward, info=state.info)

        return state

    def render_camera(self, pipeline_state):
        height = 20
        width = 20

        camera = "agent_viewpoint"

        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)

        data = mjx.make_data(self.sys)
        data = data.replace(qpos=pipeline_state.q, qvel=pipeline_state.qd)
        data = mjx.forward(self.sys, data)
        renderer.update_scene(data, camera=camera)
        image =  renderer.render()

        #image = render_array(self.sys, trajectory=pipeline_state, camera=camera,height= height, width=width)

        return image

        # image = render_array(self.sys, trajectory=pipeline_state, camera=camera,height= height, width=width)

        # d = mujoco.MjData(self.sys.mj_model)

        #d.qpos = jnp.array(d.qpos)
        #d.qpos = pipeline_state.q
        #d.qvel = pipeline_state.q


        #mujoco.mj_forward(self.sys.mj_model, data)



    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q[self.robot.idx:self.robot.info_size]
        qvel = pipeline_state.qd[self.robot.idx:self.robot.info_size][:2]

        #target_pos = pipeline_state.q[-self.target.info_size:]
        #pieslice_sensor_info = qpos[:2]-target_pos

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[self.robot.info_size:]

        camera_data = self.render_camera(pipeline_state)
        camera_data = camera_data.ravel()


        #rangefinder_sensor_data = pipeline_state.sensordata
        #pieslice_sensor_info = jnp.where(rangefinder_sensor_data == -1, 1.0, rangefinder_sensor_data)

        #rangefinder_sensor_data = rangefinder_sensor_data.reshape(4, 5)
        #print(onp.array(rangefinder_sensor_data ))
        # Check if any element in each group is not -1
        #pieslice_sensor_info = jnp.where(jnp.any(rangefinder_sensor_data != -1, axis=1), 1, 0)

        #rangefinder_sensor_info = jnp.ones([self.n_rangefinder_sensors])

        return jnp.concatenate([qpos] + [qvel] + [camera_data])

        #return jnp.concatenate([qpos] + [qvel])



