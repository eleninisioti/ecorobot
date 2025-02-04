""" An environment where a single robot needs to reach a food item
"""

from ecorobot.envs.base import EcorobotEnv
from ecorobot.robots.base import RobotWrapper
import jax.numpy as jnp
from brax.envs.base import State
from ecorobot.modules.food import Food
from ecorobot import robots
from ecorobot.modules.hazard import Hazard


class HazardousForaging(EcorobotEnv):
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

    def __init__(self, robot_type, project_dir, n_rangefinder_sensors=5, n_pieslice_sensors=0, **kwargs):

        self.episode_length = 1000

        super().__init__(episode_length=self.episode_length, project_dir=project_dir, **kwargs)
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type, n_rangefinder_sensors=n_rangefinder_sensors, n_pieslice_sensors=0)

        self.add_robot(robot)

        self.build_terrain()

        self.num_pieslice_sensors = n_pieslice_sensors

        # add food

        food = Food(loc_type="radial",
                    radius=10,
                    idx=0,
                    color="0 1 0 0" # make food invisible so that its not detected by sensors that are only for walls
                    )
        self.target = food
        self.add_module(food)
        self.distance_reached = 0.45 # minimum distance from target for solving the task

        self.init_sys()

    def build_terrain(self):
        self.num_hazards = 25

        distance = 3


        offset = 1
        width = int(self.num_hazards/10)
        self.hazard_locs = []
        for x_loc in range(-width*offset,(width+1)*offset, offset):
            for y_loc in range(-width*offset,(width+1)*offset, offset):
                loc = [x_loc, y_loc]
                self.hazard_locs.append([x_loc, y_loc])

        del self.hazard_locs[12]

        self.num_hazards = len(self.hazard_locs)




        # place hazards

        for hazard in range(self.num_hazards):
            # x = hazard/(self.num_hazards/4)*2 + offset -2
            # y = hazard%(self.num_hazards/4)*2 + offset

            loc = [self.hazard_locs[hazard][0], self.hazard_locs[hazard][1], 0]
            hazard = Hazard(xml_idx=hazard,
                            loc=loc,
                            name="hazard_" + str(hazard)
                            )
            self.add_module(hazard)
            #self.hazard_locs.append(loc[:2])


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
        done = jnp.where(state.info["current_step"] == self.episode_length, 1.0, 0.0)
        #reward = jnp.where(done, reward, 0.0)

        state = state.replace(obs=obs, reward=reward, done=done, info=state.info)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q[self.robot.idx:self.robot.info_size]
        qvel = pipeline_state.qd[self.robot.idx:self.robot.info_size][:2]

        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[self.robot.info_size:]

        # rangefinder sensor shows the distance to the nearest item. if nothing is detected we give 1
        sensor_data = pipeline_state.sensordata
        rangefinder_sensor_info = jnp.where(sensor_data == -1, 1.0, sensor_data)

        return jnp.concatenate([qpos] + [qvel] + [rangefinder_sensor_info] )

        #return jnp.concatenate([qpos] + [qvel])



