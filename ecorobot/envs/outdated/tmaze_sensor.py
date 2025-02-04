from brax.envs import State

import jax.numpy as jnp
import jax
from ecorobot.ecorobot.modules. wall import Wall
from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot import robots
from ecorobot.ecorobot.robots.base import RobotWrapper
from ecorobot.ecorobot.modules.food import Food

class TmazeSensor(EcorobotEnv):

    def __init__(self, robot_type, episode_length, **kwargs):
        super().__init__()

        # add robot
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        self.trial_length = 200 # every how many steps a new trial starts (the agent is put back to home)
        self.num_trials = int(episode_length/self.trial_length)
        self.mean_switch_period = 2 # every how many trials are the food locs switched
        self.std_switch_period = 1
        self.episode_length = episode_length

        # add walls to form maze
        self.build_maze()

        # add food
        loc_food_small = [-self.end_width, self.corridor_length+4]
        food_small = Food(loc=loc_food_small,
                          size=0.3,
                          name="food_small",
                          max_distance=jnp.linalg.norm(jnp.array(loc_food_small)),)
        self.add_module(food_small)
        self.target_small = food_small

        loc_food_big = [self.end_width, self.corridor_length+4]
        food_big = Food(loc=loc_food_big, size=1, name="food_big", max_distance=jnp.linalg.norm(jnp.array(loc_food_big)))
        self.add_module(food_big)
        self.target_big = food_big

        self.init_sys()

    def build_maze(self):

        self.height = 3
        self.corridor_width = 4

        self.end_width = 8
        self.corridor_length = self.corridor_width * 7

        self.landmarks = {"home": jnp.array([0, 0., 0.5]),
                          "end_left": jnp.array([-self.end_width, self.corridor_length+4]),
                          "end_right": jnp.array([self.end_width, self.corridor_length+4]),
                          "turn": jnp.array([0, self.corridor_length])}


        # current design uses cylinders because brax boxes are broken
        self.wall_specification = {0: {"width": 0.5,
                                       "length": self.corridor_length,
                                       "height": self.height,
                                       "loc": [-self.corridor_width / 2, 3.5],
                                       "fromto": [0, self.corridor_length+4, 0, 0, 0, 0]},  # left corridor-wall
                                   1: {"width": 0.5,
                                       "length": self.corridor_length,
                                       "height": self.height,
                                       "loc": [self.corridor_width / 2, 3.5],
                                       "fromto": [0, self.corridor_length+4, 0, 0, 0, 0]},  # left turn wall
                                   2: {"width": 0.5,
                                       "length": self.end_width,
                                       "height": self.height,
                                       "loc": [self.corridor_width / 2, self.corridor_length],
                                       "fromto": [0, 0, 0, self.end_width, 0, 0]},
                                   3: {"width": 0.5,
                                       "length": self.end_width,
                                       "height": self.height,
                                       "loc": [-self.corridor_width / 2, self.corridor_length],
                                       "fromto": [-self.end_width, 0, 0, 0, 0, 0]},
                                   4: {"width": 0.5,
                                       "length": self.corridor_length,
                                       "height": self.height,
                                       "loc": [-self.corridor_width / 2 - self.end_width,
                                               self.corridor_length],
                                       "fromto": [0, self.corridor_length / 4, 0, 0, 0, 0]},
                                   5: {"width": 0.5,
                                       "length": self.end_width,
                                       "height": self.height,
                                       "loc": [self.corridor_width / 2 + self.end_width,
                                               self.corridor_length],
                                       "fromto": [0, self.corridor_length / 4, 0, 0, 0, 0]},
                                   6: {"width": 0.5,
                                       "length": self.corridor_width+self.end_width*3,
                                       "height": self.height,
                                       "loc": [-self.corridor_width - self.end_width-1.5,
                                               self.corridor_length + self.corridor_length/4],
                                       "fromto": [-self.corridor_width, 0,  0, self.end_width*3+self.corridor_width+3,0, 0, 0]},
                                   }

        for wall_idx, wall_features in self.wall_specification.items():
            wall = Wall(xml_idx=wall_idx,
                        width=wall_features["width"],
                        length=wall_features["length"],
                        height=wall_features["height"],
                        loc=wall_features["loc"],
                        fromto=wall_features["fromto"])
            self.add_module(wall)
    def reset(self, key):
        new_state = super().reset(key)

        metrics = {
            'reward_forward': 0.0,
            'reward_food': 0.0,
            'reward_survive': 0.0,
            'reward_ctrl': 0.0,
            'reward_contact': 0.0,
            'x_position': 0.0,
            'y_position': 0.0,
            'distance_from_origin': 0.0,
            'x_velocity': 0.0,
            'y_velocity': 0.0,

        }
        new_info = new_state.info
        new_info["reset_step"] = jnp.floor((jax.random.normal(key, (1,))*self.std_switch_period + self.mean_switch_period)*self.trial_length)
        #new_info["reset_step"] = 50

        new_info["current_step"] = 0

        obs = self._get_obs(new_state.pipeline_state)


        new_state = new_state.replace(info=new_info)
        return new_state.replace(metrics=metrics, obs=obs)

    def place_robot_at_home(self, pipeline_state, current_step, reset_step):
        """ We move the robot at home when a new trial starts and when the food items are switched switched.
        """
        robot_pos = pipeline_state.x.pos[0]
        trial_end = (current_step % self.trial_length == 0)
        switch_food = (current_step % reset_step == 0)
        reset_robot_pos = jnp.where(jnp.logical_or(trial_end, switch_food), self.landmarks["home"], robot_pos)

        pos_new = pipeline_state.x.pos.at[self.robot.pos_idx].set(reset_robot_pos)


        #q_new = jnp.zeros(pipeline_state.q.shape)
        q_new = pipeline_state.q.at[
               self.robot.q_idx:self.robot.q_idx + self.robot_state_len].set(reset_robot_pos[:self.robot_state_len])
        qd_new = jnp.where(jnp.logical_or(trial_end, switch_food),  jnp.zeros(pipeline_state.qd.shape), pipeline_state.qd )


        new_pipeline_state = self.pipeline_init(q_new, qd_new)
        #pipeline_state = jnp.where(jnp.logical_or(trial_end, switch_food), self.pipeline_init(q_new, jnp.zeros(pipeline_state.qd.shape)), pipeline_state )

        #pipeline_state = pipeline_state.replace(x=x_new, q=q_new, qd=qd_new)
        return new_pipeline_state

    def switch_food(self, pipeline_state, current_step, reset_step):

        # ----- switch food locations -----

        switch_food = (current_step % reset_step == 0)
        loc_small = pipeline_state.x.pos[self.target_small.pos_idx]
        loc_big = pipeline_state.x.pos[self.target_big.pos_idx]

        new_loc_small = jnp.where(switch_food, loc_big, loc_small)
        new_loc_big = jnp.where(switch_food, loc_small, loc_big)

        pos_new = pipeline_state.x.pos.at[self.target_small.pos_idx].set(new_loc_small)
        pos_new = pos_new.at[self.target_big.pos_idx].set(new_loc_big)
        x_new = pipeline_state.x.replace(pos=pos_new)

        q_new = pipeline_state.q.at[self.target_big.q_idx:self.target_big.q_idx+self.target_big.info_size].set(new_loc_big)
        q_new = q_new.at[self.target_small.q_idx:self.target_small.q_idx+self.target_small.info_size].set(new_loc_small)

        pipeline_state = pipeline_state.replace(x=x_new, q=q_new)


        return pipeline_state

    def calculate_reward(self, pipeline_state):
        robot_pos = pipeline_state.q[self.robot.q_idx:self.robot.q_idx+self.robot_state_len]

        #loc_small = pipeline_state.x.pos[self.target_small.pos_idx]
        #distance_to_small = jnp.sqrt(jnp.sum((robot_pos - loc_small) ** 2))
        #reward_small = (1 - (distance_to_small / self.target_small.max_distance))*self.target_small.reward

        loc_big = pipeline_state.x.pos[self.target_big.pos_idx][:2]
        distance_to_big = jnp.sqrt(jnp.sum((robot_pos - loc_big) ** 2))
        reward_big = (1 - (distance_to_big / self.target_big.max_distance))*self.target_big.reward
        #return reward_big
        return  reward_big

    def step(self, state, action):
        pipeline_state0 = state.pipeline_state
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state

        current_step = state.info["current_step"] + 1
        state.info["current_step"] = current_step

        # calculate reward as distance to targets
        reward_food = self.calculate_reward(pipeline_state)

        # uncomment this if you want reward only at the end of trial
        #reward_food = jnp.where((current_step % self.trial_length == 0), reward_food, 0.0)

        reward = reward_food

        #  start new trial
        pipeline_state = self.place_robot_at_home(pipeline_state, current_step, state.info["reset_step"])
        #  switch food positions
        pipeline_state = self.switch_food(pipeline_state, current_step, state.info["reset_step"])

        obs = self._get_obs(pipeline_state)



        state.metrics.update(reward_food=reward_food)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ The observation is discrete: a) whether you are at a turning point b) whether you are at the maze-end and whether you are at home.
        """

        #ah the observation should not be the direction to the food but whether you are at home, turn, left and right
        qpos = pipeline_state.q
        qvel = pipeline_state.qd


        # we need to remove the location of the food from the qpos and qvel
        food_info_size = self.target_small.info_size + self.target_big.info_size
        qpos = qpos[:-food_info_size]
        qvel = qvel[:-food_info_size]


        #if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
        #    qpos = qpos[2:]

        robot_pos = pipeline_state.q[self.robot.q_idx:self.robot.q_idx+self.robot_state_len]

        dist_home = jnp.sqrt(jnp.sum((robot_pos - self.landmarks["home"][:2]) ** 2))
        obs_home = jnp.where(jnp.less(dist_home, 0.2), 1.0, 0.0 )

        dist_turn = jnp.sqrt(jnp.sum((robot_pos[:2] - self.landmarks["turn"]) ** 2))
        obs_turn = jnp.where(jnp.less(dist_turn, 0.3), 1.0, 0.0 )

        dist_left = jnp.sqrt(jnp.sum((robot_pos[:2] - self.landmarks["end_left"]) ** 2))
        obs_left = jnp.where(jnp.less(dist_left, 1.0), 1.0, 0.0 )

        dist_right = jnp.sqrt(jnp.sum((robot_pos[:2] - self.landmarks["end_right"]) ** 2))
        obs_right = jnp.where(jnp.less(dist_right, 1.0), 1.0, 0.0 )

        loc_big = pipeline_state.x.pos[self.target_big.pos_idx]

        #to_food = pipeline_state.x.pos[0]-loc_big

        return jnp.concatenate([qpos] + [qvel] + [dist_home.reshape((1,))] + [dist_turn.reshape((1,))] + [dist_left.reshape((1,))] + [dist_right.reshape((1,))] )
        #return jnp.concatenate([qpos] + [qvel] + [to_food])


