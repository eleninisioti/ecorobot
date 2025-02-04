""" An environment where a single robot needs to maximize its speed
"""

from ecorobot.ecorobot.envs.base import EcorobotEnv
from ecorobot.ecorobot.robots.base import RobotWrapper
import gym
from brax import envs
import math
import jax.numpy as jnp
from brax import math
from brax.envs.base import State
from ecorobot.ecorobot.modules.food import Food
from ecorobot.ecorobot.modules.pin import Pin
import jax
from ecorobot.ecorobot import robots
import brax

class SwitchingForaging(EcorobotEnv):

    def __init__(self, robot_type, episode_length, **kwargs):
        super().__init__()
        robot = robots.get_environment(env_name=robot_type, **kwargs)
        robot = RobotWrapper(env=robot, robot_type=robot_type)
        self.add_robot(robot)

        # add food
        food_0 = Food(loc="random", name="food_0")
        self.add_module(food_0)

        food_1 = Food(loc="random", name="food_1")
        self.add_module(food_1)

        food_2 = Food(loc="random", name="food_2")
        self.add_module(food_2)

        # add pin
        pin = Pin(name="pin_target", max_distance=food_0.max_distance)
        self.target = pin
        self.add_module(pin)

        self.food_modules = [food_0, food_1, food_2]

        self.trial_length = 200  # every how many steps a new trial starts (the agent is put back to home)
        self.num_trials = int(episode_length / self.trial_length)
        self.mean_switch_period = 2  # every how many trials are the food locs switched
        self.std_switch_period = 1
        self.episode_length = episode_length

        self.landmarks = {"home": jnp.array([0, 0., 0.5])}

        self.init_sys()


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
        new_info["current_step"] = 0


        #jnp.where((num_resets%steps)==0, jax.random.choice(key, jnp.array([el.q_idx for el in self.food_modules]),replace=False, )

        #targets = (jax.random.choice(key, jnp.array([el.q_idx for el in self.food_modules]),replace=False, shape=(num_resets,)))
        #info_size = self.modules[0].info_size
        ##target_pos = new_state.pipeline_state.q[targets[0]:targets[0] + info_size]
        #target_pos = target_pos.at[2].set(target_pos[2] + 0.5) # place pin a bit above the food
        #target_pos = target_pos.at[2].set(target_pos[2] + 0.5)
        #q_new = new_state.pipeline_state.q.at[
        #        self.target.q_idx:self.target.q_idx + self.target.info_size].set(target_pos)
        #pipeline_state = new_state.pipeline_state.replace(q=q_new)

        #targets_info  = [[targets[el]]*int(new_info["reset_step"]) for el in range(num_resets)]


        #new_info["targets"] = jnp.array([item for sublist in targets_info for item in sublist])
        new_info["key"] = key
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

        # q_new = jnp.zeros(pipeline_state.q.shape)
        q_new = pipeline_state.q.at[
                self.robot.q_idx:self.robot.q_idx + self.robot_state_len].set(reset_robot_pos[:self.robot_state_len])
        qd_new = jnp.where(jnp.logical_or(trial_end, switch_food), jnp.zeros(pipeline_state.qd.shape),
                           pipeline_state.qd)

        # set location food
        loc_food = pipeline_state.x.pos[self.target.pos_idx]
        q_new = q_new.at[self.target.q_idx: self.target.q_idx + self.target.info_size].set(loc_food)


        new_pipeline_state = self.pipeline_init(q_new, qd_new)
        # pipeline_state = jnp.where(jnp.logical_or(trial_end, switch_food), self.pipeline_init(q_new, jnp.zeros(pipeline_state.qd.shape)), pipeline_state )

        # pipeline_state = pipeline_state.replace(x=x_new, q=q_new, qd=qd_new)
        return new_pipeline_state

    def switch_food_old(self, pipeline_state, current_step, target_info):

        new_target = target_info[current_step]
        new_target_pos = jax.lax.dynamic_slice(pipeline_state.q, (new_target,), (self.target.info_size,))
        #new_target_pos = pipeline_state.q[
        #        new_target:new_target + self.target.info_size]
        new_target_pos = new_target_pos.at[2].set(new_target_pos[2] + 0.5)

        q_new = pipeline_state.q.at[
                self.target.q_idx:self.target.q_idx + self.target.info_size].set(new_target_pos)
        pipeline_state = pipeline_state.replace(q=q_new)
        return pipeline_state

    def switch_food(self, pipeline_state, current_step, reset_step, key):

        # we just relocate the pin
        key, next_key = jax.random.split(key)

        # should the pin be moved?
        switch_food = (current_step % reset_step == 0)

        #switch_food = jnp.equal(current_step,10)

        # current pin location

        #loc_food = [5,4,0.5]

        # choose a random food as the next location
        food_pos_idxs = jnp.array([el.pos_idx for el in self.modules if el.type == "food"])
        new_food_idx = jax.random.randint(next_key, shape=(1,), minval=0, maxval=3)

        new_food_pos_idx =jax.lax.dynamic_slice(food_pos_idxs, (new_food_idx), (1,))

        new_loc_food = pipeline_state.x.pos[new_food_pos_idx]
        new_loc_food = new_loc_food.at[2].set(0.5)
        #new_loc_food = new_loc_food.at[2].set(0.5)
        #new_loc_food = jnp.reshape(jnp.where(switch_food, new_loc_food, loc_food), (1,3))
        new_loc_food = jnp.reshape( new_loc_food, (1,3))



        # Update the slice with the new value

        q_new = jax.lax.dynamic_update_slice(pipeline_state.q, jnp.reshape(new_loc_food,(3,)), (self.target.q_idx,))



        q_new = jnp.where(switch_food, q_new, pipeline_state.q )

        qd_new = jnp.where(switch_food, jnp.zeros(pipeline_state.qd.shape),
                           pipeline_state.qd)
        new_pipeline_state = self.pipeline_init(q_new, qd_new)

        return new_pipeline_state, next_key


    def switch_food_works(self, pipeline_state, current_step, reset_step, key):

        # ----- switch food locations -----
        reset_step = 10
        switch_food = (current_step % reset_step == 0)
        loc_small = pipeline_state.q[self.target.q_idx:self.target.q_idx+self.target.info_size]

        new_loc = pipeline_state.x.pos[self.modules[0].pos_idx]

        new_loc_small = jnp.where(switch_food, new_loc, loc_small)

        pos_new = pipeline_state.x.pos.at[self.target.pos_idx].set(new_loc_small)
        x_new = pipeline_state.x.replace(pos=pos_new)

        q_new = pipeline_state.q.at[self.target.q_idx:self.target.q_idx+self.target.info_size].set(new_loc_small)
        qd_new = jnp.where(switch_food, jnp.zeros(pipeline_state.qd.shape),
                           pipeline_state.qd)
        #pipeline_state = pipeline_state.replace(x=x_new, q=q_new)
        new_pipeline_state = self.pipeline_init(q_new, qd_new)


        return new_pipeline_state, key

    def step(self, state, action):

        pipeline_state0 = state.pipeline_state
        state, reward = super().step(state, action)
        pipeline_state = state.pipeline_state



        # calculate reward as distance to targets
        robot_pos = pipeline_state.x.pos[0]

        target_pos = pipeline_state.x.pos[self.target.pos_idx]
        distance_to_target = jnp.sqrt(jnp.sum((robot_pos - target_pos) ** 2))
        reward_food = 1 - (distance_to_target / self.target.max_distance)
        reward = reward_food
        #  start new trial

        #pipeline_state = self.place_robot_at_home(pipeline_state, state.info["current_step"] , state.info["reset_step"])

        #  switch food positions
        pipeline_state, state.info["key"] = self.switch_food(pipeline_state, state.info["current_step"] , state.info["reset_step"], state.info["key"])

        current_step = state.info["current_step"] + 1
        state.info["current_step"] = current_step

        obs = self._get_obs(pipeline_state)

        state.metrics.update(reward_food=reward_food)

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, info=state.info)

        return state

    def _get_obs(self, pipeline_state: State) -> jnp.ndarray:
        """ Observe robot body position and velocities, as well as food location.
        """

        qpos = pipeline_state.q
        qvel = pipeline_state.qd


        if self.robot.robot_attributes["_exclude_current_positions_from_observation"]:
            qpos = qpos[2:]

        robot_pos = pipeline_state.x.pos[0]
        to_all_food = []
        for el in self.modules:
            if el.type == "food":
                target_pos = pipeline_state.q[el.q_idx:el.q_idx+el.info_size]
                to_all_food.append(robot_pos - target_pos)


        qpos = qpos[:-12]
        qvel = qvel[:-12]

        return jnp.concatenate([qpos] + [qvel] + to_all_food)


