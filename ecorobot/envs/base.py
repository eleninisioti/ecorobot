""" This script contains the main interface for ecorobot environments.
"""
import sys
import os
from brax import math

from brax.envs.base import PipelineEnv, State
from brax.envs.base import State
import jax
from brax.io import mjcf
from xml.etree import ElementTree
from etils import epath
import jax.numpy as jnp
import os
import jax
from jax import numpy as jp
import mujoco
from brax.io import html
import numpy as onp


class EcorobotEnv(PipelineEnv):
    def __init__(self, project_dir, episode_length, backend="mjx", **kwargs):
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        self.episode_length = episode_length
        self.modules = []

        self.world_file = project_dir + "/mujoco.xml"
        self.base_env_kwargs = kwargs
        self.sensors = []

    @property
    def action_size(self) -> int:
        return self.robot.action_size

    def modify_sys(self, new_body):
        worldbody = self.element_tree.find(".//worldbody")
        new_body.tail = "\n"
        worldbody.append(new_body)

        with open(self.world_file, "wb") as f:
            ElementTree.ElementTree(self.element_tree).write(f,method="xml")

        return mjcf.load(self.world_file)


    def add_robot(self, robot, init_loc=[0,0]):
        self.robot = robot
        self.robot.loc = jnp.array(init_loc + [self.robot.default_z_loc])

        xml_string = epath.Path(robot.xml_file).read_text()
        self.element_tree = ElementTree.fromstring(xml_string)
        self.element_tree = self.robot.add_sensors(self.element_tree)
        #self.element_tree = self.robot.add_camera(self.element_tree)
        #self.element_tree = self.robot.move(self.element_tree, init_loc+[self.robot.default_z_loc])

        self.sys = mjcf.load(robot.xml_file)

        with open(self.world_file, "wb") as f:
            ElementTree.ElementTree(self.element_tree).write(f, method="xml")

        self.robot.set_torso_size(self.sys)
        self.robot_state_len = self.robot.info_size
        self.robot.pos_idx =  [idx for idx, el in enumerate(self.sys.link_names) if el == "torso"][0]
        self.robot.q_idx = 0
        self.current_pipeline_q_index = len(self.sys.init_q)


    def add_module(self, module):

        # add to xml file and update sys
        new_body, init_q_data = module.get_element()

        qpos = self.element_tree.find(".//numeric[@name='init_qpos']")
        try:
            data_attribute = qpos.get('data')
            data_attribute += init_q_data
            qpos.set('data', data_attribute)

        except AttributeError:
            pass
        module.q_idx = self.current_pipeline_q_index
        self.sys = self.modify_sys(new_body)
        try:
            self.current_pipeline_q_index = len(self.sys.init_q)
        except AttributeError as e:
            print(e)
            print("check")
        self.modules.append(module)


    def init_sys(self):
        kwargs= self.base_env_kwargs
        backend = self.robot.env.backend

        n_frames = 5
        sys = self.sys

        if backend in ['spring', 'positional']:
            sys = self.sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        if backend == 'positional':
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )


        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        self.num_links = len(self.sys.link_names)

        for module in self.modules:
            if module.has_pos:
                module.pos_idx = [idx for idx, el in enumerate(self.sys.link_names) if el == module.name][0]

        super().__init__(sys=sys,  **kwargs)

    def reset(self, rng) -> State:

        # reset robot
        robot_key, module_key = jax.random.split(rng)
        robot_state = self.robot.reset(robot_key)
        q = robot_state.pipeline_state.q
        qd = robot_state.pipeline_state.qd

        #q = q.at[self.robot.q_idx: self.robot.q_idx+self.robot_state_len].set(self.robot.loc)

        states_to_add = []

        # reset modules
        module_keys = jax.random.split(module_key, len(self.modules))
        for module_idx, module in enumerate(self.modules):
            module_state = module.get_pipeline_state((module_keys[module_idx]))
            if module.has_pos:
                states_to_add.append( module_state)
            #q = jnp.concatenate((q, jnp.ravel(module_state)))
            #qd = jnp.concatenate((qd, jnp.ravel(jnp.zeros(module_state.shape))))

        for el in states_to_add:
            #if len(states_to_add[el]):

            q = jnp.concatenate((q, jnp.ravel(el)))
            qd = jnp.concatenate((qd, jnp.ravel(jnp.zeros(el.shape))))

        pipeline_state = self.pipeline_init(q, qd)


        obs = jnp.array([])
        info = {"current_step": 0}
        # ---------------------------------------

        # add food-related metrics
        metrics = {}

        return State(pipeline_state, obs, robot_state.reward, robot_state.done, metrics, info)



    def step(self, state, action):
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        metrics, reward =self.robot.get_metrics(pipeline_state0, pipeline_state, action)
        #state.info["current_step"] = state.info["current_step"] +1

        done =  self.robot.get_done(pipeline_state)
        #done = jnp.where(jnp.greater(state.info["current_step"], self.episode_length), 1.0, done )

        return state.replace(
            pipeline_state=pipeline_state, reward=reward, done=done
        )

    def get_obs_size(self, task):
        return super().observation_size

    def get_action_size(self, task):
        return super().action_size


    def show_rollout(self, states, save_dir, filename):
        output = html.render(self.sys, states)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_dir + "/" + filename + ".html", "w") as f:
            f.write(output)




