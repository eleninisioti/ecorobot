import copy
from etils import epath

import gym
from brax.envs.base import PipelineEnv
from brax import envs
import jax.numpy as jnp
from brax import math
from xml.etree import ElementTree

class RobotWrapper(gym.Wrapper):
    """ API for accessing all brax environments.
    """

    def __init__(self, env, robot_type, camera=False, only_forward=True, n_rangefinder_sensors=0, n_pieslice_sensors=0):
        self.robot_type = robot_type
        self.only_forward =only_forward
        self.env = env
        self.idx = 0

        self.brax_xml_files = {"ant": epath.resource_path('brax') / 'envs/assets/ant.xml',
                               "halfcheetah": epath.resource_path('brax') / 'envs/assets/half_cheetah.xml',
                               "walker2d": epath.resource_path('brax') / 'envs/assets/walker2d.xml',
                               "hopper": epath.resource_path('brax') / 'envs/assets/hopper.xml',
                               "humanoid": epath.resource_path('brax') / 'envs/assets/humanoid.xml',

                               }
        #self.brax_init_locs = {"ant": jnp.array([0.0, 0.0, 0.75])}
        self.brax_torso_radius = {"ant": 0.25,
                                  "halfcheetah": 0.25, # does not really have a torso radius
                                  "walker2d": 0.25,  # does not really have a torso radius,
                                  "hopper": 0.25,
                                  "humanoid": 0.25
                                  }
        if robot_type in ["discrete_fish", "fish"]:
            self.xml_file = self.env.xml_file
            self.default_z_loc = 0.2
            self.env.torso_radius = 0.2
            self.info_size = 3
        else:
            self.xml_file = self.brax_xml_files[robot_type]
            self.env.torso_radius = self.brax_torso_radius[robot_type]
            self.info_size = 3
            self.default_z_loc = 0.2

        self.name = self.robot_type + "_" + str(self.idx)
        self.robot_attributes = vars(env)

        # rangefinders are positioned in front of the robot, alongside the perimeter of its torso
        self.n_rangefinder_sensors = n_rangefinder_sensors
        self.n_pieslice_sensors = n_pieslice_sensors
        self.camera = camera


    def move(self, element_tree, robot_pos):
        torso = element_tree.find(".//body[@name='torso']")
        torso.set('pos', ' '.join([str(x) for x in robot_pos]))  # Replace '1.0 1.0 1.0' with the desired position
        return element_tree

    def add_sensors(self, element_tree):
        new_element_tree = copy.deepcopy(element_tree)
        idx_range = jnp.floor(self.n_rangefinder_sensors/2).astype(jnp.int32)
        for sensor_idx in range(self.n_rangefinder_sensors):
            new_element_tree = self.add_rangefinder_to_xml(sensor_idx, new_element_tree)

        for sensor_idx in range(self.n_pieslice_sensors):
            new_element_tree = self.add_pieslice_to_xml(sensor_idx, new_element_tree)
        #quit()
        return new_element_tree


    def add_camera(self, element_tree):
        # add site under torso
        if self.camera:
            new_body_name = "agent_viewpoint"
            robot_loc = self.env.init_loc
            camera_loc = jnp.array([self.env.torso_radius,0, self.env.init_loc[2]])
            quat = (jnp.cos( jnp.deg2rad((-90)/2)), 0, jnp.sin(jnp.deg2rad(-90/2)),0) # rotate around y-axis
            torso = element_tree.find(".//body[@name='torso']")

            new_site = ElementTree.SubElement(torso, "camera",
                                           {"name": new_body_name,
                                            "mode": "fixed",
                                            "pos": ' '.join([str(x) for x in camera_loc]),
                                            "quat" : ' '.join([str(x) for x in quat])})
        return element_tree



    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 and q2.

        Parameters:
        - q1: List or numpy array with 4 elements [w1, x1, y1, z1]
        - q2: List or numpy array with 4 elements [w2, x2, y2, z2]

        Returns:
        - A numpy array with 4 elements [w, x, y, z] representing the product quaternion.
        """
        # Extract the components of each quaternion
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        # Compute the product quaternion
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return jnp.array([w, x, y, z])

    def add_rangefinder_to_xml(self, idx, element_tree):

        # add site under torso
        new_body_name = "rangefinder_" + str(idx)
        robot_loc = self.loc
        sensor_angle = (idx)*180/(self.n_rangefinder_sensors-1)
        sensor_loc = jnp.array([jnp.sin(jnp.deg2rad(sensor_angle))*self.env.torso_radius,
                                jnp.cos(jnp.deg2rad(sensor_angle))*self.env.torso_radius])

        loc = [robot_loc[0] + sensor_loc[0], robot_loc[1] + sensor_loc[1], 0]
        torso = element_tree.find(".//body[@name='torso']")

        #quat_y = (jnp.cos( jnp.deg2rad((sensor_angle)/2)), 0, jnp.sin(jnp.deg2rad(sensor_angle/2),0))
        #quat_z = (jnp.cos( jnp.deg2rad(90/2)), 0,jnp.sin(jnp.deg2rad(90/2)),0) # bring z-axis to point at x-aixs

        L = (jnp.cos( jnp.deg2rad((90)/2)), 0, jnp.sin(jnp.deg2rad(90/2)),0) # rotate around y-axis

        R = (jnp.cos( jnp.deg2rad((-90+sensor_angle)/2)),  jnp.sin(jnp.deg2rad((-90+sensor_angle)/2)),0,0) # rotate around z axis
        #quat = str(quat_y[0] + quat_z[0]) + " 0 " + str(quat_z[1]) + " " + str(quat_y[1])
        #quat = self.quaternion_multiply(quat_y, quat_z)
        quat_0 = float(L[0]*R[0] -L[1]*R[1]-L[2]*R[2]-L[3]*R[3])
        quat_1 = float(L[0]*R[1] +L[1]*R[0]+L[2]*R[3]-L[3]*R[2])
        quat_2 = float(L[0]*R[2] -L[1]*R[3]+L[2]*R[0]+L[3]*R[1])
        quat_3 = float(L[0]*R[3] +L[1]*R[2]-L[2]*R[1]+L[3]*R[0])
        quat = [quat_0, quat_1, quat_2, quat_3]
        #quat = R



        print(quat)
        new_site = ElementTree.SubElement(torso, "site",
                                       {"name": new_body_name,

                                        "pos": ' '.join([str(x) for x in loc]),
                                        #"quat" : "0.7071 0 0.7071 0",
                                        #"axisangle" : "0 1 0 1.57",
                                        "quat" : ' '.join([str(x) for x in quat]),
                                       # "quat": "0.707 0 0.707 0",
                                        #"range": "0 10",
                                        #"size": "0.1",
                                        "rgba": "1 0 0 1"})
        #torso.append(new_site)

        # add sensor under mujoco
        sensor = ElementTree.SubElement(element_tree, "sensor")
        new_sensor = ElementTree.SubElement(sensor, "rangefinder",
                                       {"name": new_body_name,
                                        #"cutoff": "10",

                                        "site": new_body_name
                                        })
        #sensor.append(new_sensor)



        return element_tree


    def add_pieslice_to_xml(self, idx, element_tree):

        # add site under torso

        robot_loc = self.env.init_loc
        sensor_angle = (idx)*360/(self.n_pieslice_sensors)

        angles = [-45, -22.5, 0, 22.5, 45]
        for offset_idx, offset_angle in enumerate(angles):
            new_body_name = "rangefinder_" + str(idx) + "_offset_" + str(offset_idx)
            angle = sensor_angle + offset_angle
            sensor_loc = jnp.array([jnp.sin(jnp.deg2rad(angle))*self.env.torso_radius,
                                    jnp.cos(jnp.deg2rad(angle))*self.env.torso_radius])

            loc = [robot_loc[0] + sensor_loc[0], robot_loc[1] + sensor_loc[1], 0]
            torso = element_tree.find(".//body[@name='torso']")

            #quat_y = (jnp.cos( jnp.deg2rad((sensor_angle)/2)), 0, jnp.sin(jnp.deg2rad(sensor_angle/2),0))
            #quat_z = (jnp.cos( jnp.deg2rad(90/2)), 0,jnp.sin(jnp.deg2rad(90/2)),0) # bring z-axis to point at x-aixs

            L = (jnp.cos( jnp.deg2rad((90)/2)), 0, jnp.sin(jnp.deg2rad(90/2)),0) # rotate around y-axis

            R = (jnp.cos( jnp.deg2rad((-90+angle)/2)),  jnp.sin(jnp.deg2rad((-90+angle)/2)),0,0) # rotate around z axis
            #quat = str(quat_y[0] + quat_z[0]) + " 0 " + str(quat_z[1]) + " " + str(quat_y[1])
            #quat = self.quaternion_multiply(quat_y, quat_z)
            quat_0 = float(L[0]*R[0] -L[1]*R[1]-L[2]*R[2]-L[3]*R[3])
            quat_1 = float(L[0]*R[1] +L[1]*R[0]+L[2]*R[3]-L[3]*R[2])
            quat_2 = float(L[0]*R[2] -L[1]*R[3]+L[2]*R[0]+L[3]*R[1])
            quat_3 = float(L[0]*R[3] +L[1]*R[2]-L[2]*R[1]+L[3]*R[0])
            quat = [quat_0, quat_1, quat_2, quat_3]
            #quat = R


            new_site = ElementTree.SubElement(torso, "site",
                                           {"name": new_body_name,
                                            "pos": ' '.join([str(x) for x in loc]),
                                            #"quat" : "0.7071 0 0.7071 0",
                                            #"axisangle" : "0 1 0 1.57",
                                            "quat" : ' '.join([str(x) for x in quat]),
                                           # "quat": "0.707 0 0.707 0",
                                            #"range": "0 10",
                                            #"size": "0.1",
                                            "rgba": "1 0 0 1"})
            #torso.append(new_site)

            # add sensor under mujoco
            sensor = ElementTree.SubElement(element_tree, "sensor")
            new_sensor = ElementTree.SubElement(sensor, "rangefinder",
                                           {"name": new_body_name,
                                            #"cutoff": "10",

                                            "site": new_body_name
                                            })
        #sensor.append(new_sensor)



        return element_tree


    def set_torso_size(self, sys):
        #if self.robot_type == "ant":
        self.torso_size = sys.geom_size[1][0] # torso is always the second body under worldbody

    @property
    def action_size(self) -> int:
        return self.env.sys.act_size()

    def reset(self, rng):
        if self.robot_type == "ant":
           return self.env.reset(rng)

        else:
            return  self.env.reset(rng)


    def step(self, state, action):
        return self.env.step(state, action)

    def get_done(self, pipeline_state):

        if self.robot_type == "halfcheetah" or self.robot_type == "discrete_fish":
            done = jnp.array(0.0)
        else:
            min_z, max_z = self.env._healthy_z_range
            is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
            is_healthy = jnp.where(
                pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy
            )
            # reward = forward_reward
            done = 1.0 - is_healthy if self.env._terminate_when_unhealthy else 0.0
        return done

    def get_metrics(self, pipeline_state0, pipeline_state, action):
        reward, done, zero = jnp.zeros(3)

        metrics = {
            'reward_forward': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'reward_contact': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'forward_reward': zero,
        }

        if self.robot_type == "ant" or self.robot_type=="walker2d":

            velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.env.dt
            forward_reward = velocity[0]

        elif self.robot_type == "humanoid":

            com_before, *_ = self.env._com(pipeline_state0)
            com_after, *_ = self.env._com(pipeline_state)
            velocity = (com_after - com_before) / self.env.dt
            forward_reward = self.env._forward_reward_weight * velocity[0]

        elif self.robot_type == "fish":
            velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
            forward_reward = velocity[0]

        elif self.robot_type == "discrete_fish":
            velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
            forward_reward = velocity[0]

        elif self.robot_type == "discrete_fish_sensor":
            velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
            forward_reward = velocity[0]

        else:
            velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
            forward_reward = velocity[0]


        if self.robot_type == "halfcheetah" or self.robot_type == "discrete_fish" or self.robot_type=="hopper":
            healthy_reward = 0.0
            ctrl_cost = 0.0
        else:

            min_z, max_z = self.env._healthy_z_range
            is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
            is_healthy = jnp.where(
                pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy
            )
            if self.env._terminate_when_unhealthy:
                healthy_reward = self.env._healthy_reward
            else:
                healthy_reward = self.env._healthy_reward * is_healthy
            ctrl_cost = self.env._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0
        metrics.update(
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


        reward = jnp.where(self.only_forward, forward_reward, healthy_reward+forward_reward-ctrl_cost)
        return metrics, reward


