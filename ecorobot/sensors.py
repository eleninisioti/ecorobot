import jax.numpy as jnp
from xml.etree import ElementTree
from jax.scipy.spatial.transform import Rotation as R
from mujoco import mjMINVAL
import mujoco
import brax.math
import math as omath
import xml.etree.ElementTree as ET
import jax
def angle_between_vectors(v1, v2):
    # Calculate the dot product of the two vectors
    dot_product = jnp.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = jnp.linalg.norm(v1)
    magnitude_v2 = jnp.linalg.norm(v2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # Calculate the angle in radians
    angle_radians = jnp.arccos(cosine_theta)

    # Convert the angle to degrees
    angle_degrees = jnp.degrees(angle_radians)

    return angle_degrees


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    Quaternion q = [w, x, y, z]
    """
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])


@jax.jit
def find_cylinder_edges(center, quaternion, height):
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    # The cylinder's axis direction is the z-axis of the rotation matrix
    direction_vector = rotation_matrix[:, 2]

    # Calculate left and right edges (assuming height/2 on each side of center)
    half_height_vector = (height / 2) * direction_vector
    left_edge = center + half_height_vector
    right_edge = center - half_height_vector

    return left_edge, right_edge


@jax.jit
def infer_line_parameters(points):
    # Ensure points is a (3, 2) array

    # Select the point with the smallest x-coordinate as p1
    norms = jnp.linalg.norm(points, axis=0)
    p1 = points[jnp.argmin(norms)]



    # Compute direction vectors
    direction = points[0, ...] - points[1, ...]

    return p1, direction



class Rangefinder:
    """ A compass sensor detects whether the robot has a certain module within a fixed pie-slice.
    It is always located on the periphery of the torso of the agent at its x-direction
    """
    def __init__(self, target, robot_radius, name, id, num_rangefinders, modules, walls, max_distance, max_angle=15, type="hazard"):
        self.robot_radius = robot_radius
        self.target = target
        self.has_pos = True
        self.size = 0.1
        self.info_size = 3
        self.type = "rangefinder"
        self.name = name
        self.max_distance = max_distance
        self.max_angle = max_angle # in degrees
        self.shape = "sphere"
        self.color = "1 0 0 1"
        self.angle = 360/num_rangefinders*id # in degrees
        self.modules = modules
        self.walls = walls


    def get_element(self):
        new_body_name = self.name
        loc = [0.0, 0.0, self.size]
        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": "0 0 0"})

        joint_x = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "1 0 0", "damping": "0",
                                                             "limited": "true", "name": new_body_name + "_x",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "0",
                                                             "type": "slide"})
        joint_y = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 1 0", "damping": "0",
                                                             "limited": "true", "name": new_body_name + "_y",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "0",
                                                             "type": "slide"})

        joint_z = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 0 1", "damping": "1",
                                                             "limited": "true", "name": new_body_name + "_z",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "1",
                                                             "type": "slide"})

        geom = ElementTree.SubElement(new_body, "geom", {"name": new_body_name,
                                                         "pos": "0 0 0", "size": str(self.size),
                                                         "type": self.shape,
                                                         "rgba": self.color})
        init_q_data = " " + ' '.join([str(x) for x in loc])


        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return jnp.array([0.0, 0.0, self.size])

    def is_within_detection_range(self, max_angle, max_distance, robot_pos, sensor_position, target_position):
        # Calculate vector from dot to cube
        vector_dot_to_cube = target_position - sensor_position
        # Normalize the vector
        # dot_product = jnp.dot(sensor_position, target_position)

        # Compute the magnitudes
        # magnitude_A = jnp.linalg.norm(sensor_position)
        # magnitude_B = jnp.linalg.norm(target_position)

        # Compute the cosine of the angle
        # cos_theta = dot_product / (magnitude_A * magnitude_B)

        # Compute the angle in radians
        # theta_radians = jnp.arccos(cos_theta)

        # Convert the angle to degrees
        # angle_deg = jnp.degrees(theta_radians)
        A = sensor_position
        B = target_position
        C = robot_pos
        R = A - C

        # Compute the vector from A to B
        AB = B - A

        # Compute the dot product of R and AB
        dot_product = jnp.dot(R, AB)

        # Compute the magnitudes of R and AB
        magnitude_R = jnp.linalg.norm(R)
        magnitude_AB = jnp.linalg.norm(AB)

        # Compute the cosine of the angle
        cos_theta = dot_product / (magnitude_R * magnitude_AB)

        # Compute the angle in radians
        theta_radians = jnp.arccos(cos_theta)

        # Convert the angle to degrees
        angle_deg = jnp.degrees(theta_radians)

        # slope =vector_dot_to_cube[1]/vector_dot_to_cube[0]
        # angle_radians = jnp.arctan(slope)
        # angle_deg = jnp.degrees(angle_radians)
        # vector_dot_to_cube_normalized = vector_dot_to_cube / jnp.linalg.norm(vector_dot_to_cube)
        # Calculate angle between the vector and the direction perpendicular to the sphere
        # angle = jnp.arccos(jnp.dot(vector_dot_to_cube_normalized, sensor_position / jnp.linalg.norm(sensor_position)))
        # Convert angle to degrees
        # angle_deg = jnp.abs(jnp.degrees(angle))
        # Check if the angle is within the sensor's angle of view and the distance is within the sensor's range
        within = jnp.where(jnp.less_equal(angle_deg, max_angle), 1, 0)
        distance = jnp.linalg.norm(vector_dot_to_cube)
        within = jnp.where(jnp.less_equal(distance, max_distance), within, 0)
        distance = jnp.where(within, distance, self.max_distance)
        return distance



    def get_distance(self, state, sensor_pos):
        #sensor_pos = state.q[self.q_idx:self.q_idx + self.info_size]

        robot_pos =  state.pipeline_state.x.pos[0,...][:2]

        distances = []
        if len(self.walls):
            for el in self.walls:
                center = jnp.array(el.loc)
                quart =  jnp.array([float(x) for x in el.quart.split()])
                length = el.length + el.width*2
                left_edge, right_edge = find_cylinder_edges(center, quart, length)
                points = jnp.array([center[:2], left_edge[:2], right_edge[:2]])

                # Infer p1 and direction
                p1, direction = infer_line_parameters(points)

                # Find min and max x-values and y-values
                min_x, max_x = jnp.min(points[:, 0]), jnp.max(points[:, 0])
                min_y, max_y = jnp.min(points[:, 1]), jnp.max(points[:, 1])
                num_points = 200

                # Determine the range to cover based on the direction vector
                t = jnp.linspace(min_y, max_y, num_points)
                line_points = jnp.column_stack([jnp.full(num_points, p1[0]), t])

                t_2 = jnp.linspace(min_x, max_x, num_points)
                line_points_2 = jnp.column_stack([t_2, jnp.full(num_points, p1[1])])

                line_points = jnp.where(direction[1]==0, line_points_2, line_points)

                t_3 = jnp.linspace(0.0, 1.0, num_points)
                line_points_3 = p1 + t_3[:, None] * direction

                line_points = jnp.where(jnp.logical_and(jnp.logical_not(direction[1]),jnp.logical_not( direction[0])), line_points_3, line_points)

                point_distances = jax.vmap(self.is_within_detection_range, in_axes=(None,None,None, None,0))(self.max_angle, self.max_distance, robot_pos, sensor_pos[:2], line_points)


                distances.append(jnp.min(jnp.array(point_distances)))
        else:
            distances = []
            for module in self.modules:
                if module.type == "wall" or module.type == "hazard":
                    module_pos = state.q[module.q_idx:module.q_idx + module.info_size]
                    distance = self.is_within_detection_range(robot_pos, sensor_pos, module_pos)
                    distances.append(distance)
        return jnp.reshape(jnp.min(jnp.array(distances)),(1,))

    def angle_between_vectors(self, v1, v2):
        # Calculate the dot product of the two vectors
        dot_product = jnp.dot(v1, v2)

        # Calculate the magnitudes of the vectors
        magnitude_v1 = jnp.linalg.norm(v1)
        magnitude_v2 = jnp.linalg.norm(v2)

        # Calculate the cosine of the angle between the vectors
        cosine_theta = dot_product / (magnitude_v1 * magnitude_v2)

        # Calculate the angle in radians
        angle_radians = jnp.arccos(cosine_theta)

        # Convert the angle to degrees
        angle_degrees = jnp.degrees(angle_radians)

        return angle_degrees

    def get_obs(self, state, sensor_pos):
        """ Returns distance to closest object
        """
        # I need to fix the logic here, better have the starting index of target_pos and sensor_pos
        #sensor_pos = state.x.pos[self.pos_idx]
        #sensor_pos = self.pos

        # find angle of robot
        robot_pos = state.pipeline_state.x.pos[0,...]

        obj_distance = self.get_distance(state, sensor_pos)
        max_distance = self.max_distance

        norm_distance = obj_distance/max_distance


        return norm_distance

    def get_reward(self, state):


        target_distance = self.get_distance(state)
        reward = 1 - (target_distance / self.distance)


        return reward, target_distance



    def reset(self, pipeline_state):
        """ The compass sensor is always placed on top of the robot and
         looking forward in the positive x direction"""
        torso_loc = pipeline_state.x.pos[0]

        radians = brax_old.math.quat_to_euler(pipeline_state.x.rot[0,...])
        torso_z_rot = radians[-1] + self.angle*jnp.pi/180
        x_new = jnp.cos(torso_z_rot)*self.robot_radius
        y_new = jnp.sin(torso_z_rot)*self.robot_radius

        sensor_pos = jnp.array([torso_loc[0]+x_new , torso_loc[1]+y_new , torso_loc[2]])


        #sensor_pos = torso_loc + 0.2

        return sensor_pos[:2]









class Compass:
    """ A compass sensor detects whether the robot has a certain module within a fixed pie-slice.
    It is always located on the periphery of the torso of the agent at its x-direction
    """
    def __init__(self, target, robot_radius,idx, num_compasses, target_loc, name="compass_target"):
        self.robot_radius = robot_radius
        self.target = target
        self.has_pos = True
        self.size = 0.1
        self.info_size = 3
        self.type = "compass"
        self.name = name
        self.distance = 1000
        self.max_angle = 50 # in degrees
        self.shape = "sphere"
        self.color = "0 0 1 1"
        self.max_distance = 100
        self.angle = 360/num_compasses*idx + 10 # in degrees, make sure it is not on top of a rangefinder
        self.target_loc = target_loc

    def get_element(self):
        new_body_name = self.name
        loc = [0.0, 0.0, self.size]
        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": "0 0 0"})

        joint_x = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "1 0 0", "damping": "0",
                                                             "limited": "true", "name": new_body_name + "_x",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "0",
                                                             "type": "slide"})
        joint_y = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 1 0", "damping": "0",
                                                             "limited": "true", "name": new_body_name + "_y",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "0",
                                                             "type": "slide"})

        joint_z = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 0 1", "damping": "1",
                                                             "limited": "true", "name": new_body_name + "_z",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "1",
                                                             "type": "slide"})

        geom = ElementTree.SubElement(new_body, "geom", {"name": new_body_name,
                                                         "pos": "0 0 0", "size": str(self.size),
                                                         "type": self.shape,
                                                         "rgba": self.color})
        init_q_data = " " + ' '.join([str(x) for x in loc])


        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return jnp.array([0.0, 0.0, self.size])
    def get_distance(self, state):
        target_pos = jnp.array(self.target_loc)
        sensor_pos = state.q[self.q_idx:self.q_idx+self.info_size]
        distance = jnp.sqrt(jnp.sum((target_pos - sensor_pos[:2]) ** 2))
        return distance

    def angle_between_vectors(self, robot_pos,sensor_position, target_position):
        vector_dot_to_cube = target_position - sensor_position

        A = sensor_position
        B = target_position
        C = robot_pos
        R = A - C

        # Compute the vector from A to B
        AB = B - A

        # Compute the dot product of R and AB
        dot_product = jnp.dot(R, AB)

        # Compute the magnitudes of R and AB
        magnitude_R = jnp.linalg.norm(R)
        magnitude_AB = jnp.linalg.norm(AB)

        # Compute the cosine of the angle
        cos_theta = dot_product / (magnitude_R * magnitude_AB)

        # Compute the angle in radians
        theta_radians = jnp.arccos(cos_theta)

        # Convert the angle to degrees
        angle_deg = jnp.degrees(theta_radians)

        within = jnp.where(jnp.less_equal(angle_deg, self.max_angle), 1, 0)
        distance = jnp.linalg.norm(vector_dot_to_cube)
        within = jnp.where(jnp.less_equal(distance, self.max_distance), within, 0)
        return within


    def get_obs(self, state):
        """ Returns whether the target is within the visual field of the sensor
        """
        # I need to fix the logic here, better have the starting index of target_pos and sensor_pos
        target_pos = jnp.array(self.target_loc)
        sensor_pos = state.x.pos[self.pos_idx]

        # find angle of robot
        robot_pos = state.x.pos[0,...]

        target_detected = jnp.abs(self.angle_between_vectors(robot_pos[:2], sensor_pos[:2], target_pos[:2]))

        return target_detected

    def get_reward(self, state):


        target_distance = self.get_distance(state)
        reward = 1 - (target_distance / self.target.max_distance)


        return reward, target_distance



    def reset(self, pipeline_state):
        """ The compass sensor is always placed on top of the robot and
         looking forward in the positive x direction"""
        torso_loc = pipeline_state.x.pos[0]

        radians = brax_old.math.quat_to_euler(pipeline_state.x.rot[0,...])
        torso_z_rot = radians[-1] + self.angle*jnp.pi/180
        x_new = jnp.cos(torso_z_rot)*self.robot_radius
        y_new = jnp.sin(torso_z_rot)*self.robot_radius

        sensor_pos = jnp.array([torso_loc[0]+x_new , torso_loc[1]+y_new , torso_loc[2]])

        #sensor_pos = torso_loc + 0.2

        return sensor_pos