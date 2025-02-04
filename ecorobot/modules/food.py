from ecorobot.modules.module import Module
from xml.etree import ElementTree
import jax.numpy as jnp
import jax
from mujoco import mjMINVAL
import numpy as onp

class Food(Module):

    def __init__(self,  idx,  loc_type, ang_offset=0, shape="sphere", color="0 1 0 1",z_loc=0, init_loc=[], radius=5, name="food",size=1, max_distance=5, min_distance=2, has_pos=True):
        super().__init__(name, has_pos)
        self.has_pos = has_pos # if True, the position will be tracked by engine
        self.color = color # green food
        self.shape = shape
        self.type = "food"
        max_size = 0.6
        self.size = size*max_size
        self.info_size = 2
        if loc_type == "radial":
            self.ang_offset = ang_offset # currently only used in swtiching_foraging to make sure that the two items are not too close
            self.max_distance = radius
        elif loc_type == "fixed":
            self.max_distance = max_distance
        self.min_distance = min_distance
        self.loc_type = loc_type
        self.idx = idx
        self.radius = max_distance
        self.contype = 1
        self.dummy_loc = [0,0,self.size] # dummy loc for xml but will be changed by engine
        self.init_loc = init_loc


    def get_element(self):
        new_body_name = self.name
        loc = self.dummy_loc
        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": ' '.join([str(x) for x in loc])})

        if self.shape == "sphere":
            size = str(self.size)
        elif self.shape== "box":
            size = ' '.join([str(x) for x in [self.size, self.size, self.size]])
        if self.has_pos:

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

            """
            joint_z = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 0 1", "damping": "0",
                                                                 "limited": "true", "name": new_body_name + "_z",
                                                                 "pos": "0 0 0",
                                                                 "range": "-300 300", "stiffness": "1",
                                                                 "type": "hinge"})
            """


        geom = ElementTree.SubElement(new_body, "geom", {"contype": str(self.contype),
                                                         "conaffinity": str(self.contype),
                                                         "name": new_body_name,
                                                         "pos": "0 0 0", "size": size,
                                                         "type": self.shape,
                                                         "mass": "1e4",
                                                         "solimp": "0.99 0.99 0.01",
                                                         "solref": "0.01 1",
                                                         "friction": "2",
                                                         "rgba": self.color})

        # update initial position
        if self.has_pos:
            init_q_data = " " + ' '.join([str(x) for x in self.dummy_loc[:-1]])
        else:
            init_q_data = ""
        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return self.reset(rng)

    def reset(self, rng):
        if self.loc_type == "random":
            rng, rng1, rng2 = jax.random.split(rng, 3)

            dist = self.max_distance * jax.random.uniform(rng1) + self.min_distance
            ang = jnp.pi * 2.0 * jax.random.uniform(rng2)

            target_x = dist * jnp.cos(ang)
            target_y = dist * jnp.sin(ang)
            target_z = self.size*2
            return jnp.array([target_x, target_y])
        elif self.loc_type == "radial":
            self.direction =jax.random.choice(rng, jnp.array([0,1,2,3]))
            ang = jnp.array([45, 135, 225, 315 ])[self.direction]
            ang = ang + self.ang_offset
            target_x = self.radius * jnp.cos(jnp.deg2rad(ang))
            target_y = self.radius * jnp.sin(jnp.deg2rad(ang))
            #target_x = 3
            #target_y = 0
            self.loc = jnp.array([target_x, target_y])
            return self.loc
        elif self.loc_type == "fixed":
            return jnp.array(self.init_loc)
