from ecorobot.ecorobot.modules.module import Module
from xml.etree import ElementTree
import jax.numpy as jnp
import jax
from mujoco import mjMINVAL


class Pin(Module):

    def __init__(self, max_distance, name, loc=[0.0, 0.0], xml_idx=0):
        self.name =name
        self.has_pos = True
        self.xml_idx = xml_idx
        self.color = "1 0 0 1"
        self.shape = "sphere"
        self.info_size = 3
        self.size = 0.2
        self.type = "pin"
        self.max_distance = max_distance

        self.loc = loc + [0.5]


    def get_element(self):
        new_body_name = self.name
        loc = [0.0,0.0,0.5] # dummy loc

        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": ' '.join([str(x) for x in loc])})
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
                                                             "range": "-0.51 0.51", "stiffness": "1",
                                                             "type": "hinge"})

        geom = ElementTree.SubElement(new_body, "geom", {  "name": new_body_name,
                                                         "pos": "0 0 0", "size": str(self.size),
                                                         "type": self.shape,

                                                         "rgba": self.color})

        # update initial position
        init_q_data = " " + ' '.join([str(x) for x in loc])

        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return self.reset(rng)

    def reset(self, rng):

        return jnp.array(self.loc)
