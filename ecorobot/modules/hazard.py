from ecorobot.modules.module import Module
from xml.etree import ElementTree
import jax.numpy as jnp
import jax


class Hazard(Module):

    def __init__(self, name, xml_idx,width=0.12, length=0.1, height=2, euler="90 0 0",loc=[0,0] ):
        self.has_pos = False
        self.name=name
        super().__init__(name, self.has_pos)
        self.xml_idx = str(xml_idx)

        self.color = "1 0 0 1"
        self.shape = "capsule"
        self.type = "hazard"
        self.info_size = 0
        self.width = width
        self.length = length
        self.loc = loc
        self.euler = euler

        self.quart = "0.7071 0.7071 0.0 0.0"

        self.height = height
        self.contype = 2



    def get_element(self):
        self.size = str(self.width) + " " + str(self.length)
        new_body_name = self.name
        loc = self.loc

        new_body = ElementTree.Element("body", {"name": new_body_name,  "pos": ' '.join([str(x) for x in loc] )})


        """
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

        joint_z = ElementTree.SubElement(new_body, "joint", {"armature": "0", "axis": "0 0 1", "damping": "0",
                                                             "limited": "true", "name": new_body_name + "_z",
                                                             "pos": "0 0 0",
                                                             "range": "-300 300", "stiffness": "1",
                                                             "type": "hinge"})
        """


        geom = ElementTree.SubElement(new_body, "geom", {"contype": str(self.contype),
                                                         "conaffinity": str(self.contype),
                                                         "name": new_body_name,
                                                         "pos": "0 0 0", "size": str(self.size),
                                                         "type": self.shape,
                                                         "quat": self.quart,
                                                         #"euler": self.euler,
                                                         "mass": "100",
                                                        "rgba": self.color})



        init_q_data = ""

        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return jnp.array([])

    def reset(self, rng):
        return jnp.array([])