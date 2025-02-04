from ecorobot.modules.module import Module
from xml.etree import ElementTree
import jax.numpy as jnp
import jax


class Wall(Module):

    def __init__(self, xml_idx, size,  loc, offset , color="0 0 1 1", euler="",fromto=[],quat="0", name="wall"):
        self.name=name
        super().__init__(name, has_pos=False)
        self.xml_idx = xml_idx

        self.offset = offset
        self.color = color
        self.shape = "box"
        self.type = "wall"
        self.info_size = 3
        #self.width = width
        #self.length = length
        self.size = size
        self.loc = loc
        self.euler = euler
        self.quat = ' '.join([str(x) for x in quat])

        self.fromto = ' '.join([str(x) for x in fromto])


        #self.height = height


    def get_quat(self):
        pass
    def get_element(self):
        #self.size = str(self.width) + " " + str(self.length) + " " + str(self.height)
        #self.size = str(self.width) + " " + str(self.length)

        new_body_name = self.name
        loc = (self.loc[0], self.loc[1], self.loc[2])

        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": ' '.join([str(x) for x in loc] )})


        geom = ElementTree.SubElement(new_body, "geom", {"contype": "1", "conaffinity": "1", "name": new_body_name,
                                                         "size": ' '.join([str(x) for x in self.size]),
                                                         "type": self.shape,
                                                         "rgba": self.color,
                                                         #"euler": self.euler,
                                                         "mass": "1e9",
                                                         "solimp": "0.99 0.99 0.01",
                                                         "solref": "0.01 1",
                                                         "quat": self.quat,
                                                         #"fromto": self.fromto,
                                                         "friction": "2",

                                                         #"condim": "1"


                                                         })

        init_q_data = ""

        return new_body, init_q_data

    def get_pipeline_state(self, rng):
        return jnp.array(self.loc)

    def reset(self, rng):
        return jnp.array(self.loc)

    def reset_wall(self, target_loc, state):
        pos = state.pipeline_state.q[self.q_idx:self.q_idx+self.info_size]
        robot_loc = [0,0]
        turn_point = jnp.array([robot_loc[0], target_loc[1]-robot_loc[1]])
        self.height = 3
        self.corridor_width = 0.5

        offset = -self.offset[1]*2
        new_y = jnp.where(target_loc[1]<0, -pos[1]+offset, pos[1] )
        new_y = jnp.where(jnp.logical_and(jnp.equal(self.xml_idx,0),target_loc[1]<0), 1.0, new_y)
        new_y = jnp.where(jnp.logical_and(jnp.equal(self.xml_idx,1),target_loc[1]<0), 2.0, new_y)
        new_y = jnp.where(jnp.logical_and(jnp.equal(self.xml_idx,2),target_loc[1]<0), 3.0, new_y)

        #new_y = jnp.where(jnp.logical_and(jnp.equal(self.xml_idx,2),target_loc[1]<0), 1.5, new_y)

        #new_x = pos[0]
        #new_x = jnp.where(jnp.logical_and(jnp.equal(self.xml_idx,4),target_loc[1]<0), -1, new_x)

        new_pos = pos.at[1].set(new_y)
        return new_pos