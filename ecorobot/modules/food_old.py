from ecorobot.modules.module import Module
from xml.etree import ElementTree
import jax.numpy as jnp
import jax
from mujoco import mjMINVAL
import numpy as onp

class Food(Module):

    def __init__(self, hazard_locs=[], num_hazards=0, radius=5, name="food", idx=0, target=True, loc=[0, 0], loc_type="random", size=1, max_distance=5, min_distance=2, add_joints=True):
        self.has_pos = True
        super().__init__(name, self.has_pos)

        self.color = "0 1 0 1" # green food
        self.shape = "sphere"
        self.num_hazards = num_hazards
        self.type = "food"
        max_size = 0.1
        self.size = size*max_size
        self.reward = (self.size)/max_size
        self.info_size = 3
        self.target = target
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.loc_type = loc_type
        self.loc = loc
        self.idx = idx
        self.radius = radius/2 # for radial location

        self.hazard_locs = jnp.array(hazard_locs)
        #if loc != "random":
        #    self.loc = loc + [self.size]
        #else:
        #    self.loc = loc
        self.contype = 1
        self.add_joints = add_joints # setting to false reduces the memory requirements
        if self.add_joints:
            self.info_size = 3
        else:
            self.info_size = 0
            self.has_pos = False

        self.loc = loc + [self.size]


    def get_element(self):
        new_body_name = self.name
        #loc = [0,0,0] # dummy loc

        loc = self.loc

        new_body = ElementTree.Element("body", {"name": new_body_name, "pos": ' '.join([str(x) for x in loc])})

        if self.add_joints:

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


        geom = ElementTree.SubElement(new_body, "geom", {"contype": str(self.contype), "conaffinity": str(self.contype), "name": new_body_name,
                                                         "pos": "0 0 0", "size": str(self.size),
                                                         "type": self.shape,
                                                         "mass": "1e4",
                                                         "solimp": "0.99 0.99 0.01",
                                                         "solref": "0.01 1",
                                                         # "quat": self.quart,
                                                         # "fromto": self.fromto,
                                                         "friction": "2",

                                                         "rgba": self.color})

        # update initial position
        if self.add_joints:
            init_q_data = " " + ' '.join([str(x) for x in loc])
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
            #target_x = 45
            #target_y = 23
            #target_z = 17
            #self.location = jnp.array([target_x, target_y, target_z])
            #return self.location
            return jnp.array([target_x, target_y, target_z])
        elif self.loc_type == "radial":
            #"ang = jnp.pi * 2.0 * jax.random.uniform(rng)

            #ang = jax.random.choice(rng, jnp.array([30, 60, 120, 150, -30, -60, -120, -150 ]), shape=(1,))[0]
            self.direction =jax.random.choice(rng, jnp.array([0,1,2,3]))
            ang = jnp.array([45, 135, 225, 315 ])[self.direction]
            #ang = 30*jnp.pi/180
            target_x = self.radius * jnp.cos(jnp.deg2rad(ang))
            target_y = self.radius * jnp.sin(jnp.deg2rad(ang))
            target_z = self.size
            self.loc = jnp.array([target_x, target_y, target_z])
            print(self.loc)
            return self.loc

        elif self.loc_type == "random_hazards":


            self.direction = jax.random.choice(rng, jnp.array([0, 1, 2, 3])).astype(int)
            #self.direction = 3

            hazard_index = jnp.array([0, 4, 19, 23 ])[self.direction].astype(int)
            neighbors = jnp.array([5, 9,14, 18])
            neighbor = neighbors[self.direction]
            #hazard_index = jax.random.randint(rng, (1,), 12, 13)

            #hazard_index = jnp.array(5)
            offset = 4
            # x = hazard_index / (self.num_hazards / 4) + offset +1
            #y = hazard_index % (self.num_hazards / 4) + offset
            temp1 = self.hazard_locs[hazard_index]
            temp2 = self.hazard_locs[neighbor]
            temp_locs = onp.array(self.hazard_locs)
            hazard_loc = (self.hazard_locs[hazard_index] + self.hazard_locs[neighbor])/2
            #loc = self.hazard_locs.at[hazard_index].set(hazard_loc)
            #x= self.hazard_locs.at[hazard_index][0] -0.5
            #y= self.hazard_locs.at[hazard_index][1]

            target_z = self.size * 2
            self.loc = jnp.array([hazard_loc[0],hazard_loc[1], target_z])

            print(hazard_loc)
            #quit()





            return self.loc

        else:
            return jnp.array(self.loc + [self.size])