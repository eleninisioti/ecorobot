# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Environments for training and evaluating policies."""
from brax.envs.base import Env
import functools
from typing import Optional, Type
from ecorobot.envs import simple_foraging
from ecorobot.envs import foraging_with_sensors
from ecorobot.envs import locomotion
from ecorobot.envs import wall_following
from ecorobot.envs import dual_task
from ecorobot.envs import hazardous_foraging
from ecorobot.envs import foraging_with_camera
from ecorobot.envs import switching_foraging
from ecorobot.envs import maze_with_stepping_stones_big
from ecorobot.envs import maze_with_stepping_stones_small
from ecorobot.envs import maze_with_stepping_stones_small_old

from ecorobot.envs import maze_with_stepping_stones_small_conditioned

from ecorobot.envs import locomotion_with_obstacles
from ecorobot.envs import deceptive_maze
from ecorobot.envs import deceptive_maze_easy
from ecorobot.envs import navigation_2d

_envs = {"simple_foraging": simple_foraging.SimpleForaging,
         "foraging_with_sensors": foraging_with_sensors.ForagingWithSensors,
         "locomotion": locomotion.Locomotion,
         "locomotion_with_obstacles": locomotion_with_obstacles.LocomotionWithObstacles,

         "wall_following": wall_following.WallFollowing,
         "dual_task": dual_task.DualTask,
         "hazardous_foraging": hazardous_foraging.HazardousForaging,
         "foraging_with_camera": foraging_with_camera.ForagingWithCamera,
         "switching_foraging": switching_foraging.SwitchingForaging,
         "maze_with_stepping_stones_big": maze_with_stepping_stones_big.SteppingStoneMazeBig,
         "maze_with_stepping_stones_small_new": maze_with_stepping_stones_small.SteppingStoneMazeSmall,
         "maze_with_stepping_stones_small": maze_with_stepping_stones_small_old.SteppingStoneMazeSmall,

         "maze_with_stepping_stones_small_conditioned": maze_with_stepping_stones_small_conditioned.SteppingStoneMazeSmall,

         "deceptive_maze": deceptive_maze.DeceptiveMaze,
         "deceptive_maze_easy": deceptive_maze_easy.DeceptiveMazeEasy,
         "2d_navigation": navigation_2d.PointEnvRandGoal

         }



"""
from ecorobot.ecorobot.envs import locomotion
from ecorobot.ecorobot.envs import foraging
from ecorobot.ecorobot.envs import foraging_sensor
from ecorobot.ecorobot.envs import tmaze
from ecorobot.ecorobot.envs import switching_foraging
from ecorobot.ecorobot.envs import switching_locomotion
from ecorobot.ecorobot.envs import tmaze_sensor
from ecorobot.ecorobot.envs import wall_following
from ecorobot.ecorobot.envs import dual_task
from ecorobot.ecorobot.envs import maze_with_stepping_stones
from ecorobot.ecorobot.envs import maze_with_stepping_stones_small
from ecorobot.ecorobot.envs import point_navigation
from ecorobot.ecorobot.envs import hazardous_foraging
from ecorobot.ecorobot.envs import hazardous_point_navigation
from ecorobot.ecorobot.envs import modded_brax_ant
from ecorobot.ecorobot.envs import modded_brax_halfcheetah
from ecorobot.ecorobot.envs.metaneat import hazardous_point_navigation as metaneat_hazardous_point_navigation
from ecorobot.ecorobot.envs.metaneat import point_navigation as metaneat_point_navigation
from ecorobot.ecorobot.envs.metaneat import navigation_2d
from ecorobot.ecorobot.envs import hazardous_navigation_2d
from brax.envs.wrappers import training



_envs = {"locomotion": locomotion.Locomotion,
         "foraging": foraging.Foraging,
         "foraging_sensor": foraging_sensor.ForagingSensor,
         "switching_foraging": switching_foraging.SwitchingForaging,
         "switching_locomotion": switching_locomotion.SwitchingLocomotion,
         "hazardous_foraging": hazardous_foraging.HazardousForaging,
         "tmaze": tmaze.Tmaze,
         "tmaze_sensor": tmaze_sensor.TmazeSensor,
         "wall_following": wall_following.WallFollowing,
         "dual_task": dual_task.DualTask,
         "maze_with_stepping_stones": maze_with_stepping_stones.SteppingStoneMaze,
         "maze_with_stepping_stones_small": maze_with_stepping_stones_small.SteppingStoneMazeSmall,
         "nonmetaneat_point_navigation": point_navigation.PointNavigation,
         "nonmetaneat_hazardous_navigation_2d": hazardous_navigation_2d.HazardousPointEnvRandGoal,

         "point_navigation": metaneat_point_navigation.PointNavigation,
         "nonmetaneat_hazardous_point_navigation": hazardous_point_navigation.HazardousPointNavigation,
         "hazardous_point_navigation": metaneat_hazardous_point_navigation.HazardousPointNavigation,
         "modded_ant": modded_brax_ant.ModdedAnt,
         "modded_halfcheetah": modded_brax_halfcheetah.ModdedHalfCheetah,
         "2d_navigation": navigation_2d.PointEnvRandGoal,
         "hazardous_2d_navigation": hazardous_navigation_2d.HazardousPointEnvRandGoal

         }
"""


def get_environment(env_name, **kwargs) -> Env:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
  """Adds an environment to the registry.

  Args:
    env_name: environment name string
    env_class: the Env class to add to the registry
  """
  _envs[env_name] = env_class


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword argments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  env = _envs[env_name]( **kwargs )



  return env
