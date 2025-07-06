import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.random
import jax.numpy as jnp
from ecorobot import envs

def play(task,robot_type):


    env = envs.get_environment(task, robot_type=robot_type)

    key = jax.random.PRNGKey(0)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(key)
    done = False

    cum_reward = 0
    states = []
    while (True):

        action = jnp.ones((env.action_size,))
        state = jit_step(state, action)
        # env.render(state)
        cum_reward += float(state.reward)
        print("Reward so far" + str(cum_reward))
        states.append(state.pipeline_state)
        print(len(states))

        if state.done:
            break

    env.show_rollout(states, save_dir="example_runs",
                                    filename=task + "_robot_" + robot_type +  "_dummy_" + str(cum_reward) + ".png")


if __name__ == "__main__":

    play(task="locomotion", robot_type="halfcheetah")
    






