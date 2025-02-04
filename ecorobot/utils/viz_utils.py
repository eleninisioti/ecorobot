
from brax.io import html
from ecorobot import envs
import jax
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import numpy as onp
import jax.numpy as jnp
# ----- general figure configuration -----
cm = 1 / 2.54  # inches to cm
scale = 1
width, height = 3, 2
fig_size = (width / scale / cm, height / scale / cm)
params = {'legend.fontsize': 10,
          "figure.autolayout": True,
          'font.size': 10,
          "figure.figsize": fig_size}
plt.rcParams.update(params)




# Function to create a linear segmented colormap
def create_colormap(colors, n_colors):
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors)


def viz_task_2D_list(rollouts, save_dir):
    """ Visualize multiple rollouts in a task on a 2D plane

    Parameters
    rollout (list of pipeline)
    """

    # for the 2D navigation plots
    colormap = {
        0: ['#ffadad', '#ff5858', '#ff0202', '#ab0000', '#560000'],
        1: ['#bdb2ff', '#745cff', '#2b05ff', '#1a00ad', '#0d0057'],
        2: ['#fdffb6', '#faff60', '#f7ff08', '#aaaf00', '#555800'],
        3: ['#caffbf', '#7eff64', '#34ff0b', '#1eb100', '#0f5900'],
        4: ['#9bf6ff', '#47f0ff', '#00e0f5', '#0096a3', '#004b52'],
        5: ['#a0c4ff', '#4b90ff', '#005ff8', '#003fa5', '#002053'],
        6: ['#ffd6a5', '#ffb050', '#fb8a00', '#a75c00', '#542e00'],
        7: ['#ffc6ff', '#ff6cff', '#ff11ff', '#b600b6', '#5b005b'],
        8: ['#fffffc', '#ffff95', '#ffff30', '#caca00', '#656500']
    }
    rollouts = rollouts[:len(colormap.values())]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])
    num_rollouts = len(rollouts)
    if num_rollouts > len(list(colormap.values)):
        num_rollouts = len(list(colormap.values))
    #print("num rollouts ", str(num_rollouts))

    trajectory_data = {}
    active_colors = []

    for rollout_idx in range(num_rollouts):
        rollout = rollouts[rollout_idx]
        timesteps = len(rollout)

        trajectory_data["traj_" + str(rollout_idx)] = []


        x_locs = []
        y_locs = []
        for timestep in range(timesteps):
            temp =rollout[timestep]
            x_locs.append(float(rollout[timestep].x.pos[ 0][0]))
            y_locs.append(float(rollout[timestep].x.pos[ 0][1]))
            #x_locs.append(rollout[timestep].x.pos[0][0])
            #y_locs.append(rollout[timestep].x.pos[0][1])


        active_colors.append(colormap[rollout_idx])

        cmap = create_colormap(colormap[rollout_idx], n_colors=len(x_locs))
        colors = onp.arange(len(x_locs))
        plt.scatter(x_locs, y_locs, marker="o", c=colors, cmap=cmap,
                    label="rollout_" +str(rollout_idx))

        #print(rollout_idx, x_locs, "\n")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.tight_layout()

    # plt.legend()
    plt.savefig(save_dir + "/2d.png", dpi=300)
    plt.clf()




def viz_task_2D(rollouts, save_dir):
    """ Visualize multiple rollouts in a task on a 2D plane

    Parameters
    rollout (list of pipeline)
    """

    # for the 2D navigation plots
    colormap = {
        0: ['#ffadad', '#ff5858', '#ff0202', '#ab0000', '#560000'],
        1: ['#bdb2ff', '#745cff', '#2b05ff', '#1a00ad', '#0d0057'],
        2: ['#fdffb6', '#faff60', '#f7ff08', '#aaaf00', '#555800'],
        3: ['#caffbf', '#7eff64', '#34ff0b', '#1eb100', '#0f5900'],
        4: ['#9bf6ff', '#47f0ff', '#00e0f5', '#0096a3', '#004b52'],
        5: ['#a0c4ff', '#4b90ff', '#005ff8', '#003fa5', '#002053'],
        6: ['#ffd6a5', '#ffb050', '#fb8a00', '#a75c00', '#542e00'],
        7: ['#ffc6ff', '#ff6cff', '#ff11ff', '#b600b6', '#5b005b'],
        8: ['#fffffc', '#ffff95', '#ffff30', '#caca00', '#656500']
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(fig_size[0], fig_size[1])
    num_rollouts = len(rollouts)

    trajectory_data = {}
    active_colors = []

    for rollout_idx in range(num_rollouts):
        rollout = rollouts[rollout_idx]
        timesteps = rollout.x.pos.shape[0]

        trajectory_data["traj_" + str(rollout_idx)] = []


        x_locs = []
        y_locs = []
        for timestep in range(timesteps):
            x_locs.append(rollout.x.pos[timestep, 0, 0][0])
            y_locs.append(rollout.x.pos[timestep, 0, 0][1])
            #x_locs.append(rollout[timestep].x.pos[0][0])
            #y_locs.append(rollout[timestep].x.pos[0][1])
        active_colors.append(colormap[rollout_idx])

        cmap = create_colormap(colormap[rollout_idx], n_colors=len(x_locs))
        colors = onp.arange(len(x_locs))
        plt.scatter(x_locs, y_locs, marker="o", c=colors, cmap=cmap,
                    label="rollout_" +str(rollout_idx))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.tight_layout()

    # plt.legend()
    plt.savefig(save_dir + "/2d.png", dpi=300)
    plt.clf()

    # save just the last one
    rollout_idx = num_rollouts-1

    rollout = rollouts[rollout_idx]

    trajectory_data["traj_" + str(rollout_idx)] = []


    x_locs = []
    y_locs = []
    for timestep in range(timesteps):
        x_locs.append(rollout.x.pos[timestep, 0, 0][0])
        y_locs.append(rollout.x.pos[timestep, 0, 0][1])
    active_colors.append(colormap[rollout_idx])

    cmap = create_colormap(colormap[rollout_idx], n_colors=len(x_locs))
    colors = onp.arange(len(x_locs))
    plt.scatter(x_locs, y_locs, marker="o", c=colors, cmap=cmap,
                label="rollout_" +str(rollout_idx))
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend()
    plt.tight_layout()

    # plt.legend()
    plt.savefig(save_dir + "/2d_final.png", dpi=300)
    plt.clf()

def viz_html(rollouts, env_config, save_dir):
    """ Visualize rollout in html.

    Parameters
    rollouts: (list of  pipeline). each item corresponds to a different rollout and each rollout is a pipeline state
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_rollouts = len(rollouts)

    env = envs.get_environment(env_name=env_config["env_name"],
                               backend=env_config["backend"],
                                robot_type=env_config["robot_type"])

    for rollout_idx in range(num_rollouts):

        rollout_list = []
        num_steps = env_config["rollout_length"]

        
        for step in range(num_steps):
            step_state = jax.tree_util.tree_map(
                lambda x: x[step,0, ...],
                rollouts[rollout_idx])
            rollout_list.append(step_state)


        output = html.render(env.sys.replace(dt=env.dt), rollout_list)


        with open(save_dir + "/rollout_" + str(rollout_idx) + ".html", "w") as f:
            f.write(output)


def viz_rollout(rollout, data, env_config, save_dir):
    """ Visualize rollout in html.

    Parameters
    rollouts: (list of  pipeline). each item corresponds to a different rollout and each rollout is a pipeline state
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = envs.get_environment(env_name=env_config["env_name"],
                               backend=env_config["env_params"]["backend"],
                               robot_type=env_config["env_params"]["robot_type"],
                               project_dir=env_config["env_params"]["project_dir"],
                               episode_length=env_config["env_params"]["max_steps"])



    output = html.render(env.sys.replace(dt=env.dt), rollout)

    with open(save_dir + "/rollout_reward" + str(jnp.sum(data["reward"])) + ".html", "w") as f:
        f.write(output)



