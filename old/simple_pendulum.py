import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

##################################
#        Helper functions        #
##################################


def rotate(vec, th):
    c, s = np.cos(th), np.sin(th)
    return np.dot(np.array([[c, -s], [s, c]]), vec)


def argmax(arr):
    return np.max(np.abs(arr))


############################
#        Simultaion        #
############################

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]

# Parameters
t_max = 25.0  # [s]
dt = 0.01  # [s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
theta = np.zeros(num_steps, dtype=np.float32)
omega = np.zeros(num_steps, dtype=np.float32)

# Initial conditions
theta[0] = np.pi / 2 
omega[0] = 0

# Run simulation
for i, _ in enumerate(tqdm(time_series[1:-1]), start=1):
    a_grav = -g / L * np.sin(theta[i - 1])
    omega[i] = omega[i - 1] + a_grav * dt
    theta[i] = theta[i - 1] + omega[i] * dt

# Bob position in Cartesian coordinates
bob_pos = L * np.array([np.sin(theta), -np.cos(theta)]).T
bob_vel = (
    rotate(bob_pos.T, np.pi / 2).T
    * omega.reshape((num_steps, 1))
    / argmax(omega)
    * 0.5
)
alpha = np.diff(omega)
bob_acc = (
    rotate(bob_pos[:-1].T, np.pi / 2).T
    * alpha.reshape((num_steps - 1, 1))
    / argmax(alpha)
    * 50.0
)


###########################
#        Animation        #
###########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(13, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[1, 1])
ax_phase = fig.add_subplot(gs[1, 0])
ax_time = fig.add_subplot(gs[0, :])
figure_title = "Simple pendulum Svetozar"
fig.suptitle(figure_title, fontsize=25)
plt.get_current_fig_manager().set_window_title(
    f"{figure_title}, L={L}, dt={dt}"
)

# Visual
ax_vis.set_title("Visual view", fontsize=20)
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])
ax_vis.set_xlim(-1.25 * L, 1.25 * L)
ax_vis.set_ylim(-1.25 * L, 1.25 * L)
ax_vis.set_aspect("equal", "box")

# Phase-space
ax_phase.set_title("Phase space", fontsize=20)
ax_phase.set_xlabel(r"$\theta$\ [rad]", fontsize=15)
ax_phase.set_ylabel(r"$\omega$\ [rad/s]", fontsize=15)
ax_phase.set_xlim(min(theta) - 0.5, max(theta) + 0.5)
ax_phase.set_ylim(min(omega) - 0.5, max(omega) + 0.5)
ax_phase.set_aspect("equal", "box")

# Time plot
ax_time.set_title("Time plot", fontsize=20)
ax_time.set_xlabel(r"$t$\ [s]", fontsize=15)
ax_time.set_ylabel(r"$\theta$\ [rad]", fontsize=15)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(-1.2, 1.2)

# Set up everything
rod = ax_vis.plot(
    (0, bob_pos[0, 0]),
    (0, bob_pos[0, 1]),
    color="black",
    solid_capstyle="round",
    lw=3,
)[0]
bob = ax_vis.plot(
    (bob_pos[0, 0]),
    (bob_pos[0, 1]),
    "o",
    markersize=20,
    color="red",
)[0]
fix = ax_vis.plot([0, 0], [0, 0], "o", markersize=5, color="black")[0]
phase_plt = ax_phase.plot(theta[0], omega[0], "red")[0]
time_plt_theta = ax_time.plot(time_series[0], theta[0], "blue")[0]
time_plt_omega = ax_time.plot(time_series[0], omega[0], "green")[0]
time_plt_alpha = ax_time.plot(time_series[0], alpha[0], "purple")[0]


# Animation function
def animate(frame):
    rod.set_data([0, bob_pos[frame, 0]], [0, bob_pos[frame, 1]])
    bob.set_data(
        [bob_pos[frame, 0]],
        [bob_pos[frame, 1]],
    )
    vel_arrow = ax_vis.arrow(
        bob_pos[frame, 0],
        bob_pos[frame, 1],
        bob_vel[frame, 0],
        bob_vel[frame, 1],
        color="green",
        head_width=0.05,
        head_length=0.1,
    )
    acc_arrow = ax_vis.arrow(
        bob_pos[frame, 0],
        bob_pos[frame, 1],
        bob_acc[frame, 0],
        bob_acc[frame, 1],
        color="purple",
        head_width=0.05,
        head_length=0.1,
    )

    phase_plt.set_data(theta[:frame], omega[:frame])
    time_plt_theta.set_data(time_series[:frame], theta[:frame] / argmax(theta))
    time_plt_omega.set_data(time_series[:frame], omega[:frame] / argmax(omega))
    time_plt_alpha.set_data(
        time_series[:frame], alpha[:frame] / argmax(alpha[:-1])
    )
    return [
        rod,
        bob,
        vel_arrow,
        acc_arrow,
        phase_plt,
        time_plt_theta,
        time_plt_omega,
        time_plt_alpha,
    ]


# Run animation
anim = FuncAnimation(fig, animate, frames=num_steps - 1, interval=0, blit=True)
plt.show()