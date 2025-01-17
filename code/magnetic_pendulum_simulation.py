import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
L1 = 0.5 # [m]
L2 = 1.0  # [m]

# Parameters
t_max = 25.0  # [s]
dt = 0.01  # [s]

# Magnetic paramters
K = -15 # 'Magnetic constant'; negative (K > 0) for repulsion and positive (K < 0) for attraction
m1 = 1.0 # Mass of bob 1 [kg]
m2 = 1.0 # Mass of bob 2 [kg]
   
# Variables
time_series = np.arange(0, t_max, dt)
num_steps = len(time_series)
theta1 = np.zeros(num_steps, dtype=np.float32)
theta2 = np.zeros(num_steps, dtype=np.float32)
omega1 = np.zeros(num_steps, dtype=np.float32)
omega2 = np.zeros(num_steps, dtype=np.float32)

# Initial conditions
theta1[0] = np.pi / 2
theta2[0] = np.pi / -2
omega1[0] = 0 
omega2[0] = 0

# Run simulation
for i, _ in enumerate(tqdm(time_series[1:-1]), start=1):
    
    # Gravitational acceleration
    a_grav1 = - (g / L1) * np.sin(theta1[i - 1])
    a_grav2 = - (g / L2) * np.sin(theta2[i - 1])

    # Current positions of each bob (for magnetic force)
    x1 = L1 * np.sin(theta1[i - 1]) 
    y1 = -L1 * np.cos(theta1[i - 1])

    x2 = L2 * np.sin(theta2[i - 1])
    y2 = -L2 * np.cos(theta2[i - 1])

    # Magnetic force calculation 
    rx = x2 - x1
    ry = y2 - y1
    r = np.sqrt(rx**2 + ry**2)

    # Numercial safeguard to avoid division by zero
    if r < 1e-6:
        r = 1e-6

    Fx = K * rx / (r**3)
    Fy = K * ry / (r**3)

    # Tau on each pendulum
    tau1 = x1 * Fy - y1 * Fx
    tau2 = x2 * (-Fy) - y2 * (-Fx)

    # Angular acceleration 
    I1 = m1 * L1**2
    I2 = m2 * L2**2
    alpha_mag1 = tau1 / I1
    alpha_mag2 = tau2 / I2

    # Total angular acceleration
    alpha1 = a_grav1 + alpha_mag1
    alpha2 = a_grav2 + alpha_mag2

    # Euler update for each pendulum
    omega1[i] = omega1[i - 1] + alpha1 * dt
    theta1[i] = theta1[i - 1] + omega1[i] * dt
    
    omega2[i] = omega2[i - 1] + alpha2 * dt
    theta2[i] = theta2[i - 1] + omega2[i] * dt

# Bob position in Cartesian coordinates
bob_pos1 = L1 * np.array([np.sin(theta1), -np.cos(theta1)]).T
bob_pos2 = L2 * np.array([np.sin(theta2), -np.cos(theta2)]).T

bob_vel1 = (
    rotate(bob_pos1.T, np.pi / 2).T
    * omega1.reshape((num_steps, 1))
    / argmax(omega1)
    * 0.5
)
bob_vel2 = (
    rotate(bob_pos2.T, np.pi / 2).T
    * omega2.reshape((num_steps, 1))
    / argmax(omega2)
    * 0.5
)

alpha1 = np.diff(omega1)
alpha2 = np.diff(omega2)

bob_acc1 = (
    rotate(bob_pos1[:-1].T, np.pi / 2).T
    * alpha1.reshape((num_steps - 1, 1))
    / argmax(alpha1)
    * 15
)
bob_acc2 = (
    rotate(bob_pos2[:-1].T, np.pi / 2).T
    * alpha2.reshape((num_steps - 1, 1))
    / argmax(alpha2)
    * 15
)


###########################
#        Animation        #
###########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(4, 4))
ax_vis = fig.add_subplot(111)
figure_title = "Two magnetic pendulums (visual view)"
fig.suptitle(figure_title, fontsize=20)

# Visual
ax_vis.set_xlim(-2.0, 2.0)
ax_vis.set_ylim(-3.2, 1.2)
ax_vis.set_aspect("equal", "box")
ax_vis.axis("off")

# Set up pendulum 1
rod1 = ax_vis.plot(
    (0, bob_pos1[0, 0]),
    (0, bob_pos1[0, 1]),
    color="black",
    solid_capstyle="round",
    lw=3,
)[0]
bob1 = ax_vis.plot(
    (bob_pos1[0, 0]),
    (bob_pos1[0, 1]),
    "o",
    markersize=20,
    color="red",
)[0]

# Set up pendulum 2
rod2 = ax_vis.plot(
    (0, bob_pos2[0, 0]),
    (0, bob_pos2[0, 1]),
    color="black",
    solid_capstyle="round",
    lw=3,
)[0]
bob2 = ax_vis.plot(
    (bob_pos2[0, 0]),
    (bob_pos2[0, 1]),
    "o",
    markersize=20,
    color="green",
)[0]

# Set up fixpoint
ax_vis.plot(0, 0, "ok", ms=5, color="pink")

# Animation function
def animate(frame):
    
    # Pendulum 1
    rod1.set_data([0, bob_pos1[frame, 0]], [0, bob_pos1[frame, 1]])
    bob1.set_data([bob_pos1[frame, 0]], [bob_pos1[frame, 1]])
    vel_arrow1 = ax_vis.arrow(
        bob_pos1[frame, 0], bob_pos1[frame, 1],
        bob_vel1[frame, 0], bob_vel1[frame, 1],
        color="green", head_width=0.05, head_length=0.1
    )
    if frame < num_steps - 1:
        acc_arrow1 = ax_vis.arrow(
            bob_pos1[frame, 0], bob_pos1[frame, 1],
            bob_acc1[frame, 0], bob_acc1[frame, 1],
            color="purple", head_width=0.05, head_length=0.1
        )
    else:
        acc_arrow1 = None

    # Pendulum 2
    rod2.set_data([0, bob_pos2[frame, 0]], [0, bob_pos2[frame, 1]])
    bob2.set_data([bob_pos2[frame, 0]], [bob_pos2[frame, 1]])

    vel_arrow2 = ax_vis.arrow(
        bob_pos2[frame, 0], bob_pos2[frame, 1],
        bob_vel2[frame, 0], bob_vel2[frame, 1],
        color="red", head_width=0.05, head_length=0.1
    )
    if frame < num_steps - 1:
        acc_arrow2 = ax_vis.arrow(
            bob_pos2[frame, 0], bob_pos2[frame, 1],
            bob_acc2[frame, 0], bob_acc2[frame, 1],
            color="orange", head_width=0.05, head_length=0.1
        )
    else:
        acc_arrow2 = None    

    return [rod1, bob1, rod2, bob2, vel_arrow1, acc_arrow1, vel_arrow2, acc_arrow2]

# Run animation
anim = FuncAnimation(fig, animate, frames=num_steps - 1, interval=10, blit=True)
plt.show()
