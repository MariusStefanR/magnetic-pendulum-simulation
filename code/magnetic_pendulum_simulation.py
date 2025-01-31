import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons
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
g = 9.8     # [m/s^2]
L1 = 0.8   # [m]
L2 = 1.0    # [m]
m1 = 0.8    # Mass of bob 1 [kg]
m2 = 1.1    # Mass of bob 2 [kg]

# Parameters
t_max = 25.0    # [s]
dt = 0.01       # [s]

# Electromagnetic paramters
kC = 1.0    # Coulomb constant
Q1 = 1.0    # Charge pendulum 1
Q2 = -2.0   # Charge pendulum 2
   
# Variables
time_series = np.arange(0, t_max, dt)
num_steps = len(time_series)

# Arrays for angular displacement and velocity
theta1 = np.zeros(num_steps, dtype=np.float32)
theta2 = np.zeros(num_steps, dtype=np.float32)
omega1 = np.zeros(num_steps, dtype=np.float32)
omega2 = np.zeros(num_steps, dtype=np.float32)

# Arrays for magnetic forces
Fx_array = np.zeros(num_steps, dtype=np.float32)
Fy_array = np.zeros(num_steps, dtype=np.float32)
coulomb_force = np.zeros(num_steps, dtype=np.float32)

# Arrays for energy
KE1 = np.zeros(num_steps, dtype=np.float32)
KE2 = np.zeros(num_steps, dtype=np.float32)
PE1 = np.zeros(num_steps, dtype=np.float32)
PE2 = np.zeros(num_steps, dtype=np.float32)
E1_total = np.zeros(num_steps, dtype=np.float32)
E2_total = np.zeros(num_steps, dtype=np.float32)

colors = {
    "bob1": "red",
    "bob2": "blue",
    "vel_arrow1": "green",
    "vel_arrow2": "lime",
    "acc_arrow1": "purple",
    "acc_arrow2": "pink",
    "force_arrow1": "orange",
    "force_arrow2": "deepskyblue",
    "kinetic_energy1": "cyan",
    "kinetic_energy2": "mediumvioletred",
    "potential_energy1": "magenta",
    "potential_energy2": "indigo",
    "total_energy1": "brown",
    "total_energy2": "teal",
    "coulomb_force": "darkturquoise" 
}

# Initial conditions
theta1[0] = np.pi / 2
theta2[0] = np.pi / -2
omega1[0] = 0 
omega2[0] = 0

# Run simulation
for i, _ in enumerate(tqdm(time_series[1:-1]), start=1):
    
    # Angular acceleration due to gravity
    a_grav1 = - (g / L1) * np.sin(theta1[i - 1])
    a_grav2 = - (g / L2) * np.sin(theta2[i - 1])

    # Current positions of each bob
    x1 = L1 * np.sin(theta1[i - 1]) 
    y1 = -L1 * np.cos(theta1[i - 1])
    
    x2 = L2 * np.sin(theta2[i - 1])
    y2 = -L2 * np.cos(theta2[i - 1])

    # Distance and direction between bobs
    rx = x2 - x1
    ry = y2 - y1
    r = np.sqrt(rx**2 + ry**2)

    # Numercial safeguard to avoid division by zero
    if r < 1e-6:
        r = 1e-6

    # Coulomb force calculation with direction
    charge_product = Q1 * Q2
    Fx_array[i] = kC * charge_product * rx / (r**3)
    Fy_array[i] = kC * charge_product * ry / (r**3)
    coulomb_force[i] = kC * charge_product / (r**2)

    # Torque on each pendulum
    tau1 = x1 * Fy_array[i] - y1 * Fx_array[i]
    tau2 = x2 * (-Fy_array[i]) - y2 * (-Fx_array[i])

    # Moment of inertia
    I1 = m1 * L1**2
    I2 = m2 * L2**2

    # Angular acceleration due to electromagnetic force
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

    # Kinetic energy
    KE1[i] = 0.5 * m1 * (omega1[i] * L1)**2
    KE2[i] = 0.5 * m2 * (omega2[i] * L2)**2

    # Potential energy
    PE1[i] = m1 * g * (-L1 * np.cos(theta1[i]))
    PE2[i] = m2 * g * (-L2 * np.cos(theta2[i]))

    # Total energy
    E1_total[i] = KE1[i] + PE1[i]
    E2_total[i] = KE2[i] + PE2[i]

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
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(3, 3, width_ratios=[3, 2, 1], height_ratios=[2, 1, 1])
figure_title = "Two magnetic pendulums"
fig.suptitle(figure_title, fontsize=20)

# Visual
ax_vis = fig.add_subplot(gs[0, 0])
ax_vis.set_xlim(-2.0, 2.0)
ax_vis.set_ylim(-3.2, 1.2)
ax_vis.set_aspect("equal", "box")
ax_vis.axis("on")

# Coulomb force subplot
ax_force = fig.add_subplot(gs[1, :])
times = time_series
ax_force.set_title("Coulomb Force Over Time")
ax_force.set_xlabel("Time [s]")
ax_force.set_ylabel("Coulomb Force [N]")
ax_force.grid(True)
force_line, = ax_force.plot([], [], color=colors["coulomb_force"], lw=2)

# Energy subplot
ax_energy = fig.add_subplot(gs[2, :])
ax_energy.set_title("Energy Over Time")
ax_energy.set_xlabel("Time [s]")
ax_energy.set_ylabel("Energy [J]")
ax_energy.grid(True)

# Checkbox for selecting energy graphs
ax_checkbox = fig.add_subplot(gs[0, 1])
ax_checkbox.axis("on")
ax_checkbox.set_title("Select Energy Graphs", fontsize=12)


# Set up pendulum 1
rod1 = ax_vis.plot(
    (0, bob_pos1[0, 0]),
    (0, bob_pos1[0, 1]),
    color="black",
    solid_capstyle="round",
    lw=2,
)[0]
bob1 = ax_vis.plot(
    (bob_pos1[0, 0]),
    (bob_pos1[0, 1]),
    "o",
    markersize=20,
    color=colors["bob1"],
)[0]

# Set up pendulum 2
rod2 = ax_vis.plot(
    (0, bob_pos2[0, 0]),
    (0, bob_pos2[0, 1]),
    color="black",
    solid_capstyle="round",
    lw=2,
)[0]
bob2 = ax_vis.plot(
    (bob_pos2[0, 0]),
    (bob_pos2[0, 1]),
    "o",
    markersize=20,
    color=colors["bob2"],
)[0]

# Plot energy graphs for pendulum 1
ke_line1, = ax_energy.plot([], [], color=colors["kinetic_energy1"], label="Kinetic Energy 1")
pe_line1, = ax_energy.plot([], [], color=colors["potential_energy1"], label="Potential Energy 1")
total_line1, = ax_energy.plot([], [], color=colors["total_energy1"], label="Total Energy 1")

# Plot energy graphs for pendulum 2
ke_line2, = ax_energy.plot([], [], linestyle="--", color=colors["kinetic_energy2"], label="Kinetic Energy 2")
pe_line2, = ax_energy.plot([], [], linestyle="--", color=colors["potential_energy2"], label="Potential Energy 2")
total_line2, = ax_energy.plot([], [], linestyle="--", color=colors["total_energy2"], label="Total Energy 2")

# Store enery plot lines for checkbox control
lines = [ke_line1, pe_line1, total_line1, ke_line2, pe_line2, total_line2]
labels = [line.get_label() for line in lines]
line_colors = [line.get_color() for line in lines]

# Create checkboxes
check = CheckButtons(ax_checkbox, labels, actives=[line.get_visible() for line in lines])

# Apply colors for checkboxes
for label, color in zip(check.labels, line_colors):
    label.set_color(color)

# Toggle visibility of energy graphs
def toggle_lines(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(toggle_lines)

# Set up fixpoint
ax_vis.plot(0, 0, "ok", ms=5, color="black")

# Animation function
def animate(frame):
    
    # Pendulum 1
    rod1.set_data([0, bob_pos1[frame, 0]], [0, bob_pos1[frame, 1]])
    bob1.set_data([bob_pos1[frame, 0]], [bob_pos1[frame, 1]])
    
    vel_arrow1 = ax_vis.arrow(
        bob_pos1[frame, 0], bob_pos1[frame, 1],
        bob_vel1[frame, 0], bob_vel1[frame, 1],
        color=colors["vel_arrow1"], head_width=0.05, head_length=0.1
    )
    
    acc_arrow1 = None
    if frame < num_steps - 1:
        acc_arrow1 = ax_vis.arrow(
            bob_pos1[frame, 0], bob_pos1[frame, 1],
            bob_acc1[frame, 0], bob_acc1[frame, 1],
            color=colors["acc_arrow1"], head_width=0.05, head_length=0.1
        )
    
    force_arrow1 = ax_vis.arrow(
        bob_pos1[frame, 0], bob_pos1[frame, 1],
        Fx_array[frame] * 0.3, Fy_array[frame] * 0.3, 
        color="blue", head_width=0.05, head_length=0.1
    )
    
    # Pendulum 2
    rod2.set_data([0, bob_pos2[frame, 0]], [0, bob_pos2[frame, 1]])
    bob2.set_data([bob_pos2[frame, 0]], [bob_pos2[frame, 1]])

    vel_arrow2 = ax_vis.arrow(
        bob_pos2[frame, 0], bob_pos2[frame, 1],
        bob_vel2[frame, 0], bob_vel2[frame, 1],
        color=colors["vel_arrow2"], head_width=0.05, head_length=0.1
    )
    
    acc_arrow2 = None
    if frame < num_steps - 1:
        acc_arrow2 = ax_vis.arrow(
            bob_pos2[frame, 0], bob_pos2[frame, 1],
            bob_acc2[frame, 0], bob_acc2[frame, 1],
            color=colors["acc_arrow2"], head_width=0.05, head_length=0.1
        )

    force_arrow2 = ax_vis.arrow(
        bob_pos2[frame, 0], bob_pos2[frame, 1],
        -Fx_array[frame] * 0.3, -Fy_array[frame] * 0.3,
        color="cyan", head_width=0.05, head_length=0.1
    )

    # Update Coulomb force graph
    force_line.set_data(times[:frame], coulomb_force[:frame])
    max_force = 1.1 * max(abs(np.min(coulomb_force)), abs(np.max(coulomb_force)))
    ax_force.set_xlim(0, t_max)
    ax_force.set_ylim(-max_force, max_force)  

    # Update energy graphs
    ke_line1.set_data(times[:frame], KE1[:frame])
    pe_line1.set_data(times[:frame], PE1[:frame])
    total_line1.set_data(times[:frame], E1_total[:frame])

    ke_line2.set_data(times[:frame], KE2[:frame])
    pe_line2.set_data(times[:frame], PE2[:frame])
    total_line2.set_data(times[:frame], E2_total[:frame])

    # Adjust energy graph axes
    ax_energy.set_xlim(0, t_max)
    max_energy = max(
        np.max(E1_total), np.max(E2_total),
        np.max(KE1), np.max(KE2),
        np.max(PE1), np.max(PE2),
    ) * 1.1
    min_energy = min(
        np.min(E1_total), np.min(E2_total),
        np.min(KE1), np.min(KE2),
        np.min(PE1), np.min(PE2),
    )
    ax_energy.set_ylim(min_energy, max_energy)

    return [rod1, bob1, rod2, bob2, vel_arrow1, acc_arrow1, vel_arrow2, acc_arrow2, force_line, force_arrow1, force_arrow2, ke_line1, pe_line1, total_line1, ke_line2, pe_line2, total_line2]

# Run animation
anim = FuncAnimation(fig, animate, frames=num_steps - 1, interval=10, blit=True)
plt.tight_layout()
plt.show()
