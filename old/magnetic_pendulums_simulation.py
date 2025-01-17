import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def rotate(vec, th):
    c, s = np.cos(th), np.sin(th)
    return np.dot(np.array([[c, -s], [s, c]]), vec)

def argmax(arr):
    return np.max(np.abs(arr))

# Konstanten
g = 9.8   # Erdbeschleunigung [m/s^2]
dt = 0.01 # Zeitschritt [s]
t_max = 25 # Gesamte Simulationszeit [s]
# Zeitarray
time_series = np.arange(0, t_max, dt)
num_steps = len(time_series)

###########
#PENDEL 1#
###########
L1 = 1.0   # Pendellänge [m]
theta1 = np.zeros(num_steps, dtype=np.float32)
omega1 = np.zeros(num_steps, dtype=np.float32) 
# Anfangsbedingungen Pendel 1
theta1[0] = np.pi / 2  # Startwinkel (90°)
omega1[0] = 0          # Startgeschwindigkeit


###########
#PENDEL 2#
###########
L2 = 1.2   # Pendellänge [m]
theta2 = np.zeros(num_steps, dtype=np.float32)
omega2 = np.zeros(num_steps, dtype=np.float32)
# Anfangsbedingungen Pendel 2
theta2[0] = np.pi / 3.5  # Startwinkel (60°)
omega2[0] = 0          # Startgeschwindigkeit

# Euler-Integration
for i, _ in enumerate(tqdm(time_series[1:-1]), start=1):
    # Pendel 1
    a_grav1 = - (g / L1) * np.sin(theta1[i - 1])  # Winkelbeschleunigung
    omega1[i] = omega1[i - 1] + a_grav1 * dt      # omega1[i+1] = ...
    theta1[i] = theta1[i - 1] + omega1[i] * dt    # theta1[i+1] = ...

    # Pendel 2
    a_grav2 = - (g / L2) * np.sin(theta2[i - 1])  # Winkelbeschleunigung
    omega2[i] = omega2[i - 1] + a_grav2 * dt      # omega2[i+1] = ...
    theta2[i] = theta2[i - 1] + omega2[i] * dt    # theta2[i+1] = ...

############
#PENDEL 1 #
############

# Bob-Position (x,y)
bob_pos1 = L1 * np.array([np.sin(theta1), -np.cos(theta1)]).T

# Für die Visualisierung: Geschwindigkeits- und Beschleunigungsvektoren (optional)
bob_vel1 = (
    rotate(bob_pos1.T, np.pi / 2).T
    * omega1.reshape((num_steps, 1))
    / argmax(omega1)
    * 0.5
)
alpha1 = np.diff(omega1)
bob_acc1 = (
    rotate(bob_pos1[:-1].T, np.pi / 2).T
    * alpha1.reshape((num_steps - 1, 1))
    / argmax(alpha1)
    * 15.0
)

############
#PENDEL 2 #
############

# Bob-Position (x,y)
bob_pos2 = L2 * np.array([np.sin(theta2), -np.cos(theta2)]).T

# Für die Visualisierung: Geschwindigkeits- und Beschleunigungsvektoren (optional)
bob_vel2 = (
    rotate(bob_pos2.T, np.pi / 2).T
    * omega2.reshape((num_steps, 1))
    / argmax(omega2)
    * 0.5
)
alpha2 = np.diff(omega2)
bob_acc2 = (
    rotate(bob_pos2[:-1].T, np.pi / 2).T
    * alpha2.reshape((num_steps - 1, 1))
    / argmax(alpha2)
    * 15.0
)


################
#Animation     #
################
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(4, 4))
ax_vis = fig.add_subplot(111)

# Fenster-Titel (optional)
figure_title = "Two Simple pendulums (Only Visual)"
fig.suptitle(figure_title, fontsize=20)

# Achseneinstellungen
ax_vis.set_xlim(-2.0, 2.0)
ax_vis.set_ylim(-3.2, 1.2)
ax_vis.set_aspect("equal", "box")
ax_vis.axis("off")  # Keine Achsen anzeigen


# Pendel 1
# Zeichne Pendelstange und Bob
rod1, = ax_vis.plot([0, bob_pos1[0, 0]], [0, bob_pos1[0, 1]],
                   color="black", lw=3)
bob1, = ax_vis.plot(bob_pos1[0, 0], bob_pos1[0, 1],
                   "o", markersize=20, color="red")

# Pendel 2
# Zeichne Pendelstange und Bob
rod2, = ax_vis.plot([0, bob_pos2[0, 0]], [0, bob_pos2[0, 1]],
                   color="blue", lw=3)
bob2, = ax_vis.plot(bob_pos2[0, 0], bob_pos2[0, 1],
                     "o", markersize=20, color="green")


# Fixpunkt (Drehpunkt)
ax_vis.plot(0, 0, "ok", ms=5, color="black")

def animate(frame):
    #Pendel 1
    # Aktualisiere Stange
    rod1.set_data([0, bob_pos1[frame, 0]], [0, bob_pos1[frame, 1]])
    # Aktualisiere Bob
    bob1.set_data([bob_pos1[frame, 0]], [bob_pos1[frame, 1]])

    # Geschwindigkeitsvektor
    vel_arrow1 = ax_vis.arrow(
        bob_pos1[frame, 0], bob_pos1[frame, 1],
        bob_vel1[frame, 0], bob_vel1[frame, 1],
        color="green", head_width=0.05, head_length=0.1
    )
    # Beschleunigungsvektor
    if frame < num_steps - 1:
        acc_arrow1 = ax_vis.arrow(
            bob_pos1[frame, 0], bob_pos1[frame, 1],
            bob_acc1[frame, 0], bob_acc1[frame, 1],
            color="purple", head_width=0.05, head_length=0.1
        )
    else:
        acc_arrow1 = None

    #Pendel 2
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
anim = FuncAnimation(fig, animate, frames=num_steps - 1, interval=10, blit=True)
plt.show()
