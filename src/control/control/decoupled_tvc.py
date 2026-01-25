import matplotlib.pyplot as plt
import numpy as np

from rocket_util.integrator import RK_

# step input
ref_force_x = 1  # N

pos_x = 0
pos_y = 0
vel_x = 0
vel_y = 0
theta = 0
dtheta = 0

drymass = 1  # kg
propmass = 9  # kg
mass = drymass + propmass  # kg

Isp = 300  # s
g0 = 9.81  # m/s^2
tw_ratio = 1.2  # thrust to weight ratio
max_thrust = tw_ratio * (drymass + propmass) * g0  # N

total_length = 0.2  # m
drymoi = (1 / 12) * drymass * total_length**2  # kg*m^2
propmoi = (1 / 12) * propmass * total_length**2  # kg*m^2

min_throttle = 0.4  # minimum throttle fraction
max_throttle = 1.0  # maximum throttle fraction

alpha = 0
throttle = 1

t0 = 0
tf = 5
dt = 0.02

prev_t = 0
prev_err = 0

kp = -0.1
ki = -0.1
kd = -0.1

points = [[], [], [], [], [], [], [], [], [], [], [], []]  # time, theta, dtheta, error, control input, pos_x, pos_y, vel_x, vel_y, mass, derr, int_err

for t in np.arange(t0, tf, dt):
    points[0].append(t)
    points[1].append(theta * 180 / np.pi)  # convert to degrees for plotting
    points[2].append(dtheta * 180 / np.pi)  # convert to degrees for plotting
    points[5].append(pos_x)
    points[6].append(pos_y)
    points[7].append(vel_x)
    points[8].append(vel_y)
    points[9].append(mass)

    desired_throttle = mass * g0 / np.cos(theta + alpha) / max_thrust
    force_x = max_thrust * desired_throttle * np.sin(theta + alpha)

    # compute feedback
    err = ref_force_x - force_x
    derr = (err - prev_err) / dt
    _, int_err = RK_.RK4(lambda t, err: err, start_dvar=err, start_indvar=prev_t, step=dt, end_indvar=t)
    int_err = int_err.ravel()[-1]

    points[3].append(force_x)
    points[10].append(derr)
    points[11].append(int_err)

    u = kp * err + ki * int_err + kd * derr
    u = min(max(u, -3 / 180 * np.pi), 3 / 180 * np.pi)

    points[4].append(u * 180 / np.pi)  # convert to degrees for plotting

    def dynamics(t, x):
        pos_x, pos_y, vel_x, vel_y, theta, dtheta, mass = x
        
        force_y = max_thrust * desired_throttle * np.cos(theta[0] + alpha) - mass[0] * g0
        ddtheta = -(drymoi + propmoi)**-1 * (max_thrust * desired_throttle * total_length / 2 * np.sin(u))

        return np.array([
                [vel_x[0]],
                [vel_y[0]],
                [force_x / mass[0]],
                [force_y / mass[0]],
                [dtheta[0]],
                [ddtheta],
                [-(max_thrust * desired_throttle) / (Isp * g0)],
            ]
        )

    _, out = RK_.RK4(dynamics, start_dvar=[[pos_x], [pos_y], [vel_x], [vel_y], [theta], [dtheta], [mass]], start_indvar=t, step=dt, end_indvar=t + dt)
    pos_x, pos_y, vel_x, vel_y, theta, dtheta, mass = out.ravel()

    prev_t = t
    prev_err = err

fig, axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
axs[0].plot(points[0], points[1], label="Theta (deg)")
axs[0].plot(points[0], points[2], label="dTheta (deg/s)")
axs[0].set_ylabel("Angle / Rate (deg)")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(points[0], points[4], label="Control Input (deg)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("gimbal (deg)")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(points[0], points[3], label="Force X (N)")
axs[2].axhline(ref_force_x, color='r', linestyle='--',  label="Reference Force X (N)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Force X (N)")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(points[0], points[10], label="dErr (N/s)")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("dErr (N/s)")
axs[3].legend()
axs[3].grid(True)

axs[4].plot(points[0], points[11], label="Int Err (N*s)")
axs[4].set_xlabel("Time (s)")
axs[4].set_ylabel("Int Err (N*s)")
axs[4].legend()
axs[4].grid(True)


fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
axs2[0].plot(points[0], points[5], label="Position X (m)")
axs2[0].set_ylabel("Position (m)")
axs2[0].plot(points[0], points[6], label="Position Y (m)")
axs2[0].legend()
axs2[0].grid(True)

axs2[1].plot(points[0], points[7], label="Velocity X (m/s)")
axs2[1].set_xlabel("Time (s)")
axs2[1].set_ylabel("Velocity (m/s)")
axs2[1].plot(points[0], points[8], label="Velocity Y (m/s)")
axs2[1].legend()
axs2[1].grid(True)

axs2[2].plot(points[0], points[9], label="Mass (kg)")
axs2[2].set_xlabel("Time (s)")
axs2[2].set_ylabel("Mass (kg)")
axs2[2].legend()
axs2[2].grid(True)

fig.tight_layout()
fig2.tight_layout()
plt.show()
# fig.savefig('rocket_control_results.png')
