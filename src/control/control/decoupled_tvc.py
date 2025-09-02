import matplotlib.pyplot as plt
import numpy as np

from rocket_util.integrator import RK_

# step input
ref_theta = 10 * np.pi / 180  # 1 degree in radians
theta = 0
dtheta = 0
A = 1 * 20_000 / 1000

t0 = 0
tf = 5
dt = 0.02

prev_t = 0
prev_err = 0

kp = -0.1
ki = -0.1
kd = -0.1
# assert -kp > ki/kd/A

points = [[], [], [], [], []]  # time, theta, dtheta, error, control input

for t in np.arange(t0, tf, dt):
    points[0].append(t)
    points[1].append(theta * 180 / np.pi)  # convert to degrees for plotting
    points[2].append(dtheta * 180 / np.pi)  # convert to degrees for plotting

    # compute feedback
    err = ref_theta - theta
    derr = (err - prev_err) / dt
    _, int_err = RK_.RK4(lambda t, err: err, start_dvar=err, start_indvar=prev_t, step=dt, end_indvar=t)
    int_err = int_err.ravel()[-1]

    points[3].append(err * 180 / np.pi)  # convert to degrees for plotting

    u = kp * err + ki * int_err + kd * derr
    u = min(max(u, -3 / 180 * np.pi), 3 / 180 * np.pi)

    points[4].append(u * 180 / np.pi)  # convert to degrees for plotting

    def dynamics(t, x):
        _, dtheta = x
        ddtheta = -A * np.sin(u)
        return np.array([[dtheta[0]], [ddtheta]])

    _, out = RK_.RK4(dynamics, start_dvar=[[theta], [dtheta]], start_indvar=t, step=dt, end_indvar=t + dt)
    theta, dtheta = out.ravel()

    prev_t = t
    prev_err = err

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(points[0], points[1], label="Theta (deg)")
axs[0].plot(points[0], points[2], label="dTheta (deg/s)")
axs[0].set_ylabel("Angle / Rate (deg)")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(points[0], points[3], label="Error (deg)")
axs[1].plot(points[0], points[4], label="Control Input (deg)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Error / Control (deg)")
axs[1].legend()
axs[1].grid(True)

fig.tight_layout()
plt.show()
# fig.savefig('rocket_control_results.png')
