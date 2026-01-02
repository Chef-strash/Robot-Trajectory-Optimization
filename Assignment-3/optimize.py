import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

T = 5.0
N = 100
dt = T / (N - 1)
time = np.linspace(0, T, N)

q_start = np.array([0.0, 0.0])
q_end   = np.array([np.pi/2, np.pi/3])

def cubic_trajectory(q0, qf, T, t):
    a0 = q0
    a1 = 0
    a2 = 3*(qf - q0)/T**2
    a3 = -2*(qf - q0)/T**3
    return a0 + a1*t + a2*t**2 + a3*t**3

q1_poly = cubic_trajectory(q_start[0], q_end[0], T, time)
q2_poly = cubic_trajectory(q_start[1], q_end[1], T, time)

def cost_function(x):
    q1 = x[:N]
    q2 = x[N:]
    
    q1_ddot = np.diff(q1, n=2) / dt**2
    q2_ddot = np.diff(q2, n=2) / dt**2
    
    return np.sum(q1_ddot**2 + q2_ddot**2)

x0 = np.concatenate([q1_poly, q2_poly])

constraints = [
    {'type': 'eq', 'fun': lambda x: x[0] - q_start[0]},
    {'type': 'eq', 'fun': lambda x: x[N] - q_start[1]},
    {'type': 'eq', 'fun': lambda x: x[N-1] - q_end[0]},
    {'type': 'eq', 'fun': lambda x: x[2*N-1] - q_end[1]},
]

result = minimize(cost_function, x0, method='SLSQP', constraints=constraints)

q1_opt = result.x[:N]
q2_opt = result.x[N:]

plt.figure()
plt.plot(time, q1_poly, '--', label='q1 Polynomial')
plt.plot(time, q1_opt, label='q1 Optimized')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.title('Joint 1 Trajectory')
plt.show()

plt.figure()
plt.plot(time, q2_poly, '--', label='q2 Polynomial')
plt.plot(time, q2_opt, label='q2 Optimized')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.title('Joint 2 Trajectory')
plt.show()

print("Polynomial cost:", cost_function(x0))
print("Optimized cost :", cost_function(result.x))
