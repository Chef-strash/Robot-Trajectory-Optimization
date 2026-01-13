import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_trajectory(q_start, q_end):
    T = 5.0
    N = 50 
    dt = T / (N - 1)
    time = np.linspace(0, T, N)

    def cubic_trajectory(q0, qf, T, t):
        return q0 + (3*(qf - q0)/T**2)*t**2 + (-2*(qf - q0)/T**3)*t**3

    q1_poly = cubic_trajectory(q_start[0], q_end[0], T, time)
    q2_poly = cubic_trajectory(q_start[1], q_end[1], T, time)
    x0 = np.concatenate([q1_poly, q2_poly])

    def cost_function(x):
        q1 = x[:N]
        q2 = x[N:]
        q1_ddot = np.diff(q1, n=2) / dt**2
        q2_ddot = np.diff(q2, n=2) / dt**2
        return np.sum(q1_ddot**2 + q2_ddot**2)

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - q_start[0]},
        {'type': 'eq', 'fun': lambda x: x[N] - q_start[1]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - q_end[0]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - q_end[1]},
        {'type': 'eq', 'fun': lambda x: x[1] - x[0]},
        {'type': 'eq', 'fun': lambda x: x[N+1] - x[N]},
        {'type': 'eq', 'fun': lambda x: x[N-1] - x[N-2]},
        {'type': 'eq', 'fun': lambda x: x[2*N-1] - x[2*N-2]},
    ]

    result = minimize(cost_function, x0, method='SLSQP', constraints=constraints)
    return result.x

num_samples = 500 
X_data = [] 
Y_data = [] 

for i in range(num_samples):
    s = np.random.uniform(-np.pi, np.pi, 2)
    e = np.random.uniform(-np.pi, np.pi, 2)
    traj = generate_trajectory(s, e)
    X_data.append(np.concatenate([s, e]))
    Y_data.append(traj)
    if (i+1) % 100 == 0: print(f"Generated {i+1} samples...")

X_data = np.array(X_data)
Y_data = np.array(Y_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2)

model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=1000, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error (MSE): {mse:.6f}")