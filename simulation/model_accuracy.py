import numpy as np
from scipy.integrate import odeint

# Constants
g = 9.81  # acceleration due to gravity
# Differential equation as per the user's model
# def differential_equation(state, t, m, k, c):
#     dydt = [state[1], -k/m*state[0] - c/m*state[1] - g]
#     return dydt

def differential_equation(state, t, m, a1):
        dydt = [state[1], -g + (1/m) * a1]
        return dydt

# Simulation parameters
m = 1.0  # mass (arbitrary value)
k = 3  # spring constant (arbitrary value)
c = 0.3  # damping coefficient (arbitrary value)
a1 = 5
initial_state = [0, 0]  # initial state [position, velocity]
t = np.linspace(0, 10, 500)  # time points

# Solving the differential equation (noise-free signal)
# noise_free_signal = odeint(differential_equation, initial_state, t, args=(m, k, c))

noise_free_signal = odeint(differential_equation, initial_state, t, args=(m, a1))

# Function to add Gaussian noise
def add_gaussian_noise(signal, std_dev):
    noise = np.random.normal(0, std_dev, signal.shape)
    return signal + noise

# Function to calculate the average error percentage
def calculate_error_percentage(noisy_signal, original_signal):
    error = np.abs(noisy_signal - original_signal)
    mean_error = np.mean(error)
    mean_original = np.mean(np.abs(original_signal))
    error_percentage = (mean_error / mean_original) * 100
    return error_percentage

# # Initial standard deviation for Gaussian noise
std_dev = 0.1
error_percentage = 0

# Iteratively adjust noise level to achieve 20-25% error
while not (20 <= error_percentage <= 25):
    noisy_signal = add_gaussian_noise(noise_free_signal, std_dev)
    error_percentage = calculate_error_percentage(noisy_signal, noise_free_signal)
    # Adjust the standard deviation based on the current error
    if error_percentage < 20:
        std_dev *= 1.1  # Increase noise
    elif error_percentage > 25:
        std_dev *= 0.9  # Decrease noise

# Final adjusted standard deviation
std_dev, error_percentage

# Print the final adjusted standard deviation and error percentage
print(f"Final Standard Deviation: {std_dev}")
print(f"Error Percentage: {error_percentage:.2f}%")

# std_dev = 0.7
# noisy_signal = add_gaussian_noise(noise_free_signal, std_dev)
# error_percentage = calculate_error_percentage(noisy_signal, noise_free_signal)
# print(error_percentage)
#0.7 stdv error percentage 21.244497130918162 for the sprinf-damper-mass%
#14.2 stdv error percentage 20.83 for the mass%

