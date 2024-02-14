import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters and initial conditions
R = 5  # Radius in meters
Vmax = 48  # Maximum voltage
b = 0.01  # Proportionality constant for inflow
H0 = 1  # Initial water height in meters
H_desired = 5  # Desired water height in meters

# Time span for the simulation
t_span = np.linspace(0, 16000, 1001)  # in minutes

# Function for a(t) that defines the outflow proportionality constant
def a(t):
    return 0.1 if t <= 20 else 0.2

# Differential equation with control logic to maintain desired height
def dHdt(H, t, R, b, Vmax, H_desired):
    if H < H_desired:
        V = Vmax
    else:
        # Adjust the voltage to maintain the water at the desired height
        V = (a(t) * np.sqrt(H_desired)) / b
    return (b * V - a(t) * np.sqrt(H)) / (np.pi * R**2)

# Solve the differential equation
H = odeint(dHdt, H0, t_span, args=(R, b, Vmax, H_desired))

# Plot the results
plt.plot(t_span, H, label='Water Height')
plt.axhline(y=H_desired, color='r', linestyle='--', label='Desired Water Height')
plt.xlabel('Time (min)')
plt.ylabel('Water Height (m)')
plt.title('Water Level Over Time with Control')
plt.legend()
plt.grid()
plt.show()
