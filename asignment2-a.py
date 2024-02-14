import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#SIMULATION FOR a:

# Parameters and initial conditions
R = 5  # Radius in meters
Vmax = 48  # Maximum voltage
a = 0.1  # Proportionality constant for outflow
b = 0.01  # Proportionality constant for inflow
H0 = 0  # Initial water height in meters
H_desired = 4  # Desired water height in meters

# Time span for the simulation
t_span = np.linspace(0, 1600, 1001)  # in minutes

# Differential equation with control logic to maintain desired height
def dHdt(H, t, R, b, Vmax, H_desired, a):
    if H[0] < H_desired:
        V = Vmax
    else:
        # Calculate the voltage required to maintain the desired height
        V = (a * np.sqrt(H_desired)) / b
    return (b * V - a * np.sqrt(H[0])) / (np.pi * R**2)

# Solve the differential equation
H = odeint(dHdt, H0, t_span, args=(R, b, Vmax, H_desired, a))

# Plot the results
plt.plot(t_span, H, label='Water Height')
plt.axhline(y=H_desired, color='r', linestyle='--', label='Desired Water Height')
plt.xlabel('Time (min)')
plt.ylabel('Water Height (m)')
plt.title('Water Level Over Time with Control to Maintain Desired Height')
plt.legend()
plt.grid()
plt.show()
