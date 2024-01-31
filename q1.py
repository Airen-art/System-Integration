
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Define the transfer function coefficients
numerator_coeffs = [13, 9]
denominator_coeffs = [1, 5, 17, 11]

# Create the transfer function system
system = signal.TransferFunction(numerator_coeffs, denominator_coeffs)

# Compute the impulse and step responses
t_impulse, impulse_response = signal.impulse(system)
impulse_response *= 33  # scale for 33 units
t_step, step_response = signal.step(system)
step_response *= 2.5  # scale for 2.5 units

# Get the poles, zeros, and gain of the transfer function
zeros, poles, gain = signal.tf2zpk(numerator_coeffs, denominator_coeffs)

# Compute the Bode plot
frequency, mag, phase = signal.bode(system)

# Plotting
plt.figure(figsize=(14, 10))

# Impulse response plot
plt.plot(t_impulse, impulse_response)
plt.title('Impulse Response')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# Step response plot
plt.plot(t_step, step_response)
plt.title('Step Response')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()


# Pole-Zero plot
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='r', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='b', label='Poles')
plt.title('Pole-Zero Plot')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid()
plt.legend()
plt.show()


# Bode magnitude plot
plt.semilogx(frequency, mag)
plt.title('Bode Magnitude Plot')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.grid()

plt.tight_layout()
plt.show()

