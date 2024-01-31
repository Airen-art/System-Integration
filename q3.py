#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:07:30 2024

@author: arafatjahannova
"""

# Required library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, impulse, step, freqresp
from scipy.integrate import odeint
from pyswarm import pso

# System Configuration: Establishing the Transfer Function
num_coeffs = [13, 9]  # Numerator coefficients
denom_coeffs = [1, 5, 17, 11]  # Denominator coefficients
lti_system = TransferFunction(num_coeffs, denom_coeffs)

# System Dynamics Function: Outlines the differential equations of the system
def system_dynamics(y, t, coefficients):
    a3, a2, a1, b3, b2, b1 = coefficients
    dydt = [y[1], y[2], -sum([a1*y[2], a2*y[1], a3*y[0]]) + sum([b1*y[2], b2*y[1], b3*y[0]])]
    return dydt

# PSO Objective Function: Assesses the fit between predicted and actual responses
def pso_objective(coeffs, time, true_response):
    y0 = [0, 0, 0]  # Setting initial conditions
    model_response = odeint(system_dynamics, y0, time, args=(coeffs,))
    error = true_response - model_response[:, 0]
    return np.sum(np.square(error))

# Response Plots Function: Generates various system response plots
def generate_system_plots(tf_system):
    # Plotting Impulse Response
    t_impulse, resp_impulse = impulse(tf_system)
    plt.figure()
    plt.plot(t_impulse * 33, resp_impulse * 33)
    plt.title('Scaled Impulse Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.show()

    # Plotting Step Response
    t_step, resp_step = step(tf_system)
    plt.figure()
    plt.plot(t_step, resp_step * 2.5)
    plt.title('Scaled Step Response')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.show()

    # Generating Pole-Zero Plot
    plt.figure()
    plt.scatter(np.real(tf_system.poles), np.imag(tf_system.poles), marker='x', label='Poles')
    plt.scatter(np.real(tf_system.zeros), np.imag(tf_system.zeros), marker='o', label='Zeros')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.title('Pole-Zero Map')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.legend()
    plt.show()

# Bode Plot Function
def generate_bode_plot(tf_system):
    frequencies, sys_response = freqresp(tf_system)
    plt.figure()
    plt.subplot(211)
    plt.semilogx(frequencies, 20 * np.log10(np.abs(sys_response)))
    plt.title('Magnitude Response in Bode Plot')
    plt.ylabel('Magnitude [dB]')
    plt.subplot(212)
    plt.semilogx(frequencies, np.angle(sys_response, deg=True))
    plt.title('Phase Response in Bode Plot')
    plt.xlabel('sFrequency [rad/s]')
    plt.ylabel('Phase [degrees]')
    plt.show()

# Main script execution
if __name__ == "__main__":
    # Displaying system plots
    generate_system_plots(lti_system)
    generate_bode_plot(lti_system)

    # System Identification with PSO
    time_vector = np.linspace(0, 10, 100)
    actual_sys_response = odeint(system_dynamics, [0, 0, 0], time_vector, args=([1, 5, 17, 11, 13, 9],))

    # Running PSO Optimization
    param_bounds = [-10] * 6, [10] * 6
    optimal_params, min_obj_val = pso(pso_objective, param_bounds[0], param_bounds[1], args=(time_vector, actual_sys_response[:, 0]))
    print(f"Optimized Parameters: {optimal_params}")
    print(f"Minimum Objective Value: {min_obj_val}")
