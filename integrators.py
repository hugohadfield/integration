
from typing import Callable

import numpy as np
from scipy.optimize import minimize


def forward_euler_step(
    diff_eq: Callable, t: float, y: np.ndarray, h: float
):
    """
    A forward Euler integrator step
    """
    return y + h * diff_eq(t, y)


def solve_forward_euler(
    diff_eq: Callable, t0: float, y0: np.ndarray, h: float, t_end: float
):
    """
    Solve an ODE using the forward Euler method
    """
    t = t0
    y = y0
    t_generator = np.arange(t0 + h, t_end + h, h)
    output = np.zeros((len(t_generator) + 1, y0.shape[0]))
    output[0] = y
    for i, t in enumerate(t_generator):
        y = forward_euler_step(diff_eq, t, y, h)
        output[i, :] = y
    return output


def heun_step(
    diff_eq: Callable, t: float, y: np.ndarray, h: float
):
    """
    A Heun integrator step
    """
    y_pred = y + h * diff_eq(t, y)
    return y + h / 2.0 * (
        diff_eq(t, y) + diff_eq(t + h, y_pred)
    )


def solve_heun(
    diff_eq: Callable, t0: float, y0: np.ndarray, h: float, t_end: float
):
    """
    Solve an ODE using the Heun method
    """
    t = t0
    y = y0
    t_generator = np.arange(t0 + h, t_end + h, h)
    output = np.zeros((len(t_generator) + 1, y0.shape[0]))
    output[0] = y
    for i, t in enumerate(t_generator):
        y = heun_step(diff_eq, t, y, h)
        output[i, :] = y
    return output


def RK4_step(
    diff_eq: Callable, t: float, y: np.ndarray, h: float
):
    """
    A forth order runge-kutta integrator step
    """
    k1 = diff_eq(t, y)
    k2 = diff_eq(t + h / 2.0, y + h * k1 / 2.0)
    k3 = diff_eq(t + h / 2.0, y + h * k2 / 2.0)
    k4 = diff_eq(t + h, y + h * k3)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def solve_RK4(
    diff_eq: Callable, t0: float, y0: np.ndarray, h: float, t_end: float
):
    """
    Solve an ODE using the forth order runge-kutta method
    """
    t = t0
    y = y0
    t_generator = np.arange(t0 + h, t_end + h, h)
    output = np.zeros((len(t_generator) + 1, y0.shape[0]))
    output[0] = y
    for i, t in enumerate(t_generator):
        y = RK4_step(diff_eq, t, y, h)
        output[i, :] = y
    return output


def backward_euler_step(
    diff_eq: Callable, t: float, y: np.ndarray, h: float
):
    """
    A backward Euler integrator step
    """
    def func(yk1):
        return np.sum((y + h * diff_eq(t + h, yk1) - yk1)**2)
    
    yk1_initial_guess = y
    res = minimize(
        func, 
        yk1_initial_guess, 
        method = 'l-bfgs-b',
        options={'disp': False,
                'ftol': 1e-10,
                'gtol': 1e-10,}
        )
    yk1 = res.x
    return yk1


def solve_backward_euler(
    diff_eq: Callable, t0: float, y0: np.ndarray, h: float, t_end: float
):
    """
    Solve an ODE using the backward Euler method
    """
    t = t0
    y = y0
    t_generator = np.arange(t0 + h, t_end + h, h)
    output = np.zeros((len(t_generator) + 1, y0.shape[0]))
    output[0] = y
    for i, t in enumerate(t_generator):
        y = backward_euler_step(diff_eq, t, y, h)
        output[i, :] = y
    return output


def oscillator_diff_eq(t: float, y: np.ndarray, omega: float = 1.0):
    """
    The differential equation for the osscilator
    y'' = -omega**2 * y
    """
    return np.array([y[1], -omega**2 * y[0]])


def evaluate_integrators():
    """
    Evaluate the integrators
    """
    import matplotlib.pyplot as plt

    t0 = 0.0
    y0 = np.array([1.0, 0.0])
    h = 0.2
    t_end = 10.0
    omega = 2.0

    forward_euler_output = solve_forward_euler(
        lambda t, y: oscillator_diff_eq(t, y, omega), t0, y0, h, t_end
    )
    heun_output = solve_heun(
        lambda t, y: oscillator_diff_eq(t, y, omega), t0, y0, h, t_end
    )
    RK4_output = solve_RK4(
        lambda t, y: oscillator_diff_eq(t, y, omega), t0, y0, h, t_end
    )
    backward_euler_output = solve_backward_euler(
        lambda t, y: oscillator_diff_eq(t, y, omega), t0, y0, h, t_end
    )

    t = np.linspace(t0, t_end, int((t_end - t0) / h) + 1)
    y = np.cos(omega * t)
    y_dot = -omega * np.sin(omega * t)

    plt.figure()
    plt.plot(t, forward_euler_output[:, 0], label="Forward Euler")
    plt.plot(t, heun_output[:, 0], label="Heun")
    plt.plot(t, RK4_output[:, 0], label="RK4")
    plt.plot(t, backward_euler_output[:, 0], label="Backward Euler")
    plt.plot(t, y, label="Analytical")
    plt.legend()

    plt.figure()
    plt.plot(t, forward_euler_output[:, 1], label="Forward Euler")
    plt.plot(t, heun_output[:, 1], label="Heun")
    plt.plot(t, RK4_output[:, 1], label="RK4")
    plt.plot(t, backward_euler_output[:, 1], label="Backward Euler")
    plt.plot(t, y_dot, label="Analytical")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_integrators()
