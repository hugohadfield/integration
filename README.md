# ODE Integrators

This repository contains implementations of various numerical methods for solving ordinary differential equations (ODEs). The methods included are:

- Forward Euler Method
- Adams-Bashforth 2nd Order Method
- Adams-Bashforth 3rd Order Method
- Heun's Method
- Runge-Kutta 4th Order Method (RK4)
- Backward Euler Method
- Predictor-Corrector Adams-Bashforth 2nd Order Method

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
  - [Forward Euler](#forward-euler)
  - [Adams-Bashforth 2nd Order](#adams-bashforth-2nd-order)
  - [Adams-Bashforth 3rd Order](#adams-bashforth-3rd-order)
  - [Heun's Method](#heuns-method)
  - [Runge-Kutta 4th Order (RK4)](#runge-kutta-4th-order-rk4)
  - [Backward Euler](#backward-euler)
  - [Predictor-Corrector Adams-Bashforth 2nd Order](#predictor-corrector-adams-bashforth-2nd-order)
- [Evaluation](#evaluation)
- [Example](#example)
- [License](#license)

## Installation

To use the code, clone the repository and ensure you have the required dependencies installed. You can install the dependencies using `pip`:

```bash
pip install numpy scipy matplotlib
```

## Usage

The main methods provided in this repository are used to solve ODEs. Each method requires the following parameters:

- `diff_eq`: The differential equation to be solved, given as a callable function.
- `t0`: The initial time.
- `y0`: The initial value(s) of the dependent variable(s).
- `h`: The step size.
- `t_end`: The end time for the integration.

Below is an example of how to use one of the integrators:

```python
import numpy as np
from your_module_name import solve_forward_euler, oscillator_diff_eq

# Define the differential equation
def diff_eq(t, y):
    return oscillator_diff_eq(t, y, omega=2.0)

# Initial conditions
t0 = 0.0
y0 = np.array([1.0, 0.0])
h = 0.2
t_end = 10.0

# Solve the ODE
solution = solve_forward_euler(diff_eq, t0, y0, h, t_end)

# The solution is an array of y values at each time step
print(solution)
```

## Methods

### Forward Euler

The Forward Euler method is a simple and explicit method for solving ODEs.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Euler_method).

```python
from your_module_name import solve_forward_euler
solution = solve_forward_euler(diff_eq, t0, y0, h, t_end)
```

### Adams-Bashforth 2nd Order

The Adams-Bashforth 2nd Order method is an explicit multi-step method.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods).

```python
from your_module_name import solve_adams_bashforth2
solution = solve_adams_bashforth2(diff_eq, t0, y0, h, t_end)
```

### Adams-Bashforth 3rd Order

The Adams-Bashforth 3rd Order method is an explicit multi-step method with higher accuracy.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods).

```python
from your_module_name import solve_adams_bashforth3
solution = solve_adams_bashforth3(diff_eq, t0, y0, h, t_end)
```

### Heun's Method

Heun's method is a predictor-corrector method that improves the Euler method's accuracy.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Heun%27s_method).

```python
from your_module_name import solve_heun
solution = solve_heun(diff_eq, t0, y0, h, t_end)
```

### Runge-Kutta 4th Order (RK4)

The RK4 method is a popular and accurate method for solving ODEs.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).

```python
from your_module_name import solve_RK4
solution = solve_RK4(diff_eq, t0, y0, h, t_end)
```

### Backward Euler

The Backward Euler method is an implicit method that is stable for stiff equations.
More information can be found [on the wikipedia page](https://en.wikipedia.org/wiki/Backward_Euler_method).

```python
from your_module_name import solve_backward_euler
solution = solve_backward_euler(diff_eq, t0, y0, h, t_end)
```

### Predictor-Corrector Adams-Bashforth 2nd Order

This method combines the Adams-Bashforth 2nd Order method with a predictor-corrector scheme.
More information can be found [on the wikipedia page for predictor correctors](https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method).

```python
from your_module_name import solve_predictor_corrector_ad2
solution = solve_predictor_corrector_ad2(diff_eq, t0, y0, h, t_end)
```

## Evaluation

The `evaluate_integrators` function compares the different integrators by solving a simple harmonic oscillator ODE and plotting the results.

```python
from your_module_name import evaluate_integrators
evaluate_integrators()
```

## Example

An example usage of the integrators is provided in the `evaluate_integrators` function, which solves the oscillator differential equation and plots the results.

## License

This project is licensed under the MIT License. See the [LICENSE](licence.txt) file for details.

---

Feel free to explore the code and use these methods in your own projects. Contributions and feedback are welcome!