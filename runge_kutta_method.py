def runge_kutta_3rd(f, x0, y0, h, x_end):
    """
    Applies the 3rd order Runge-Kutta method.

    Args:
        f (function): The function f(x, y) defining the ODE.
        x0 (float): The initial x value.
        y0 (float): The initial y value.
        h (float): The step size.
        x_end (float): The end x value.

    Returns:
        tuple: Lists of x values and approximated y values.
    """
    x = x0
    y = y0
    x_values = [x]
    y_values = [y]

    while x < x_end:
      k1 = h * f(x, y)
      k2 = h * f(x + h/2, y + k1/2)
      k3 = h * f(x + h, y - k1 + 2*k2)
      y = y + (k1 + 4*k2 + k3)/6
      x = x + h
      x_values.append(x)
      y_values.append(y)
    return x_values, y_values


def runge_kutta_4th(f, x0, y0, h, x_end):
    """
    Applies the 4th order Runge-Kutta method.

    Args:
        f (function): The function f(x, y) defining the ODE.
        x0 (float): The initial x value.
        y0 (float): The initial y value.
        h (float): The step size.
        x_end (float): The end x value.

    Returns:
        tuple: Lists of x values and approximated y values.
    """
    x = x0
    y = y0
    x_values = [x]
    y_values = [y]

    while x < x_end:
      k1 = h * f(x, y)
      k2 = h * f(x + h/2, y + k1/2)
      k3 = h * f(x + h/2, y + k2/2)
      k4 = h * f(x + h, y + k3)
      y = y + (k1 + 2*k2 + 2*k3 + k4)/6
      x = x + h
      x_values.append(x)
      y_values.append(y)
    return x_values, y_values
if __name__ == '__main__':
    def example_ode(x, y):
        return y - x**2 + 1

    x0 = 0.0
    y0 = 0.5
    h = 0.1
    x_end = 1.0

    x_vals, y_vals = runge_kutta_3rd(example_ode, x0, y0, h, x_end)
    print("3rd Order Runge-Kutta Results:")
    for x, y in zip(x_vals, y_vals):
        print(f"x = {x:.2f}, y = {y:.4f}")
    x_vals, y_vals = runge_kutta_4th(example_ode, x0, y0, h, x_end)
    print("4th Order Runge-Kutta Results:")
    for x, y in zip(x_vals, y_vals):
        print(f"x = {x:.2f}, y = {y:.4f}")