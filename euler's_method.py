def euler_method(f, x0, y0, h, x_end):
    """
    Applies Euler's method to approximate the solution to dy/dx = f(x, y).

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
        y = y + h * f(x, y)
        x = x + h
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

if __name__ == '__main__':
    def example_ode(x, y):
        return y - x**2 +1

    x0 = 0.0
    y0 = 0.5
    h = 0.1
    x_end = 1.0

    x_vals, y_vals = euler_method(example_ode, x0, y0, h, x_end)
    print("Euler Method Results:")
    for x, y in zip(x_vals, y_vals):
        print(f"x = {x:.2f}, y = {y:.4f}")