def forward_diff_1st_order(f, x, h):
    """
    Approximates the first derivative using Newton's forward difference.

    Args:
        f (function): The function to differentiate.
        x (float): The point at which to evaluate the derivative.
        h (float): The step size.

    Returns:
        float: An approximation of f'(x).
    """
    return (f(x + h) - f(x)) / h


def forward_diff_2nd_order(f, x, h):
    """
    Approximates the second derivative using Newton's forward difference.

    Args:
        f (function): The function to differentiate.
        x (float): The point at which to evaluate the derivative.
        h (float): The step size.

    Returns:
        float: An approximation of f''(x).
    """
    return (f(x + 2*h) - 2*f(x+h) + f(x)) / (h**2)


if __name__ == '__main__':
    import math
    def example_function(x):
        return math.sin(x)

    x = 1.0
    h = 0.1

    first_deriv = forward_diff_1st_order(example_function, x, h)
    second_deriv = forward_diff_2nd_order(example_function, x, h)

    print(f"Approximated 1st derivative at x={x}: {first_deriv}")
    print(f"Approximated 2nd derivative at x={x}: {second_deriv}")
    #Actual values are:
    print(f"Actual 1st derivative at x={x}: {math.cos(x)}")
    print(f"Actual 2nd derivative at x={x}: {-math.sin(x)}")