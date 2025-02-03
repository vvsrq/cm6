import numpy as np

def forward_diff_1st_order(f, x, h):
    """
    Approximates the first derivative using Newton's forward difference.

    Args:
        f (function or array-like): The function to differentiate (or a list of function values).
        x (float or int): The point at which to evaluate the derivative (or index into a list).
        h (float): The step size.

    Returns:
        float: An approximation of f'(x).
    """

    if callable(f):  # If f is a function
        return (f(x + h) - f(x)) / h
    else: # If f is a list of values, assume x is an index
        if isinstance(x, int):
            return (f[x + 1] - f[x]) / h
        else:
            raise TypeError("If f is an array, x must be an integer.")

def forward_diff_2nd_order(f, x, h):
    """
    Approximates the second derivative using Newton's forward difference.

    Args:
        f (function or array-like): The function to differentiate (or a list of function values).
        x (float or int): The point at which to evaluate the derivative (or index into a list).
        h (float): The step size.

    Returns:
        float: An approximation of f''(x).
    """
    if callable(f):
       return (f(x + 2*h) - 2*f(x+h) + f(x)) / (h**2)
    else:
        if isinstance(x, int):
            return (f[x+2] - 2*f[x+1] + f[x]) / (h**2)
        else:
            raise TypeError("If f is an array, x must be an integer.")


if __name__ == '__main__':
    # Example 1: Using a Function
    import math
    def example_function(x):
        return math.sin(x)

    x = 1.0  # Point at which to estimate derivatives
    h = 0.1

    first_deriv = forward_diff_1st_order(example_function, x, h)
    second_deriv = forward_diff_2nd_order(example_function, x, h)

    print("Example 1: Using a Function")
    print(f"Approximated 1st derivative at x={x}: {first_deriv}")
    print(f"Approximated 2nd derivative at x={x}: {second_deriv}")
    print(f"Actual 1st derivative at x={x}: {math.cos(x)}")
    print(f"Actual 2nd derivative at x={x}: {-math.sin(x)}")


    # Example 2: Using a list of values
    values = [1.2, 2.1, 3.2, 4.5, 5.8, 7.1, 8.2]
    x = 2 # Index where we wish to evaluate the derivative
    h = 1 # h=1 because we are only looking at indices.

    first_deriv = forward_diff_1st_order(values, x, h)
    second_deriv = forward_diff_2nd_order(values, x, h)

    print("\nExample 2: Using a List of values")
    print(f"Approximated 1st derivative at index x={x}: {first_deriv}")
    print(f"Approximated 2nd derivative at index x={x}: {second_deriv}")
    # We cannot easily provide the "actual" derivative values here, because we don't know the function that creates these values, just the points.