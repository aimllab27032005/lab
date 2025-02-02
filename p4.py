import numpy as np

def hill_climbing(func, start, s_s=0.01, max_i=1000):
    c_p = start
    c_v = func(c_p)
    
    for i in range(max_i):
        n_p_p = c_p + s_s
        n_v_p = func(n_p_p)
        
        n_p_n = c_p - s_s
        n_v_n = func(n_p_n)
        
        if n_v_p > c_v and n_v_p >= n_v_n:
            c_p = n_p_p
            c_v = n_v_p
        elif n_v_n > c_v and n_v_n > n_v_p:
            c_p = n_p_n
            c_v = n_v_n
        else:
            break
    
    return c_p, c_v

# Get the function from the user
while True:
    func_str = input("\nEnter a function of x: ")
    try:
        # Test the function with a dummy value
        x = 0
        eval(func_str)
        break
    except Exception as e:
        print(f"Invalid function. Please try again. Error: {e}")

# Convert the string into a function
func = lambda x: eval(func_str)

# Get the starting point from the user
while True:
    start_str = input("\nEnter the starting value to begin the search: ")
    try:
        start = float(start_str)
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

maxima, max_value = hill_climbing(func, start)
print(f"The maxima is at x = {maxima}")
print(f"The maximum value obtained is {max_value}")