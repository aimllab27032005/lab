import numpy as np  # Import NumPy for array manipulations
from queue import PriorityQueue  # Import PriorityQueue for A* search

# Define a class to represent a state in the 8-puzzle
class State:
    def __init__(self, state, parent):
        self.state = state  # The current 3x3 puzzle state
        self.parent = parent  # Parent state to reconstruct the solution path

    def __lt__(self, other):
        return False  # Required for PriorityQueue, but not used for comparison

# Define the Puzzle class that implements A* search
class Puzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state  # Store the initial state
        self.goal_state = goal_state  # Store the goal state

    # Method to print the puzzle state
    def print_state(self, state):
        print(state[:, :])  # Print the 3x3 matrix

    # Check if a given state is the goal state
    def is_goal(self, state):
        return np.array_equal(state, self.goal_state)

    # Generate possible moves by moving the empty tile (0)
    def get_possible_moves(self, state):
        possible_moves = []  # List to store new states
        zero_pos = np.where(state == 0)  # Find the empty tile's position
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        
        for direction in directions:
            new_pos = (zero_pos[0] + direction[0], zero_pos[1] + direction[1])
            
            # Ensure the new position is within bounds
            if 0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3:
                new_state = np.copy(state)  # Copy the current state
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]  # Swap tiles
                possible_moves.append(new_state)  # Add new state to possible moves
        
        return possible_moves  # Return all possible states

    # Heuristic function: counts the number of misplaced tiles
    def heuristic(self, state):
        return np.count_nonzero(state != self.goal_state)

    # A* search algorithm to solve the 8-puzzle
    def solve(self):
        queue = PriorityQueue()  # Create a priority queue
        initial_state = State(self.initial_state, None)  # Create the initial state object
        queue.put((0, initial_state))  # Add the initial state to the queue
        visited = set()  # Keep track of visited states

        while not queue.empty():
            priority, current_state = queue.get()  # Get state with lowest heuristic value
            
            if self.is_goal(current_state.state):
                return current_state  # Return the solution path if goal is reached
            
            for move in self.get_possible_moves(current_state.state):  # Generate next moves
                move_state = State(move, current_state)  # Create a new state object
                
                if str(move_state.state) not in visited:
                    visited.add(str(move_state.state))  # Mark state as visited
                    priority = self.heuristic(move_state.state)  # Calculate heuristic value
                    queue.put((priority, move_state))  # Add new state to the queue
        
        return None  # Return None if no solution is found

# Define the initial and goal states
initial_state = np.array([[1, 2, 3], [0, 8, 4], [7, 6, 5]])  # Example start configuration
goal_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])  # Goal configuration

puzzle = Puzzle(initial_state, goal_state)  # Create a puzzle object
solution = puzzle.solve()  # Attempt to solve the puzzle

# Trace back the solution path
move_count = -1  # Initialize move counter
if solution is not None:
    moves = []
    while solution is not None:  # Trace back to the initial state
        moves.append(solution.state)  # Store each move
        solution = solution.parent  # Move to the previous state
    
    for move in reversed(moves):  # Print moves in order
        move_count += 1
        puzzle.print_state(move)  # Print each step
    print("Number of moves:", move_count)  # Display total moves
else:
    print("No solution found.")  # If no solution, print a message
