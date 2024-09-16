import numpy as np
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.reward = 0

def distance(node1, node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def is_valid(node, grid):
    return 0 <= node.x < grid.shape[0] and 0 <= node.y < grid.shape[1] and grid[node.x, node.y] != -1

def nearest_node(nodes, point):
    return min(nodes, key=lambda node: distance(node, point))

def new_node(from_node, to_node, step_size):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    dist = np.sqrt(dx**2 + dy**2)
    if dist <= step_size:
        return Node(to_node.x, to_node.y)
    else:
        theta = np.arctan2(dy, dx)
        return Node(int(from_node.x + step_size * np.cos(theta)),
                    int(from_node.y + step_size * np.sin(theta)))

def rrt_grid_world(grid, start, goal, max_iterations=1000, step_size=1):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    nodes = [start_node]
    
    for _ in range(max_iterations):
        if random.random() < 0.1:  # 10% chance to select goal as the random point
            random_point = goal_node
        else:
            random_point = Node(random.randint(0, grid.shape[0]-1),
                                random.randint(0, grid.shape[1]-1))
        
        nearest = nearest_node(nodes, random_point)
        new = new_node(nearest, random_point, step_size)
        
        if is_valid(new, grid):
            new.parent = nearest
            new.reward = nearest.reward + max(0, grid[new.x, new.y])
            nodes.append(new)
            
            if distance(new, goal_node) <= step_size and is_valid(goal_node, grid):
                goal_node.parent = new
                goal_node.reward = new.reward + max(0, grid[goal_node.x, goal_node.y])
                return goal_node
    
    return nearest_node(nodes, goal_node)

def backtrack_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

# Example usage
grid = np.array([
    [0, 0, 0, 1, 0],
    [0, -1, 0, 2, 0],
    [0, -1, 0, -1, 0],
    [0, 1, 0, 3, 0],
    [0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)

final_node = rrt_grid_world(grid, start, goal)
path = backtrack_path(final_node)

print(f"Path found: {path}")
print(f"Total reward: {final_node.reward}")