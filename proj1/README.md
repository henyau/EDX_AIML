# **8 piece sliding puzzle solver** 
**Use breadth-first search, depth-first search, and a-star search**


---

Solve using BFS, DFS, or A-Star
call from command line:
two arguments: solver type and board
solver types are 'bfs','dfs', 'ast'
board is comma separated string 6,1,8,4,0,2,7,3,5
outputs steps to solve puzzle in `output.txt` along with other information:

e.g. usage:
python3 driver_3.py ast 6,1,8,4,0,2,7,3,5

e.g. output:
path_to_goal: ['Down', 'Left', ...., 'Up']
cost_of_path: 70
nodes_expanded: 470
search_depth: 70
max_search_depth: 95
running_time: 0.0325319766998291



