import heapq
import numpy as np

# Get new location based on current location and direction
def move(loc, dir):
    # dir=0: up; dir=1: right; dir=2: down; dir=3: left;
    # directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # dir=4: wait;
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

# Calculate all solution paths' travelled cost
def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst

# Calculate a corresponding heuristic values look-up table rooted from the goal to every node in my_map
def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        # start from current location to calculate costs of peripheral 4-directions' nodes correponding to possible child, i.e. child_loc
        #********* we need to calculate the cost for our topology map based on graph search*********#
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            # check if the child_loc (possible new child location) is not legitimate, i.e. obstacle in the map
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            # if the child denoted by child_loc is already in the existed dictionary, i.e. 'closed_list', which stores the unique child_loc and corresponding cost
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                # if child_loc corresponding cost 'child_cost' is less than 'existing_node['cost']', substitute the existed node's cost
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            # if the child denoted by child_loc is not in the existed dictionary 'closed_list', get it in
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    constraint_table = dict()
    for constraint in constraints:

        # Task 4.1 add key positive

        if not ("positive" in constraint.keys()):
            constraint['positive'] = False

        if constraint['agent'] == agent:
            if constraint['timestep'] in constraint_table.keys():
                # this will be used for more constraints related to continuous space-time conflict relationship
                constraint_table[constraint['timestep']].append(constraint['loc'])
                # append new value to the existed key in the dictionary corresponding value, which is a list of location
            else:
                constraint_table[constraint['timestep']] = [constraint['loc']]

    return constraint_table

# Get the agent location related to discrete time-step
def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location

# Reverse from the goal node to its parent's node (in the generated path) to get the path
def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path

def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    
    # if len(constraint_table) < next_time and len(constraint_table)!=0:
    #     next_time = len(constraint_table)

    if next_time in constraint_table:
        for loc in constraint_table[next_time]:
            if len(loc) == 1:
                if loc == [next_loc]:
                    return True
            elif loc == [curr_loc, next_loc]:
                return True

    return False

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))

def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

# compare the heuristic value btw nodes
def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

# space time a star
def space_time_a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    constraint_table = build_constraint_table(constraints, agent)

    # initialize the map boundaries
    max_map_x = len(my_map)
    max_map_y = len(my_map[0])

    open_list = []  # priority queue for states search
    closed_list = dict()    # Initialize the dictionary of explored states
    earliest_goal_timestep = 0
    if len(constraint_table.keys()) > 0:
        # print(constraint_table)
        earliest_goal_timestep = max(constraint_table.keys()) # = 3 for task 2.3 and task 2.4

    h_value = h_values[start_loc]
    # Task 1.1 Added the timestep of root to be 0
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep':0}
    push_node(open_list, root)
    # Task 1.1 Made closed list a tuple of (cell, timestep) 
    closed_list[(root['loc'], root['timestep'])] = root

    while len(open_list) > 0:
        # record the location and cost of on-going explored states with lower cost
        opt_open_set_loc = []
        opt_open_set_cost = []

        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints

        # Task 2.4 Max timestep allowed is 10

        # if curr['timestep']>10:
        #     print("No solutions in time")
        #     return None

        # Upperbound searching space by limiting maximum state's time_steps
        if curr['timestep']>max(max_map_x,max_map_y)**3:
            print("No solutions in time")
            return None

        # Check if the current state is the goal state
        if curr['loc'] == goal_loc and curr['timestep'] >= earliest_goal_timestep:
            # print(constraint_table.keys())
            return get_path(curr)

        # Add the current state to the set of explored states
        closed_list[(curr['loc'], curr['timestep'])] = curr

        # record current open_list's "location:cost"
        for n in range(len(open_list)):
            opt_open_set_loc.append(open_list[n][2])
            opt_open_set_cost.append(open_list[n][0])
        # print("current open_list location {}".format(opt_open_set_loc))
        # print("corresponding current open_list cost {}".format(opt_open_set_cost))

        for dir in range(5):
            child_loc = move(curr['loc'], dir)

            # check the space location is out of the range or not
            if child_loc[0] < 0 or child_loc[1] < 0 or child_loc[0] >= max_map_x or child_loc[1] >= max_map_y:
                continue

            if my_map[child_loc[0]][child_loc[1]] or is_constrained(curr['loc'],child_loc,curr['timestep']+1,constraint_table):
                continue

            # Task 1.1 Timestep of each child is 1 more than its parents
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep': curr['timestep'] + 1}

            if (child['loc'], child['timestep']) in closed_list:
                continue

            if child['loc'] in opt_open_set_loc:
                existing_node_idxes = [i for i in range(len(opt_open_set_loc)) if
                                       opt_open_set_loc[i] == child['loc']]
                nparray_open_set_cost = np.array(opt_open_set_cost)
                if child['g_val'] + child['h_val'] < min(nparray_open_set_cost[existing_node_idxes]):
                    push_node(open_list, child)
            else:
                push_node(open_list, child)
                # print('push_open_list')

    print("low_level: no solution")
    return None  # Failed to find solutions