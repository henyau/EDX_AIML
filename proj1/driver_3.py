#! python
import itertools
from collections import deque
from queue import PriorityQueue
from sys import stdout


import sys
#import resource
import time


def print_board (state):
    print(state[0], state[1], state[2])
    print(state[3], state[4], state[5])
    print(state[6], state[7], state[8])
    print()

#this moves the 0 tile up
def move_up (state):
    #find the 0 tile
    out_state = state[:] # holy shit if don't do copy, then it is just a reference....
    ind0 = out_state.index(0)
    if ind0 not in [0, 1, 2]:
        #swap ind0 with ind0-3
        tempVal0 = out_state[ind0]
        out_state[ind0] = out_state[ind0-3]
        out_state[ind0-3] = tempVal0
        return out_state
    else:
        return None
        
def move_down (state):
    #find the 0 tile
    out_state = state[:]
    ind0 = out_state.index(0)
    
    if ind0 not in [6, 7, 8]:
        #swap ind0 with ind0-3
        tempVal0 = out_state[ind0]
        out_state[ind0] = out_state[ind0+3]
        out_state[ind0+3] = tempVal0
        return out_state
    else:
        return None
        
        
def move_left (state):
    #find the 0 tile
    out_state = state[:]
    ind0 = out_state.index(0)
    if ind0 not in [0, 3, 6]:
        #swap ind0 with ind0-3
        tempVal0 = out_state[ind0]
        out_state[ind0] = out_state[ind0-1]
        out_state[ind0-1] = tempVal0
        return out_state
    else:
        return None
        
def move_right (state):
    #find the 0 tile
    out_state = state[:]
    ind0 = out_state.index(0)
    if ind0 not in [2, 5, 8]:
        #swap ind0 with ind0-3
        tempVal0 = out_state[ind0]
        out_state[ind0] = out_state[ind0+1]
        out_state[ind0+1] = tempVal0
        return out_state
    else:
        return None

def check_goal(state):
    if (state==[0,1,2,3,4,5,6,7,8]):
        return True
    else:
        return False

class Node:
    def __init__(self, parent, op, state, depth, cost):
        self.parent = parent
        self.op = op
        self.state = state
        self.depth = depth
        self.cost = cost
    def __lt__ (self, other):
##        return ManhattanDist(self.state)<ManhattanDist(other.state)
        return self.cost<other.cost
    def stateHash(self):
        i = 1
        hash = 0
        for elem in self.state:
            hash +=i*elem
            i*=10
        return hash
        
def BFS(state):
    path_to_goal = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = time.time()
    max_ram_usage = 0
    
    #if check_goal(state):
    #fill out a tree while checking the goal
    frontier = deque([Node( None, None, state, 0, 0)])
    #frontierHash = set()
    #Node( parent, op, state, depth, cost)
    #frontier.append(Node( None, None, state, 0, 0)) #this is the frontier
    explored = set()
    node = []
    i = 0

    while len(frontier) !=0: #check_goal(state) == False:
    
        node = frontier.popleft()
        explored.add(node.stateHash())
        ##print_board(node.state);
        if check_goal(node.state):
            print('Found you sucker')
            temp = node
            search_depth = temp.depth
            while True:
                
                if temp.parent == None:
                    break
                path_to_goal.append(temp.op) ##backwards, reverse this
                
                cost_of_path+=temp.cost                            
                    
                temp = temp.parent
            break
            
        ##need a global cost counter.
        neighbor_nodes = []
        tempState1 = move_up(node.state)
        if tempState1!=None:
            neighbor_nodes.append(Node(node,"Up", tempState1, node.depth+1, 1))
             
        tempState = move_down(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Down", tempState, node.depth+1, 1))
             
        tempState = move_left(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Left", tempState, node.depth+1, 1))
            
        tempState = move_right(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Right", tempState, node.depth+1, 1))
            
        
        #frontierAndExplorer = frontier+explored
##        frontierStates = set()
##        for elem in frontier:
##            frontierStates.add(elem.stateHash())
         
        for element in neighbor_nodes:
            if element.stateHash() not in explored:# and element not in frontier:
##                if element.stateHash() not in frontierStates:# and element not in frontier:
                    
                    frontier.append( element )
                    if element.depth>max_search_depth:
                        max_search_depth = element.depth
        i = i+1
      
        nodes_expanded = i
##        if i == 0:
##        print(len(explored))
        nodes_expanded=len(explored)

    running_time = time.time()-running_time
    path_to_goal.reverse()
    print('path_to_goal: ',path_to_goal)
    print('cost_of_path: ',cost_of_path)
    print('nodes_expanded: ',nodes_expanded)
    print('search_depth: ',search_depth)
    
    print('max_search_depth: ',max_search_depth)
    print('running_time: ',running_time)
    print('max_ram_usage: ',max_ram_usage)
    
##    output to txt
    write_target = open("output.txt",'w')
    write_target.write('path_to_goal: '+str(path_to_goal)+'\n')
    
    write_target.write('cost_of_path: '+str(cost_of_path)+'\n')
    write_target.write('nodes_expanded: '+str(nodes_expanded)+'\n')
    write_target.write('search_depth: '+str(search_depth)+'\n')
    
    write_target.write('max_search_depth: '+str(max_search_depth)+'\n')
    write_target.write('running_time: '+str(running_time)+'\n')
    write_target.write('max_ram_usage: '+str(max_ram_usage))
    
    
    write_target.close()

def DFS(state):
    path_to_goal = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = time.time()
    max_ram_usage = 0
    
    #if check_goal(state):
    #fill out a tree while checking the goal
    frontier = deque([Node( None, None, state, 0, 0)])
    frontierHash = set()
    frontierHash.add(frontier[0].stateHash())
    #frontierHash = set()
    #Node( parent, op, state, depth, cost)
    #frontier.append(Node( None, None, state, 0, 0)) #this is the frontier
    explored = set()
    node = []
    i = 0

    while len(frontier) !=0: #check_goal(state) == False:
    
        node = frontier.pop()
        frontierHash.discard(node.stateHash())
        
        explored.add(node.stateHash())
        ##print_board(node.state);
        if check_goal(node.state):
            print('Found you sucker')
            temp = node
            search_depth = temp.depth
            while True:
                
                if temp.parent == None:
                    break
                path_to_goal.append(temp.op) ##backwards, reverse this
                
                cost_of_path+=temp.cost                            
                    
                temp = temp.parent
                
            break
            
        ##need a global cost counter.
        neighbor_nodes = []
        tempState1 = move_up(node.state)
        if tempState1!=None:
            neighbor_nodes.append(Node(node,"Up", tempState1, node.depth+1, 1))
             
        tempState = move_down(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Down", tempState, node.depth+1, 1))
             
        tempState = move_left(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Left", tempState, node.depth+1, 1))
            
        tempState = move_right(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Right", tempState, node.depth+1, 1))
            
         
        neighbor_nodes.reverse() #the for loop append appears to do things backwards??
         
        for element in neighbor_nodes:
            if element.stateHash() not in explored:# and element not in frontier:
                if element.stateHash() not in frontierHash:# and element not in frontier:
                    
                    frontier.append( element )
                    frontierHash.add(element.stateHash())
                    if element.depth>max_search_depth:
                        max_search_depth = element.depth
        i = i+1
      
        nodes_expanded = i
        if i%1000 == 0:
            print(i)
            
        nodes_expanded=len(explored)

    running_time = time.time()-running_time
    path_to_goal.reverse()
    print('path_to_goal: ',path_to_goal)
    print('cost_of_path: ',cost_of_path)
    print('nodes_expanded: ',nodes_expanded)
    print('search_depth: ',search_depth)
    
    print('max_search_depth: ',max_search_depth)
    print('running_time: ',running_time)
    print('max_ram_usage: ',max_ram_usage)
    
    
##    output to txt
    write_target = open("output.txt",'w')
    write_target.write('path_to_goal: '+str(path_to_goal)+'\n')
    
    write_target.write('cost_of_path: '+str(cost_of_path)+'\n')
    write_target.write('nodes_expanded: '+str(nodes_expanded)+'\n')
    write_target.write('search_depth: '+str(search_depth)+'\n')
    
    write_target.write('max_search_depth: '+str(max_search_depth)+'\n')
    write_target.write('running_time: '+str(running_time)+'\n')
    write_target.write('max_ram_usage: '+str(max_ram_usage))
    
    
    write_target.close()

def GetDist(index, value):
    if value == 0:
        return 0
    i = 0
    j = 0
    i1 = 0
    j1 = 0
    if index == 0:
        i = 0
        j = 0
    elif index == 1:
        i = 0
        j = 1
    elif index == 2:
        i = 0
        j = 2
    elif index == 3:
        i = 1
        j = 0
    elif index == 4:
        i = 1
        j = 1
    elif index == 5:
        i = 1
        j = 2
    elif index == 6:
        i = 2
        j = 0
    elif index == 7:
        i = 2
        j = 1
    elif index == 8:
        i = 2
        j = 2
    if value == 0:
        i1 = 0
        j1 = 0
    elif value == 1:
        i1 = 0
        j1 = 1
    elif value == 2:
        i1 == 0
        j1 == 2
    elif value == 3:
        i1 = 1
        j1 = 0
    elif value == 4:
        i1 = 1
        j1 = 1
    elif value == 5:
        i1 = 1
        j1 = 2
    elif value == 6:
        i1 = 2
        j1 = 0
    elif value == 7:
        i1 = 2
        j1 = 1
    elif value == 8:
        i1 = 2
        j1 = 2

    return abs(i1-i)+abs(j1-j)

        
def ManhattanDist(state):
    totcost = 0
    for i in range(0, 9):
        totcost += GetDist(state[i],i)        
    return totcost

def AST(state):
    path_to_goal = []
    cost_of_path = 0
    nodes_expanded = 0
    search_depth = 0
    max_search_depth = 0
    running_time = time.time()
    max_ram_usage = 0
    
    #if check_goal(state):
    #fill out a tree while checking the goal
    frontier = PriorityQueue()
    frontier.put((ManhattanDist(state), Node( None, None, state, 0, 0)))
     
    tempState = Node( None, None, state, 0, 0)
    frontierHash = set()
    frontierHash.add(tempState.stateHash())
    #frontierHash = set()
    #Node( parent, op, state, depth, cost)
    #frontier.append(Node( None, None, state, 0, 0)) #this is the frontier
    explored = set()
    node = []
    i = 0

    #while len(frontier) !=0: #check_goal(state) == False:
    while ~frontier.empty():
        print(i)
        node = frontier.get()[1]
        frontierHash.discard(node.stateHash())
        
        explored.add(node.stateHash())
        ##print_board(node.state);
        if check_goal(node.state):
##            print('Found you sucker')
            temp = node
            search_depth = temp.depth
            while True:
                
                if temp.parent == None:
                    break
                path_to_goal.append(temp.op) ##backwards, reverse this
                
                cost_of_path+=temp.cost                            
                    
                temp = temp.parent
                
            break
            
        ##need a global cost counter.
        neighbor_nodes = []
        tempState1 = move_up(node.state)
        if tempState1!=None:
            neighbor_nodes.append(Node(node,"Up", tempState1, node.depth+1, 1))
             
        tempState = move_down(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Down", tempState, node.depth+1, 1))
             
        tempState = move_left(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Left", tempState, node.depth+1, 1))
            
        tempState = move_right(node.state)
        if tempState!=None:
            neighbor_nodes.append(Node(node,"Right", tempState, node.depth+1, 1))
            
         
   ##     neighbor_nodes.reverse() #the for loop append appears to do things backwards??
         
        for element in neighbor_nodes:
            if element.stateHash() not in explored:# and element not in frontier:
##                if element.stateHash() not in frontierHash:# and element not in frontier:
##                    print("add some shits")
                   
                    frontier.put(( ManhattanDist(element.state), element ))
                    frontierHash.add(element.stateHash())
                    if element.depth>max_search_depth:
                        max_search_depth = element.depth
##                    else:
##                        frontier.
                        
##                        for jj in range(0, frontier.qsize()):
##                            frontier.queue[jj][0] =0.5*frontier.queue[jj][0]
                    #decrease key???
        i = i+1
      
##        print(frontier.queue)
        nodes_expanded = i
        if i%1000 == 0:
            print(i)
            
        nodes_expanded=len(explored)

    running_time = time.time()-running_time
    path_to_goal.reverse()
    print('path_to_goal: ',path_to_goal)
    print('cost_of_path: ',cost_of_path)
    print('nodes_expanded: ',nodes_expanded)
    print('search_depth: ',search_depth)
    
    print('max_search_depth: ',max_search_depth)
    print('running_time: ',running_time)
    print('max_ram_usage: ',max_ram_usage)
    
    
##    output to txt
    write_target = open("output.txt",'w')
    write_target.write('path_to_goal: '+str(path_to_goal)+'\n')
    
    write_target.write('cost_of_path: '+str(cost_of_path)+'\n')
    write_target.write('nodes_expanded: '+str(nodes_expanded)+'\n')
    write_target.write('search_depth: '+str(search_depth)+'\n')
    
    write_target.write('max_search_depth: '+str(max_search_depth)+'\n')
    write_target.write('running_time: '+str(running_time)+'\n')
    write_target.write('max_ram_usage: '+str(max_ram_usage))
    
    
    write_target.close()
#order of moves UDLR    
    
def main():

##    solverType = str(sys.argv[1])
##    board_state = []
##    board_string = sys.argv[2].split(',')
##
##    for ichar in board_string:
##        board_state.append(int(ichar))
##
##    print_board(board_state)
##
##    if solverType == 'bfs':
##        BFS(board_state)
##    elif solverType == 'dfs':
##        DFS(board_state)
##    elif solverType == 'ast':
##        AST(board_state)
##        
        
    board_state = [6,1,8,4,0,2,7,3,5]
    #board_state = [8,6,4,2,1,3,5,7,0]
##    board_state = [1,2,5,3,4,0,6,7,8]
    AST(board_state)  
    
if __name__ == "__main__":
    main()
    
#print("Resources used: ", resource.ru_maxrss(resource.RUSAGE_SELF)) #not available in windows and I have no room on VM


#can do all this output stuff later, make sure we are doing stuff right first
#outfilename = "out.txt"
#writetarget = open(outfilename, 'w')