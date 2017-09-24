#! python
import sys
#import resource
import time

#print("The name of script:", sys.argv[0])
#print("Number of argumnts:", len(sys.argv))
#print("Arguments are:", str(sys.argv))
def print_board (state):
    print(state[0], state[1], state[2])
    print(state[3], state[4], state[5])
    print(state[6], state[7], state[8])
    print()

#this moves the 0 tile up
def move_up (state):
    #find the 0 tile
    ind0 = state.index(0)
    if ind0 not in [0, 1, 2]:
        #swap ind0 with ind0-3
        tempVal0 = state[ind0]
        state[ind0] = state[ind0-3]
        state[ind0-3] = tempVal0
        
def move_down (state):
    #find the 0 tile
    ind0 = state.index(0)
    if ind0 not in [6, 7, 8]:
        #swap ind0 with ind0-3
        tempVal0 = state[ind0]
        state[ind0] = state[ind0+3]
        state[ind0+3] = tempVal0
        
        
def move_left (state):
    #find the 0 tile
    ind0 = state.index(0)
    if ind0 not in [0, 3, 6]:
        #swap ind0 with ind0-3
        tempVal0 = state[ind0]
        state[ind0] = state[ind0-1]
        state[ind0-1] = tempVal0
        
def move_right (state):
    #find the 0 tile
    ind0 = state.index(0)
    if ind0 not in [2, 5, 8]:
        #swap ind0 with ind0-3
        tempVal0 = state[ind0]
        state[ind0] = state[ind0+1]
        state[ind0+1] = tempVal0        

def check_goal(state):
    if (state==[0,1,2,3,4,5,6,7,8]):
        return True
    else:
        return False


#order of moves UDLR    
    
def main():
    board_state = [1,2,5,3,4,0,6,7,8]
    print_board(board_state)
    move_up(board_state)
    print_board(board_state)
    move_left(board_state)
    print_board(board_state)
    move_left(board_state)
    print_board(board_state)
##    move_down(board_state)
##    print_board(board_state)
##    move_right(board_state)
##    print_board(board_state)
    
    print(check_goal(board_state))
    
if __name__ == "__main__":
    main()

#print("Resources used: ", resource.ru_maxrss(resource.RUSAGE_SELF))


#can do all this output stuff later, make sure we are doing stuff right first
#outfilename = "out.txt"
#writetarget = open(outfilename, 'w')