import sys
from collections import deque
from queue import PriorityQueue
##queue
##nlist = [1, 2, 3, 4, 5, 0]
##print(nlist)
##n = nlist.pop(0)
##print(nlist)
##print(n)
##nlist.append(n)
##print(nlist)
##
####stack
##nlist = [1, 2, 3, 4, 5, 0]
##print(nlist)
##n = nlist.pop(-1)
##print(nlist)
##nlist.append(n)
##print(nlist)
##
####nlist to a number
##i = 1
##hash = 0
##for elem in nlist:
##    hash +=i*elem
##    i*=10
##print(hash)
##
####
##
##frontierHash = deque([0])
##frontierHash.append(1)
##frontierHash.append(2)
##print(frontierHash)
##frontierHash.pop()
##print(frontierHash)
def GetDist(index, value):
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
       
test_state = [6,1,8,4,0,2,7,3,5]
print(ManhattanDist(test_state))

frontier = PriorityQueue()
frontier.put((ManhattanDist(test_state), test_state))
test_state = [8,6,4,2,1,3,5,7,0]
frontier.put((ManhattanDist(test_state), test_state))


print(frontier.queue[1][0])
    