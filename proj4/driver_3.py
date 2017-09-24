import numpy as np
import queue
import sys

##print("This is the name of the script: ", sys.argv[0])
##print("Number of arguments: ", len(sys.argv))
##print("The arguments are: " , str(sys.argv))
##print(str(sys.argv[1]))

given_board = str(sys.argv[1])

solution_board = given_board

def iterateCell(R,C):
    return [r+c for r in R for c in C]

cols = '123456789'
rows = 'ABCDEFGHI'
cell_labels = iterateCell(rows,cols)

digits = cols

#list all groups of shared cells
sharedRCB_list = ([iterateCell(r, cols) for r in rows]+[iterateCell(rows, c) for c in cols]+
                  [iterateCell(r, c) for r in ('ABC','DEF','GHI') for c in ('123','456','789')])

sharedBox_list = ([iterateCell(r, c) for r in ('ABC','DEF','GHI') for c in ('123','456','789')])

sharedBox = dict((s, [u for u in sharedBox_list if s in u]) for s in cell_labels)
boxed = dict((s, set(sum(sharedBox[s],[]))-set([s])) for s in cell_labels)

#arrays of arrays elements of same row, column or box
sharedRCB = dict((s, [u for u in sharedRCB_list if s in u]) for s in cell_labels)

#array of elements in RCB, first flatten with sum(,[]) and remove self
peers = dict((s, set(sum(sharedRCB[s],[]))-set([s])) for s in cell_labels)

##neighbors = dict((s, set(sum(neighbor_list[s],[]))-set([s])) for s in cell_labels)
##print(peers["B1"])
def getNeighbors(Xa):

    pRow = chr(ord(Xa[0])-1)
    nRow = chr(ord(Xa[0])+1)
    
    pCol = chr(ord(Xa[1])-1)
    nCol = chr(ord(Xa[1])+1)
    
    neighborSet = []
    
    if pRow in rows:
        if pCol in cols:
            neighborSet.extend([pRow+pCol])
            neighborSet.extend([Xa[0]+pCol])            
            neighborSet.extend([pRow+Xa[1]])
        if nCol in cols:
            neighborSet.extend([pRow+nCol])
            neighborSet.extend([Xa[0]+nCol])            
            neighborSet.extend([pRow+Xa[1]])
    if nRow in rows:
        if pCol in cols:
            neighborSet.extend([nRow+pCol])
            neighborSet.extend([Xa[0]+pCol])            
            neighborSet.extend([nRow+Xa[1]])
        if nCol in cols:
            neighborSet.extend([nRow+nCol])
            neighborSet.extend([Xa[0]+nCol])            
            neighborSet.extend([nRow+Xa[1]])
    
    neighborSet = set((neighborSet))
    
    
    return neighborSet


def AC3_revise(board,Xi,Xj,D):
    revised = False
    
    for nummer in D[Xi].copy():
##        domainSetj =  
##        if nummer not in D[Xj]: #is this right? if there's no number in domain for j that
##        if len(set(D[Xj])-set([nummer])) == 0: # if there's no number available to use left,
        if len(D[Xj]) == 1 and nummer in D[Xj]: # if there's no number available to use left,
            D[Xi].remove(nummer)            
##            print('Required Numer: ', D[Xj])
            revised = True    
    return revised

def AC3(board,D):
##    queue of arcs, all arcs in CSP
    q = []
    
##    print([board[s] for s in cell_labels])

##    for every variable (things that are zero) make a arc pair and put in the queue
    for s in cell_labels:
        if board[s] == '0':
           q.extend([(s,s2) for s2 in peers[s]])

    while q != []: #len(q) != 0:
         [Xi, Xj] = q.pop(0)
##         print([Xi, Xj])
         
         if AC3_revise(board,Xi,Xj,D):# and AC3_revise(board,Xj,Xi,D):
             if len(D[Xi]) == 0:
                 return False
##only thing to do is neighbor....
##             for Xk in peers[Xi]-set([Xj]):
##                 print([Xk, Xi])
##                 q.extend([[Xk, Xi]])
             Neighbors = getNeighbors(Xi)
             for Xk in peers[Xi]-set([Xj]):
##                 print([Xk, Xi])
                 q.extend([[Xk, Xi]])

    return True


def createGrid(grid):
    chars = [c for c in grid if c in digits or '0']
    return dict(zip(cell_labels, chars))

board = createGrid(given_board)

##initially the domain is whatever 123456789 unless already zero
##should be a dictionary too
domain = dict((s, set(digits) if board[s] in '0' else {board[s]}) for s in cell_labels)

##print(domain)

##print(createGrid(board))    

AC3_bool = AC3(board,domain)
##print(AC3_bool)

##print(domain)
##convert domain to a board
AC3_sln  = ''
for r in rows:
    for c in cols:
        for e in domain[r+c]:
            AC3_sln = AC3_sln+(e)
##remove brackets and ''
if len(AC3_sln)!= 81:
    AC3_bool = False
else:
    AC3_bool = True
##print(AC3_sln)
##print(AC3_bool)

##IF AC3 false then do BTS... maybe need to fix AC3 first. there is a bug with the domain. sometimes it is [] sometimes it is {}
def is_consistent(assignmentC,ind, val):
    for s in peers[ind]:
        if val == assignmentC[s]:
            return False
    return True

def backtrack(assignment,D):
    complete = True
    var = []
    for s in cell_labels:
        if assignment[s] == '0':            
            complete = False
##            var.append(s) #set of unassigned variables
            var = s
    if complete ==  True:
        return assignment
    
    index = var 
##    for index in var:
    for value in D[index].copy():        
        if is_consistent(assignment,index, value):
            assignment[index] = value
            result = backtrack(assignment,D) #need to alter D?
            if result != False:                
                return result
##            assignment[index] = '0'
            assignment[index] = '0'
##        D[index].remove(value)
##        D[index].remove(value)
        
        
 ##   print('when am i here')    
    return False

    
def backtracking_seach(board,D):
    return backtrack(board,D)
##domain = dict((s, set(digits) if board[s] in '0' else {board[s]}) for s in cell_labels)

BTS_slnGrid = backtracking_seach(board,domain)



BTS_sln  = ''
for r in rows:
    for c in cols:
        for e in BTS_slnGrid[r+c]:
            BTS_sln = BTS_sln+(e)
##print(BTS_sln)


write_target = open("output.txt",'w')
if (AC3_bool):
    write_target.write(str(AC3_sln)+' AC3')
else:
    write_target.write(str(BTS_sln)+' BTS')
    
write_target.close()



