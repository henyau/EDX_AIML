"""
Implements an adversarial AI agent to solve 4096 Puzzle

Use weights to create a snake like structure to merge efficiently

@author: Henry Yau
"""
from random import randint
from BaseAI import BaseAI
import math
 

 
class PlayerAI(BaseAI):
    """derive from BaseAI class, implements getMove() called in the GameManager"""
    def getNewTileValue(self):
        if randint(0,99) < 100 * 0.9:
            return 2
        else:
            return 4;
    
    def Heuristic(self, grid):
        """Cost function. try to keep largest tile in top left and others in 
        decreasing order"""
        emptyTiles = len([i for i, x in enumerate(grid.map) if x == 0])
        maxTile = max(grid.map)
        MergeBonus = 0
        OrderBonus = 0
        Ord = 0
        penalty = 0
        ##    weights = [10,8,7,6.5,.5,.7,1,3,-.5,-1.5,-1.8,-2,-3.8,-3.7,-3.5,-3]
        weights = [65536,32768,16384,8192,512,1024,2048,4096,256,128,64,32,2,4,8,16]
        if maxTile == grid.map[0][0]:
            Ord += (math.log(grid.map[0][0])/math.log(2))*weights[0]
        for i in range(0,3):
            for j in range(0,3):
                if grid.map[i][j] >= 8:
                    Ord += weights[i+(j)*4]*(math.log(grid.map[i][j])/math.log(2))
        ##        if i < 4 and grid[i] == 0 :
        ##            Ord -=weights[i]*(math.log(maxTile)/math.log(2))
                    
        cost2 = 0
        averageTile = 0
        maxTile = grid.getMaxTile();
        
        for x in range(grid.size):
             for y in range(grid.size):
                cost2 += grid.map[x][y]
         
                
        costLocation = 0
        if(grid.map[0][0] == maxTile):
            costLocation = 200;
        elif(grid.map[0][3] == maxTile):
            costLocation = 200;
        elif(grid.map[3][3] == maxTile):
            costLocation = 200;
        elif(grid.map[3][0] == maxTile):
            costLocation = 200;
                
##        return weight1 * averageTileNumber + weight2 * MedianTileNumber
        #try to keep max tile in the corner 0,0
       #give bonus if max in corner                
                
        availableCells = len(grid.getAvailableCells())
        cellNum = 17-availableCells
                
        #cost = maxTile + availableCells*100+10*cost2/cellNum+costLocation;        
        #cost = maxTile*2+availableCells*20+5*cost2/cellNum+costLocation;        
        #cost = availableCells*20+costLocation+math.log(cost2);        
    
        cost = maxTile*5+availableCells*20+50*cost2/cellNum+costLocation;   
                    
        return Ord/(16-emptyTiles) + cost

##    def Heuristic(self, grid):
##        #for now just pick the largest tile
####        largest combine value
##        cost2 = 0
##        averageTile = 0
##        maxTile = grid.getMaxTile();
##        
##        for x in range(grid.size):
##             for y in range(grid.size):
##                cost2 += grid.map[x][y]
##         
##                
##        costLocation = 0
##        if(grid.map[0][0] == maxTile):
##            costLocation = 200;
##        elif(grid.map[0][3] == maxTile):
##            costLocation = 200;
##        elif(grid.map[3][3] == maxTile):
##            costLocation = 200;
##        elif(grid.map[3][0] == maxTile):
##            costLocation = 200;
##                
####        return weight1 * averageTileNumber + weight2 * MedianTileNumber
##        #try to keep max tile in the corner 0,0
##       #give bonus if max in corner                
##                
##        availableCells = len(grid.getAvailableCells())
##        cellNum = 17-availableCells
##                
##        #cost = maxTile + availableCells*100+10*cost2/cellNum+costLocation;        
##        #cost = maxTile*2+availableCells*20+5*cost2/cellNum+costLocation;        
##        #cost = availableCells*20+costLocation+math.log(cost2);        
##    
##        cost = maxTile*5+availableCells*20+50*cost2/cellNum+costLocation;        
##        
##        return cost
    
    def Terminate(self, grid, depth): 
        """Terminate using depth test"""
##        cells = grid.getAvailableCells()
    ##        moves = grid.getAvailableMoves()
    ##        if len(moves)==0:
##        if len(cells) < 10:
        if depth>4:
            return True
        else:
            return False
    
    def Maximize(self, grid, alpha, beta,depth):
        """Attempt to maximize score"""
        
        if(self.Terminate(grid,depth)):
            return (None, self.Heuristic(grid))
        (maxChild, maxUtility) = (None, -float('inf'))
        
        #iterate through the possible moves
        moves = grid.getAvailableMoves()
        
        for i in range (0, len(moves) - 1):
            child = grid.clone()
            child.move(moves[i]) #need to copy?
            (tmp, utility) = self.Minimize(child,alpha,beta,depth+1)
            
            if utility > maxUtility:
                (maxChild, maxUtility) = (moves[i], utility)
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility
##        print('depth:', depth) 
        return (maxChild, maxUtility)
    
    
    def Minimize(self,grid,alpha, beta, depth):
        """Attempt to minimize score"""
##        if self.Terminate(grid):
##            return (None, self.Heuristic(grid))
        
        (minChild, minUtility) = (None, float('inf'))
        cells = grid.getAvailableCells()
        
                
        for i in range (0, len(cells) - 1):
            child = grid.clone()
            child.insertTile(cells[i], self.getNewTileValue()) #could be 4?
            (tmp, utility) = self.Maximize(child,alpha,beta,depth+1) 
            
            
            if utility < minUtility:
                (minChild, minUtility) = (child, utility)
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility
        
        return (minChild, minUtility)
        
        
    
    def Decision(self, grid1):
  
        return self.Maximize(grid1,-float('inf'),float('inf'),1)[0]

    def getMove(self, grid):
        """Return optimal move """
        moves = grid.getAvailableMoves()
        
        print(moves)
        
        gridCopy = grid.clone() #get a copy of the board so we know what we're looking at.
        
        
        moveMax = self.Decision(gridCopy)
        
        #print('sausage:', moveMax)
        if moveMax != None:
            return moveMax
        else:
            return moves[randint(0, len(moves) - 1)] if moves else None
            

