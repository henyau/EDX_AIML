from random import randint
from BaseAI import BaseAI
import math
 

 
class PlayerAI(BaseAI):
    def Heuristic(self, grid):
        #for now just pick the largest tile
        return grid.getMaxTile()
    
    def Terminate(self, grid, depth): #test depth (many moves cells free
##        cells = grid.getAvailableCells()
    ##        moves = grid.getAvailableMoves()
    ##        if len(moves)==0:
##        if len(cells) < 10:
        if depth>:
            return True
        else:
            return False
    
    def Maximize(self, grid, depth):
##        print(grid)
##        print(grid.getAvailableMoves())
        if(self.Terminate(grid,depth)):
            return (None, self.Heuristic(grid))
        (maxChild, maxUtility) = (None, -float('inf'))
        
        #iterate through the possible moves
        moves = grid.getAvailableMoves()
        for i in range (0, len(moves) - 1):
            child = grid.clone()
            child.move(moves[i]) #need to copy?
            (tmp, utility) = self.Minimize(child,depth+1)
            
            if utility > maxUtility:
                (maxChild, maxUtility) = (moves[i], utility)
        
        return (maxChild, maxUtility)
    
    
    def Minimize(self,grid,depth):
##        if self.Terminate(grid):
##            return (None, self.Heuristic(grid))
        
        (minChild, minUtility) = (None, float('inf'))
        cells = grid.getAvailableCells()
        
                
        for i in range (0, len(cells) - 1):
            child = grid.clone()
            child.insertTile(cells[i], 2) #could be 4?
            (tmp, utility) = self.Maximize(child,depth+1) 
            
            
            if utility < minUtility:
                (minChild, minUtility) = (child, utility)
        
        return (minChild, minUtility)
        
        
    
    def Decision(self, grid1):
  
        return self.Maximize(grid1,1)[0]

    def getMove(self, grid):
        moves = grid.getAvailableMoves()
        
        
        gridCopy = grid.clone() #get a copy of the board so we know what we're looking at.
        
        
        moveMax = self.Decision(gridCopy)
        
        print(moveMax)       
        return moveMax if moves else None

