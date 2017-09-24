from random import randint
from BaseAI import BaseAI
import math
 

 
class PlayerAI(BaseAI):
  

    def getNewTileValue(self):
        if randint(0,99) < 100 * 0.9:
            return 2
        else:
            return 4;
    
    def Heuristic(self, grid):
   
        averageTile = 0
        maxTile = grid.getMaxTile();
        arrayTiles = [] 
        dist = 0
        for x in range(0,3):
             for y in range(0,3):
                if grid.map[x][y] != 0:
                    arrayTiles.append(grid.map[x][y])
                    dist += ((3-x)+(3-y))^2*grid.map[x][y]
        arrayTiles.sort(reverse=True)
        #medianTile = 0
  
        if len(arrayTiles)>0:
           #medianTile = arrayTiles[(int)(len(arrayTiles)/2)]
           averageTile = sum(arrayTiles)/len(arrayTiles)
        max4Tiles = sum(arrayTiles[:4])
       
        nextmax4Tiles = sum(arrayTiles[5:8])
       
                
        costLocation = 0
        if(grid.map[0][0] == maxTile):
        #    costLocation = 20200;
        #if(grid.map[0][3] == maxTile):
        #    costLocation = 20200;
        #elif(grid.map[3][3] == maxTile):
         #   costLocation = 20200;
        #if(grid.map[3][0] == maxTile):
           costLocation = 20*maxTile
        #else:
        #   costLocation = -20*maxTile
        
        costLocation = grid.map[0][0]*10000
##        return weight1 * averageTileNumber + weight2 * MedianTileNumber
        #try to keep max tile in the corner 0,0
       #give bonus if max in corner                
                
        availableCells = len(grid.getAvailableCells())
        
                
        #cost = maxTile + availableCells*100+10*cost2/cellNum+costLocation;        
        #cost = maxTile*2+availableCells*20+5*cost2/cellNum+costLocation;        
        #cost = availableCells*20+costLocation+math.log(cost2);        
    
        #cost = maxTile*5+availableCells*20+50*cost2/cellNum+costLocation;
        #cost = availableCells+cost2/cellNum+costLocation;     
           
                    
        
        #do median too?
        #return maxTile+availableCells+(cost2/cellNum)+median*10+costLocation

        #i think after the max is 512 we need a new strategy...
        
        #from a comment on stack overflow, guy got a lot of success from using a "snake" pattern. monotonically increasing
        snake = []
        for i, col in enumerate(zip(*grid.map)):
            snake.extend(reversed(col) if i % 2 == 0 else col)

        m = max(snake)
        #return sum(x/10**n for n, x in enumerate(snake))-math.pow((grid.map[0][0] != m)*abs(grid.map[0][0] - m), 2)
        
        #print("what")
        #return availableCells*100+dist*4+max4Tiles+costLocation*10
        #return sum(x/10**n for n, x in enumerate(snake))*100+max4Tiles
        #return sum(x/10**n for n, x in enumerate(snake))+15*max4Tiles+availableCells*10+dist*0.01#+costLocation*2
       
        #return sum(x/8**n for n, x in enumerate(snake))*500+5*max4Tiles+availableCells*20+dist*0.4+costLocation*10
        #return sum((x*(16-n)) for n, x in enumerate(snake))*2+10*max4Tiles#+availableCells*5       
        #return dist+max4Tiles+availableCells*500
        
        
        #just create a set of weights that snake.
       
        
        
        

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
    
    def Terminate(self, grid, depth): #test depth (many moves cells free
##        cells = grid.getAvailableCells()
    ##        moves = grid.getAvailableMoves()
    ##        if len(moves)==0:
##        if len(cells) < 10:
        if depth>5:
            return True
        else:
            return False
    
    def Maximize(self, grid, alpha, beta,depth):
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
            (tmp, utility) = self.Minimize(child,alpha,beta,depth+1)
            
            if utility > maxUtility:
                (maxChild, maxUtility) = (moves[i], utility)
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        return (maxChild, maxUtility)
    
    
    def Minimize(self,grid,alpha, beta, depth):
        if self.Terminate(grid,depth):
          return (None, self.Heuristic(grid))
        
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
        moves = grid.getAvailableMoves()
        
        
        gridCopy = grid.clone() #get a copy of the board so we know what we're looking at.
        
        
        moveMax = self.Decision(gridCopy)
        
        #print('sausage:', moveMax)
        if moveMax != None:
       	    return moveMax
        else:
            return moves[randint(0, len(moves) - 1)] if moves else None
            


