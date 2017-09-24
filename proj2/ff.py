
map = [[1,1,1,2],
       [1,1,1,17],
       [1,1,1,32],
       [3,1,1,4]]
snake = []
for i, col in enumerate(zip(*map)):
    print(col)
    snake.extend(reversed(col) if i % 2 == 0 else col)
print(snake)

##print(max(snake))
##print(list(enumerate(snake)))
##
print(sum(x/10**n for n, x in enumerate(snake)))

print(list(x/5**n for n, x in enumerate(snake)))

print(map[3][0])

moves = [0,1,2,3]
moves.remove(0)
moves.append(0)
print(moves)