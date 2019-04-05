"""tictactoe game for 2 players
from blogpost: http://thebillington.co.uk/blog/posts/writing-a-tic-tac-toe-game-in-python by  BILLY REBECCHI,
slightly improved by Horst JENS"""
from __future__ import print_function
import sys
choices = []

for x in range (0, 9) :
    choices.append(str(x + 1))

playerOneTurn = True
winner = False

def printBoard(choices) :
    board=choices.copy()
    for i in range(len(choices)):
        if(choices[i]==1):
            board[i]='X'
        elif(choices[i]==-1):
            board[i]='O'
        else:
            board[i]=' '
    print( '\n -----')
    print( '|' + str(board[0]) + '|' + str(board[1]) + '|' + str(board[2]) + '|')
    print( ' -----')
    print( '|' + str(board[3]) + '|' + str(board[4]) + '|' + str(board[5]) + '|')
    print( ' -----')
    print( '|' + str(board[6]) + '|' + str(board[7]) + '|' + str(board[8]) + '|')
    print( ' -----\n')


def play(choice,turn,choices,count):
    """printBoard()"""
    if(count>=9):
        return -2,choices,count
    if playerOneTurn :
        random_number=0
        #print( "Player 1:")
    else :
        random_number=1
        #print( "Player 2:")
    choice
    if(choice>9):
        sys.exit()
    if choices[choice - 1] == -1 or choices [choice-1] == 1:
        return -1,choices,count
        #print("illegal move, plase try again")
        

    if turn :
        count+=1
        choices[choice - 1] = 1
    else :
        count+=1
        choices[choice - 1] = -1

    #playerOneTurn = not playerOneTurn

    for x in range (0, 3) :
        y = x * 3
        if ((choices[y] == choices[(y + 1)]==-1 and choices[y] == choices[(y + 2)]==-1) or (choices[y] == choices[(y + 1)]==1 and choices[y] == choices[(y + 2)]==1)) :
            return 1,choices,count
            #winner = True
            #printBoard()
        if ((choices[x] == choices[(x + 3)]==1 and choices[x] == choices[(x + 6)]==1) or (choices[x] == choices[(x + 3)]==-1 and choices[x] == choices[(x + 6)]==-1)) :
            return 1,choices,count
            #winner = True
            #printBoard()

    if(((choices[0] == choices[4]==1 and choices[0] == choices[8]==1) or choices[0] == choices[4]==-1 and choices[0] == choices[8]==-1) or 
        ((choices[2] == choices[4]==1 and choices[4] == choices[6]==1) or choices[2] == choices[4]==-1 and choices[4] == choices[6]==-1)) :
        return 1,choices,count
        #winner = True
        #printBoard()
    return 0,choices,count