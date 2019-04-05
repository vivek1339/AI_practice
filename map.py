from tictactoe import play,printBoard
import random
import sys
from brain_single import dgn
from tqdm import tqdm
from datetime import datetime
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
last_reward_1=0
last_reward_2=0
ai_wins=0
ai_looses=0
ai_draw=0
ai_illegal=0
total_reward=0
def player_two_move(playeroneturn,choices,count,possible_moves):
	choice=random.choice(possible_moves)
	result,choices,count=play(choice,playeroneturn,choices,count)
	#print(choice)
	#print(possible_moves)
	possible_moves=possible_moves.remove(choice)
	return result,choices,count,possible_moves
def play_game(brain2):
	global dgn
	global last_reward_1,last_reward_2
	global ai_wins
	global ai_wins
	global ai_looses
	global ai_draw
	global ai_illegal
	total_reward_1=0
	total_reward_2=0
	choices=[]
	possible_moves=[1,2,3,4,5,6,7,8,9]
	playeroneturn=True
	winner=False
	count=0
	choice=10
	for x in range (0, 9) :
	    choices.append(0)
	while not winner and possible_moves != []:
		total_reward_1+=last_reward_1
		total_reward_2+=last_reward_2
		if(playeroneturn):
			choice=int(brain2.update(last_reward_1,choices))+1
			result,choices,count=play(choice,playeroneturn,choices,count)
		else:
			#result,choices,count,possible_moves=player_two_move(playeroneturn,choices,count,possible_moves)
			choice=random.choice(possible_moves)
			#choice=int(brain1.update(last_reward_2,choices))+1
			result,choices,count=play(choice,playeroneturn,choices,count)
			#possible_moves.remove(choice)
		#print(str(choice))
		if(result==1):
			winner=True
			possible_moves.remove(choice)
			#printBoard(choices)
			if(playeroneturn):
				#print("ai wins")
				last_reward_1=10
				last_reward_2=-10
				ai_wins+=1
			else:
				#print("random wins")
				last_reward_1=-10
				last_reward_2=10
				ai_looses+=1
		elif(result==0):
			#printBoard(choices)
			possible_moves.remove(choice)
			if playeroneturn:
				last_reward_1=-1
				last_reward_2=0
			else:
				last_reward_2=-1
				last_reward_1=0
			
			playeroneturn= not playeroneturn
			#print(choice)
			
		elif(result==-1):
			#printBoard(choices)
			#print("Illegal move")
			if playeroneturn:
				last_reward_1=-50
				last_reward_2=0
			else:
				last_reward_2=-50
				last_reward_1=0
			ai_illegal+=1
		if len(possible_moves) <1:
			last_reward_1=last_reward_2=5
			ai_draw+=1
			winner=True
	return total_reward_1,total_reward_2
def find_result(result,playeroneturn):
	global last_reward
	global ai_wins
	global ai_wins
	global ai_looses
	global ai_draw
	global ai_illegal
	if(playeroneturn):
		if(result==1):
			#print("ai wins")
			ai_wins+=1
			last_reward=10
			return True,1
		elif(result==0):
			last_reward=0


#brain2=dgn(9,0.1)
try:
	with open('nn_obj3.file','rb') as input:
		try:
			brain2 = pickle.load(input)
		except:
			print("object abscent")
			brain2=dgn(9,0.1)
		print("object loaded from memory")
except:
	print("new object created")
	brain2=dgn(9,0.1)
loss_a=[]
try:
	with open('loss.file','rb') as input:
		try:
			loss_a = pickle.load(input)
		except:
			print("object abscent")
			loss_a=[]
		print("object loaded from memory")
except:
	print("new object created")
	loss_a=[]

"""with open('nn_obj.pkl','wb') as output1:
	pickle.dump(brain1,output1,pickle.HIGHEST_PROTOCOL)
	print("NN saved")"""
"""brain2.temprature=70
brain2.model.lr=0.1
brain2.gamma=0.9"""
brain2.temprature=70
brain2.model.lr=0.05
brain2.gamma=0.9
mean_reward_1=0
mean_reward_2=0
last_mean_reward=-99999
print(brain2.temprature)
t_start=time.time()
error=[]
loss=[]
itera=[]
for i in tqdm(range(10000)):
	total_reward_1,total_reward_2=play_game(brain2)
	mean_reward_1+=total_reward_1
	mean_reward_2+=total_reward_2
	if(i%1000==0 and i != 0):
		print("mean_reward_1 ="+str(mean_reward_1/1000))
		print("mean_reward_2 ="+str(mean_reward_2/1000))
		mean_reward_1=mean_reward_1/1000
		mean_reward_1=mean_reward_1/1000
		#if(last_mean_reward>mean_reward):
		#	brain2.model.lr=brain2.model.lr/10
		#	brain2.gamma=brain2.gamma/2
		#last_mean_reward=mean_reward
		error.append(mean_reward_1)
		loss.append(np.sum(brain2.loss_array)/len(brain2.loss_array))
		itera.append(i)
		loss_a.append(brain2.score())
		mean_reward_1=0
print("ai_wins ="+str(ai_wins))
print("ai_looses ="+str(ai_looses))
print("ai_draw ="+str(ai_draw))
print("ai_illegal ="+str(ai_illegal))
print("score = "+str(brain2.score()))
t_end=time.time()
print("did all 100 times in time = "+str((t_end-t_start))+"s")
with open('nn_obj3.file','wb') as output1:
	pickle.dump(brain2,output1,pickle.HIGHEST_PROTOCOL)
	print("NN saved")
with open('loss.file','wb') as output2:
	pickle.dump(loss_a,output2,pickle.HIGHEST_PROTOCOL)

plt.plot(itera,error)
plt.show()
plt.plot(itera,loss)
plt.show()
plt.plot(loss_a)
plt.show()