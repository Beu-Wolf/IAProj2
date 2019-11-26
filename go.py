import pickle
import sys
import time
import random
import curses
from solve import *

nStates = []
maxActions = []
initialState = []
transitions = []
rewards = []

with open("mapasgraph2.pickle", "rb") as fp:   #Unpickling
        AA = pickle.load(fp)

def getns(envNumber):
    return nStates[envNumber]


def getma(envNumber):
    return maxActions[envNumber]


def getis(envNumber):
    return initialState[envNumber]


def gett(envNumber):
    return transitions[envNumber]


def getr(envNumber):
    return rewards[envNumber]







def setEnv0():
    global nStates
    global maxActions
    global initialState
    global transitions
    global rewards

    nStates.append(114)
    maxActions.append(15)
    initialState.append(1)
    transitions.append(AA[0])
    R = [-1]*114
    R[7] = 1
    R[1] = 0
    R[2] = 0
    R[3] = 0
    R[4] = 0
    rewards.append(R)

def setEnv1():
    global nStates
    global maxActions
    global initialState
    global transitions
    global rewards

    nStates.append(114)
    maxActions.append(15)
    initialState.append(1)
    transitions.append(AA[0])
    R = [-1]*114
    R[10] = 1
    rewards.append(R)



def runagent(A, T, R, I = 1, learningphase=True, nlearn = 1000, ntest = 100):
        J = 0
        if learningphase:
                n = nlearn
        else:
                n = ntest
                
        st = I
        for ii in range(1,n):
                aa = T[st][0]
                if learningphase:
                        a = A.selectactiontolearn(st,aa)
                else:
                        a = A.selectactiontoexecute(st,aa)
                try:
                        nst = T[st][0][a]
                except:
                        print(st,a)
                r = R[st]
                J += r
                #print(st,nst,a,r)

                if learningphase:
                        A.learn(st,nst,a,r)
                else:
                        #print(st,nst,a,r)
                        pass
                
                st = nst

                # if ii is multiple of 15
                if not ii%15:
                        st = I

        # avg reward
        return J/n

def testEnv(envNr, lr=0.9, gamma=0.9, tao=1, flearn = 1000, slearn = 10000):
    ns = getns(envNr)
    ma = getma(envNr)
    iS = getis(envNr)
    T = gett(envNr)
    R = getr(envNr)
    
    res = [] # res[0] = fastScore, res[1] = slowScore
    A = LearningAgent(ns,ma, lr, gamma, tao)
    runagent(A, T, R, I = iS, learningphase=True, nlearn = flearn)
    Jn = runagent(A, T, R, I = iS, learningphase=False, ntest = 10)
    res.append(Jn)

    runagent(A, T, R, I = iS, learningphase=True, nlearn = slearn)
    Jn = runagent(A, T, R, I = iS, learningphase=False, ntest = 10)
    res.append(Jn)
    return res


# due to the randomness in the learning process, we will run everythin NREP times
# the final grades is based on the average on all of them
def epoch(NREP = 5, lr = 0.9, gamma = 0.9, tao = 1):
        NREP = 1
        val = [0,0,0,0]
        #print("exemplo 1")
        for nrep in range(0,NREP):       
                # create agent
                A = LearningAgent(114,15, lr, gamma, tao)

                # next states
                T = AA[0]

                # rewards
                R = [-1]*114
                R[7] = 1
                R[1] = 0
                R[2] = 0
                R[3] = 0
                R[4] = 0
                
                # T contains the list of possible next states
                # T[14][0] - contains the possible next states of state 14

                #print("# 1st learning phase")
                # in this phase your agent will learn about the world
                # after these steps the agent will be tested
                runagent(A, T, R, I = 1, learningphase=True, nlearn = 500)
                #print("# testing phase")
                # in this phase your agent will execute what it learned in the world
                # the total reward obtained needs to be the optimal
                Jn = runagent(A, T, R, I = 1, learningphase=False, ntest = 10)
                val[0] += Jn
                print("average reward",Jn)
                #print("# 2nd learning phase")
                runagent(A, T, R, I = 1, learningphase=True, nlearn = 10000)
                #print("# testing phase")
                Jn = runagent(A, T, R, I = 1, learningphase=False, ntest = 10)
                val[1] += Jn
                print("average reward",Jn)

        #print("exemplo 2")
        for nrep in range(0,NREP):
                A = LearningAgent(114,15, lr, gamma, tao)

                T = AA[0]
                R = [-1]*114
                R[10] = 1
                # T contains the list of possible next states
                # T[14][0] - contains the possible next states of state 14


                #print("# learning phase")
                # in this phase your agent will learn about the world
                # after these steps the agent will be tested
                # runagent(A, T, R, I = 1, learningphase=True, nlearn = 500) # mais critico. usar para testes de stresse?
                # runagent(A, T, R, I = 1, learningphase=True, nlearn = 750) # mais critico. usar para testes de stresse?
                runagent(A, T, R, I = 1, learningphase=True, nlearn = 1000)
                #print("# testing phase")
                # in this phase your agent will execute what it learned in the world
                # the total reward obtained needs to be the optimal
                Jn = runagent(A, T, R, I = 1, learningphase=False, ntest = 10)
                val[2] += Jn
                print("average reward",Jn)
                #print("# 2nd learning phase")
                runagent(A, T, R, I = 1, learningphase=True, nlearn = 10000)
                #print("# testing phase")
                Jn = runagent(A, T, R, I = 1, learningphase=False, ntest = 10)
                val[3] += Jn
                print("average reward",Jn)        


        val = list([ii/NREP for ii in val])
        #print(val)
        cor = [(val[0]) >= 0.3, (val[1]) >= 0.3, (val[2]) >= -0.85, (val[3]) >= -0.6]
        # these values are not the optimal, they include some slack
        #print(cor)

        grade = 0
        for correct,mark in zip(cor,[3,7,3,7]):
                if correct:
                        grade += mark
        #print("Grade in these tests (the final will also include hidden tests) : ", grade)
        return grade

setEnv0()
setEnv1()

for i in range(2):
    print(testEnv(i))

epoch()

quit()

'''
sum = 0
for j in range(100):
    i = 1
    while(epoch() == 20):
            print(i, end=' ')
            sys.stdout.flush()
            i+=1
    percent = (i-1)/i
    print(f"\ngrade[{j}] percentage: {percent}")
    sum+=(i-1)/(i * 100)

print("avgGrade percentage: {}".format((sum)))
'''

for i in range(101):
    print(f"[{'='*i}>{' '*(100-i)}]{i}%", end="\r")
    time.sleep(0.1)
print("")
