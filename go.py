import sys
import time
import random
import curses
import os
import math
from solve import *
from aiEnv import *

def trainAgent(A, T, R, I = 1, nIterations = 1000):
        st = I  # initial state

        n = nIterations
        for ii in range(1,n):
                # get action to learn
                aa = T[st][0]
                a = A.selectactiontolearn(st,aa)
                nst = T[st][0][a]

                # give reward
                A.learn(st,nst,a,R[st])

                # update state
                st = nst

                # if ii is multiple of 15, start from begining
                if not ii%15:
                        st = I

def evaluateAgent(A, T, R, I = 1, nIterations = 100):
        J = 0   # score
        st = I  # initial state

        n = nIterations
        for ii in range(1,n):
                # get 'best' state
                aa = T[st][0]
                a = A.selectactiontoexecute(st,aa)
                nst = T[st][0][a]

                # get reward
                J += R[st]
                if(R[st] > 0):
                        break
                st = nst

                # if ii is multiple of 15
                if not ii%15:
                        st = I

        return J/n # avg reward


def testEnv(envNr, lr=0.995, gamma=0.98, tao=1, flearn = 10000, slearn = 100000, numReps = 1):
    ns = getns(envNr)
    ma = getma(envNr)
    iS = getis(envNr)
    T = gett(envNr)
    R = getr(envNr)

    fLearnScore = 0
    sLearnScore = 0
    for rep in range(numReps):
        A = LearningAgent(ns,ma, lr, gamma, tao)

        # res = [fastscore, slowscore]
        res = [] 

        # first learn
        trainAgent(A, T, R, I = iS, nIterations = flearn)
        fLearnScore += evaluateAgent(A, T, R, I = iS, nIterations = 10)

        # second learn
        trainAgent(A, T, R, I = iS, nIterations = slearn)
        sLearnScore += evaluateAgent(A, T, R, I = iS, nIterations = 10)
    return [fLearnScore/numReps, sLearnScore/numReps]

while True:
        randomEnv()
        print(testEnv(NENVS - 1))

# quit()
