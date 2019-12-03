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
                st = nst

                # if ii is multiple of 15
                if not ii%15:
                        st = I

        return J/n # avg reward


def testEnv(envNr, lr=0.9, gamma=0.9, tao=1, flearn = 1000, slearn = 10000, numReps = 1):
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


# quit()



# genetic algorithm

# envWeights = [0.3, 0.7]
envWeights = [1, 0]
runWeights = [0.67, 0.33]

numSteps = [1000, 750, 500]
stepWeights = [0.2, 0.3, 0.5]

assert(sum(runWeights) == 1)
assert(sum(envWeights) == 1)
assert(sum(stepWeights) == 1 and len(stepWeights) == len(numSteps))


POPSIZE = 200
TAORANGE = 9
NREPS = 5
NPARENTS = 2
MUTATIONRATE = [0.01, 0.01, 0.01]
MUTATIONAMPLITUDE = [0.05, 0.05, TAORANGE/5]

LIMSTEPS = 1

def cumFitness(scores):
    # make scores positive
    mv = min(scores)
    res = [scores[i] - mv for i in range(len(scores))]

    # increase score difference
    res = [math.exp(res[i]*10) for i in range(len(scores))]

    # normalize scores
    total = sum(res)
    for i in range(1, len(res)):
        res[i] += res[i-1]

    for i in range(len(res)):
        res[i] = res[i]/total
    return res

def score(results):
    return results[-1][0]

# generate first population
# agent = [lr, gamma, tao]
population = [(random.random(), random.random(), random.random() * TAORANGE) for i in range(POPSIZE)]

bestAgent = population[0]
bestResults = [['?'] * 2]*2
bestScore = -math.inf
print('\n'*6)

generation = 0
while True:
    generation += 1
    
    maxDigitSize = len(str(POPSIZE))
    existingDigits = len('[=>](/)')
    rightPadding = 0
    extra = existingDigits + 2*maxDigitSize + rightPadding
    cols = os.get_terminal_size(1).columns

    # calculate fitness
    results = [[] for _ in population]
    scores = []
    for agentID in range(len(population)):
        agent = population[agentID]
        for env in range(NENVS):
            for ns in range(min(LIMSTEPS, len(numSteps))):
                out = testEnv(env, agent[0], agent[1], agent[2], flearn = ns, numReps = NREPS)
                results[agentID].append(out)
        scores.append(score(results[agentID]))
        if(scores[agentID] >= bestScore):
            bestScore = scores[agentID]
            bestResults = results[agentID]
            bestAgent = agent
        percent = int(agentID/POPSIZE * (cols - extra))
        sys.stdout.write(u"\u001b[" + str(5) + "A") # Move up
        print(f"melhor agente: ({bestAgent[0]:.4f}, {bestAgent[1]:.4f}, {bestAgent[2]:.4f}) [[{bestResults[0][0]:.4f}, {bestResults[0][1]:.4f}], [{bestResults[1][0]:.4f}, {bestResults[1][1]}]]: {{{bestScore:.4f}}}\n")
        # print(f"melhor geracao: ({random.random():.4f}, {random.random():.4f}, {random.random():.4f}) [[{random.random():.4f}, {random.random():.4f}], [{random.random():.4f}]]: {{{random.random():.4f}}}\n")
        print(f"Generation #{generation}:")
        # print(f"[{'=' * percent}>{' ' * (cols - percent)}] ({j:03}/{n:03})")
        print(f"[={'=' * max([percent, 0])}>{' ' * (cols - percent - extra)}]({agentID:0{maxDigitSize}}/{POPSIZE:0{maxDigitSize}})")
        print(f"({agent[0]:.4f}, {agent[1]:.4f}, {agent[2]:.4f}) [[{results[agentID][0][0]:.4f}, {results[agentID][0][1]:.4f}], [{results[agentID][1][0]:.4f}, {results[agentID][1][1]:.4f}]]: {{{scores[agentID]:.4f}}}")
        # input()
    genHighScore = max(scores)
    probDistribution = cumFitness(scores)
    # print(probDistribution)

    # next generation
    nextGen = []
    for _ in range(POPSIZE):

        # chose parents
        parents = []
        for p in range(NPARENTS):
            decision = random.random()
            for i in range(len(probDistribution)):
                    if(probDistribution[i] >= decision):
                        parents.append(population[i])
                        break

        # reproduce
        # print(parents)
        child = [sum(q)/len(q) for q in list(zip(*parents))]
        # print(child)
    
        # mutate
        for i in range(len(child)):
            if random.random() < MUTATIONRATE[i]:
                child[i] += (random.random() - 0.5) * MUTATIONAMPLITUDE[i]

        nextGen.append(tuple(child))

    # print(nextGen)
    population = nextGen
    # print(f"{generation}: {genHighScore}")

# for i in range(2):
    # print(testEnv(i))

