# Grupo 007 - Afonso Gonçalves 89399, Daniel Seara 89427
import random
import math
import numpy

def maxind(iterable):
        max = -math.inf
        mi = False
        for i in range(len(iterable)):
                if iterable[i] > max:
                        max = iterable[i]
                        mi = i
        return mi

def softMaxDistribution(qvals, tao):
        cumProb = []
        sum = 0
        for q in qvals:
                prob = math.exp(q/tao)
                cumProb.append(sum + prob)
                sum += prob
        
        return [p/sum for p in cumProb]


# LearningAgent
class LearningAgent:
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA, lr = 0.99, gamma = 0.98, tao = 1):
                self.nS = nS
                self.nA = nA

                # meta parameters
                self.learningCount = 0          
                self.lr = lr                    
                self.gamma = gamma              
                self.tao = tao              
                
                self.Qvals = [False for _ in range(nS)]
                self.Alphavals = [[self.lr for _ in range(self.nA)] for _ in range(self.nS)]
                self.freqTable = [[0 for _ in range(self.nA)] for _ in range(self.nS)]


        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                if(not self.Qvals[st]):
                        self.Qvals[st] = [0 for _ in aa]
        
                if all([ qv == 0 for qv in self.Qvals[st]]):
                    return random.randrange(0, len(aa))

                # softmax
                probs = softMaxDistribution(self.Qvals[st], self.tao)
                decision = random.random()
                for i in range(len(probs)):
                        if(probs[i] >= decision):
                            return i
                
                return random.randrange(0, len(aa))

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                return maxind(self.Qvals[st]) if self.Qvals[st] else random.randrange(0, len(aa))


        # this function is called after every action
        # st - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                self.Alphavals[ost][a] = self.newAlpha(self.freqTable[ost][a]) if self.Alphavals[ost][a] > 0.1 else self.Alphavals[ost][a]
                self.freqTable[ost][a] += 1
                self.Qvals[ost][a] += self.Alphavals[ost][a] * (r + (self.gamma * max(self.Qvals[nst]) if self.Qvals[nst] else 0) - self.Qvals[ost][a])


        def newAlpha(self, x):
                alphaMax = self.lr
                alphaMin = 0.1
                k = 1000
                x0 = 20000
                return (alphaMax - alphaMin)/(1 + numpy.exp(-k*(-x + x0))) + alphaMin