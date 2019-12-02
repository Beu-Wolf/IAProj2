# Grupo 007 - Afonso Gonçalves 89399, Daniel Seara 89427
import random
import math

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

# TODO: PROFILING
# ajustar valor inicial dos Q-values (0 ou -0.01, ou...?)
# ajustar discount rate
# ajustar learning rate
# ajustar temperatura do softmax

# LearningAgent
class LearningAgent:
        # nS maximum number of states
        # nA maximum number of action per state
        #def __init__(self,nS,nA):
        def __init__(self,nS,nA, lr = 0.9, gamma = 0.9, tao = 1):
                self.nS = nS
                self.nA = nA

                # meta parameters
                self.learningCount = 0          # number of learning steps
                self.lr = lr                    # learing rate (0.1? 0.5??)
                self.gamma = gamma              # discount rate (1? 0.75?)
                self.tao = tao                  # exploitation rate.
                                                #       can change with learningCount

                self.Qvals = [False for _ in range(nS)]


        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                # TODO: Discuss between epsilon-greedy policy or Softmax function
                # TODO: initial value for Qtable? 0 or negative value?
                # podemos fazer bruteforce com varias politicas e vemos qual e a melhor. deixamos a correr no sigma e esta feito
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
                self.Qvals[ost][a] += self.lr * (r + (self.gamma * max(self.Qvals[nst]) if self.Qvals[nst] else 0) - self.Qvals[ost][a])
