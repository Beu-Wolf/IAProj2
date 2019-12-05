import pickle
import random

NENVS = 0

nStates = []
maxActions = []
initialState = []
transitions = []
rewards = []


with open("mapasgraph2.pickle", "rb") as fp:   #Unpickling
        AA = pickle.load(fp)

def bfs(model, src, dst):
    searchThree = {
        src: 0
    }

    queue = []
    queue.append(src)

    while(len(queue) != 0):
        curr = queue.pop(0)
        if(curr == dst):
            return searchThree[curr]
        
        depth = searchThree[curr]+1
        for a in model[curr][0]:
            if(a not in searchThree):
                queue.append(a)
                searchThree[a] = depth
    return False

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

def createEnvironment(src, dst):
    global nStates
    global maxActions
    global initialState
    global transitions
    global rewards
    global NENVS

    assert len(nStates) == len(maxActions) == len(initialState) == len(transitions) == len(rewards)

    nStates.append(114)
    maxActions.append(15)
    initialState.append(src)
    transitions.append(AA[0])

    stateCost = -1
    R = [stateCost] * 114

    # BFS to find minimum 
    pathCost = bfs(transitions[-1], src, dst)
    print(pathCost)

    # definir reward do objetivo (custo optimo = 0)
    R[dst] = -(pathCost * stateCost)

    # return indice do novo ambiente
    rewards.append(R)
    NENVS += 1
    return len(rewards)-1


# setEnv0()
# setEnv1()

for i in range(5):
    src = random.randint(1, 114-1)
    dst = random.randint(1,114-1)
    while dst == src:
        dst = random.randint(1,114-1)

    print("src: ", src)
    print("dst: ", dst)

    createEnvironment(src, dst)