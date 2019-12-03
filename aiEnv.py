import pickle

NENVS = 2

nStates = []
maxActions = []
initialState = []
transitions = []
rewards = []


with open("mapasgraph2.pickle", "rb") as fp:   #Unpickling
        AA = pickle.load(fp)

def bfs(src, dst):
    global transitions
    transitions.append(AA[0])
    

    searchThree = {
            src: 0,
        }

    queue = []
    queue.append(src)


    while(len(queue) != 0):
        curr = queue.pop(0)
        if(curr == dst):
            return searchThree[curr]
        
        depth = searchThree[curr]+1
        for a in transitions[0][curr][0]:
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

    assert len(nStates) == len(maxActions) == len(initialState) == len(transitions) == len(rewards)

    nStates.append(114)
    maxActions.append(15)
    initialState.append(src)
    transitions.append(AA[0])

    R = [-1] * 114

    # BFS to find minimum 
    pathCost = bfs(src, dst)
    print(pathCost)
    # definir reward do objetivo (custo optimo = 0)

    R[dst] = pathCost

    # return indice do novo ambiente
    rewards.append(R)
    return len(rewards)-1


# setEnv0()
# setEnv1()

createEnvironment(3, 7)