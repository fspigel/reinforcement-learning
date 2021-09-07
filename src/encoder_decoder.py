import numpy as np

def get_permutations():
    permutations = [np.array(range(9), dtype = np.int8)]
    permutations.append(np.array([[2, 5, 8],
                                  [1, 4, 7],
                                  [0, 3, 6]], dtype = np.int8).reshape(-1,1)[:,0])
    permutations.append(np.array([[8, 7, 6],
                                  [5, 4, 3],
                                  [2, 1, 0]], dtype = np.int8).reshape(-1,1)[:,0])
    permutations.append(np.array([[6, 3, 0],
                                  [7, 4, 1],
                                  [8, 5, 2]], dtype = np.int8).reshape(-1,1)[:,0])
    for i in range(4):
        permutations.append(permutations[i].reshape(3,3)[::-1,:].reshape(-1,1)[:,0])
        permutations.append(permutations[i].reshape(3,3)[:,::-1].reshape(-1,1)[:,0])
    return permutations

def compare_states(state1, state2, permutations):
    for p in permutations:
        if (state1 == state2[p]).all(): return True, p
    return False, permutations[0]