import numpy as np
from progressbar import progressbar as pb
from pickle import dump, load
import encoder_decoder as ed

def initialize_statelist():

    # generate
    statelist = np.zeros((3**9, 9), dtype = np.int8)
    for i in range(9):
        a = np.array([-1,0,1], dtype = np.int8)
        b = np.ones(3, dtype = np.int8)
        for j in range(i):
            a = np.kron(a,b)
        for j in range(8-i):
            a = np.kron(b,a)
        statelist[:,i] = a

    # purge
    statelist = statelist[np.abs(np.sum(statelist, axis=1)) <= 1] # invalid board states (number of crosses and circles differs by more than 1)
    # statelist = statelist[(np.sum(statelist, axis=1)) <= 0] # invalid board states (number of crosses and circles differs by more than 1)
    # statelist = statelist[(np.sum(statelist, axis=1)) >= -1] # invalid board states (number of crosses and circles differs by more than 1)
    # permutations = ed.get_permutations()
    # for i in pb(range(statelist.shape[0])):
    #     if i >= statelist.shape[0]-1: break
    #     j = i+1
    #     while j < statelist.shape[0]:
    #         match, _ = ed.compare_states(statelist[i], statelist[j], permutations)
    #         if match: statelist = np.delete(statelist, j, axis=0)
    #         else: j += 1
    return statelist

if __name__=='__main__':
    statelist = initialize_statelist()
    with open('data/statelist', 'wb') as file:
        dump(statelist, file)