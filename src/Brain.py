import numpy as np

class Brain:
    def __init__(self, statelist) -> None:
        self.statelist = statelist
        self.brain = np.zeros(statelist.shape)
        for i, state in enumerate(self.statelist):
            idx = state==0
            s = np.sum(idx)
            if s > 0:
                self.brain[i, idx] = 1/s

    
    def move(self, state):
        pass
