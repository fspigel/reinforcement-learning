import numpy as np
import matplotlib.pyplot as plt
import learn
from progressbar import progressbar as pb
from pickle import load, dump
from imp import reload

def display_brain_slice(brain, state):
    brain_slice = brain[learn.find_state_id(state.reshape(1,-1)[0], statelist)].reshape(3,3)
    img = np.zeros((300,300))
    for i in range(3):
        for j in range(3):
            if state[i,j] == 1:
                img[i*100:(i+1)*100,j*100:(j+1)*100] = 1
            elif state[i,j] == -1:
                img[i*100:(i+1)*100,j*100:(j+1)*100] = -1
            else:
                plt.text(50+j*100, 50+i*100, '{:.2f}'.format(brain_slice[i,j]))
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    project_dir = 'C:/Users/fspig/Documents/Projects/tictactoe_bot/'
    with open(project_dir+'data/statelist', 'rb') as f: statelist = load(f)
    with open(project_dir+'data/brain_12', 'rb') as f: brain = load(f)
    problem_state = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, -1]
    ], dtype=np.int8)*-1
    print(learn.find_state_id(problem_state.reshape(1,-1)[0], statelist))
    # display_brain_slice(brain, problem_state)
    print(brain[156])

    brain1 = np.copy(brain)
    log = [brain1[156, 6]]

    yvalues1 = np.zeros(1000, dtype=np.float32) 
    yvalues2 = np.zeros(1000, dtype=np.float32) 

    plt.ion()

    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # line1, = ax.plot(yvalues1)
    # ax.set_ylim([0,1])
    # ax2 = fig.add_subplot(212)
    # line2, = ax2.plot(yvalues2)
    # ax2.set_ylim([-5,5])

    yvalues = np.zeros((3,3,1000), dtype=np.float32)
    fig, ax = plt.subplots(3,3)
    lines = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    for axis in ax.flatten(): 
        axis.set_ylim([0,1])
        axis.set_xticks([])
    for i in range(3):
        for j in range(3):
            line, = ax[i,j].plot(yvalues[i,j,:])
            ax[i,j].set_ylim([0,1])
            lines[i][j] = line
    fig.tight_layout()    
            
    win_tracker = np.zeros(5, dtype=np.int8)

    for i in pb(range(10000)):
        w, d, h, h_r = learn.game(statelist, brain1, brain1, problem_state.reshape(1,-1)[0])

        if (h_r[-1].reshape(3,3)[:,0] == -1*np.ones(3)).all(): print('boop!')
        for j in range(3):
            for k in range(3):
                yvalues[j,k,:-1] = yvalues[j,k,1:]
                yvalues[j,k,-1] = brain1[156, k+j*3]
                lines[j][k].set_ydata(yvalues[j,k,:])
        # yvalues1[:-1] = yvalues1[1:]
        # yvalues1[-1] = brain1[156, 6]
        # win_tracker[:-1] = win_tracker[1:]
        # win_tracker[-1] = w
        # yvalues2[:-1] = yvalues2[1:]
        # yvalues2[-1] = np.sum(win_tracker)
        # line1.set_ydata(yvalues1)
        # line2.set_ydata(yvalues2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        learn.update(brain1, h, d, w, 0.1, statelist)

    display_brain_slice(brain1, problem_state)
