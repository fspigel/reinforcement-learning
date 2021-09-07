# %%
import numpy as np
import matplotlib.pyplot as plt
import learn
from pickle import load, dump
from imp import reload
from progressbar import progressbar as pb
project_dir = 'C:/Users/fspig/Documents/Projects/tictactoe_bot/'
with open(project_dir+'data/statelist', 'rb') as f: statelist = load(f)

def display_brain_slice(brain, state):
    i, p = learn.find_state_id(state.reshape(1,-1)[0], statelist) 
    brain_slice = brain[i,np.argsort(p)].reshape(3,3)
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
# %%
len(statelist)
# %%
type(statelist)
# %%
with open(project_dir+'data/brain_13', 'rb') as f: brain = load(f)
# %%
print(learn.evaluate(brain, statelist))
# %%
new_brain = np.copy(brain)
for i in range(len(statelist)):
    # find matching state
    idx, p = learn.find_state_id(statelist[i]*-1, statelist)
    p_inv = np.argsort(p)
    if i == 1253 or idx == 1253:
        print(flush=True)
        print(i, idx, p)
        print(statelist[i])
        print(statelist[idx])

        print(['{:.2f}'.format(item) for item in new_brain[i]])
        print(['{:.2f}'.format(item) for item in new_brain[idx]])

        # print(['{:.2f}'.format(item) for item in brain[i, p_inv]])
        # print(['{:.2f}'.format(item) for item in brain[idx]])


        print()

    new_row = (brain[i]+brain[idx, p_inv])/2
    new_brain[i] = new_row
    new_brain[idx, p_inv] = new_row
    if i == 1253 or idx == 1253:
        print(flush=True)
        print(i, idx, p)
        print(statelist[i])
        print(statelist[idx])

        print(['{:.2f}'.format(item) for item in new_brain[i]])
        print(['{:.2f}'.format(item) for item in new_brain[idx]])

        # print(['{:.2f}'.format(item) for item in brain[i, p_inv]])
        # print(['{:.2f}'.format(item) for item in brain[idx]])


        print()
# %%
for i in pb(range(len(statelist))):
    # find matching state
    idx, p = learn.find_state_id(statelist[i]*-1, statelist)
    p_inv = np.argsort(p)

    # print(statelist[i])
    # print(statelist[idx, p])

    # print(['{:.2f}'.format(item) for item in brain[i]])
    # print(['{:.2f}'.format(item) for item in brain[idx,p]])

    # print(['{:.2f}'.format(item) for item in brain[i, p_inv]])
    # print(['{:.2f}'.format(item) for item in brain[idx]])

    # print()
    if (brain[i] != brain[idx,p]).any(): 
        print(i, idx)
        print(['{:.2f}'.format(item) for item in brain[i]])
        print(['{:.2f}'.format(item) for item in brain[idx,p]])
# %%
print(statelist[1000]!=0)
print(statelist[1002]!=0)
print((statelist[1000]!=0)*(statelist[1002]!=0))
# print((statelist[1000]!=0)*-1)
# %%
for i in range(len(statelist)):
    if ((statelist[i]!=0)*(new_brain[i]!=0)).any():
        print(i)
# %%
i = 1253
print(statelist[i])
print(['{:.2f}'.format(item) for item in new_brain[i]])
print(['{:.2f}'.format(item) for item in brain[i]])
# %%
i = 1257
idx, p = learn.find_state_id(statelist[i]*-1, statelist)
# print(statelist[i])
# print(statelist[idx, p])
# print(['{:.2f}'.format(item) for item in brain[i]])
# print(['{:.2f}'.format(item) for item in brain[idx,p]])
# print(['{:.2f}'.format(item) for item in new_brain[i]])
# print(['{:.2f}'.format(item) for item in new_brain[idx,p]])


print(statelist[i])
print(['{:.2f}'.format(item) for item in brain[i]])
print(['{:.2f}'.format(item) for item in new_brain[i]])

print(statelist[idx])
print(['{:.2f}'.format(item) for item in brain[idx]])
print(['{:.2f}'.format(item) for item in new_brain[idx]])

# %%
brain0 = learn.initialize_brain(statelist)
count = 0
for i in range(len(statelist)):
    if np.linalg.norm(brain[i]-brain0[i]) > 0.001: count += 1
print(count/len(statelist))
# %%
for item in new_brain: print(np.sum(item))
# %%
print(learn.evaluate(brain, statelist))
print(learn.evaluate(new_brain, statelist))
# %%
reload(learn)
state = np.array([
    [1, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype = np.int8)
display_brain_slice(brain, state)
# %%
brain0 = learn.initialize_brain(statelist)
# %%
reload(learn)
brain2 = np.copy(brain)
for i in range(1000):
        w, d, h, _ = learn.game(statelist, brain2, brain0, state=state.reshape(1,-1)[0])
        print(i, brain2[1055][1])
        if w == 0: continue
        learn.update(brain2, h, d, w, 0.01, statelist)
plt.figure(0)
display_brain_slice(brain, state)
plt.figure(1)
display_brain_slice(brain2, state)
# %%
# %%
with open(project_dir+'data/brain_8', 'rb') as f: brain = load(f)
# %%
reload(learn)
state = np.array([
    [1, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype = np.int8)
display_brain_slice(brain, state)

# %%
learn.evaluate(brain, statelist)
# %%
with open(project_dir+'data/hit_9', 'rb') as f: hit_tracker = load(f)
# %%
plt.plot(np.sort(hit_tracker))
# %%
with open(project_dir+'data/brain_12', 'rb') as f: brain = load(f)
# %%
reload(learn)
state = np.array([
    [1, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype = np.int8)
display_brain_slice(brain, state)

# %%
with open(project_dir+'data/hit_12', 'rb') as f: hit_tracker = load(f)

# %%
reload(learn)
brain0 = learn.initialize_brain(statelist)
w = 0
count = 0
while w != 1:
    w, d, h, h_readable = learn.game(statelist, brain0, brain)
    print(w)
    count += 1
print(count)

# %%
for item in h_readable:
    print(item.reshape(3,3))
    print()

# %%
problem_state = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, -1]
], dtype=np.int8)*-1
print(learn.find_state_id(problem_state.reshape(1,-1)[0], statelist))
display_brain_slice(brain, problem_state)
print(brain[156])
# %%
brain1 = np.copy(brain)
log = []
for i in range(10000):
    w, d, h, _ = learn.game(statelist, brain1, brain1, problem_state.reshape(1,-1)[0])
    log.append(brain1[156, 6])
    learn.update(brain1, h, d, w, 0.001, statelist)
display_brain_slice(brain1, problem_state)
# %%
plt.plot(log)
# %%
print(hit_tracker[156])
# %%
print(len(statelist))
# %%
import initialize_statelist
reload(initialize_statelist)
statelist2 = initialize_statelist.initialize_statelist()
print(len(statelist2))
# %%
statelist2[np.sum(statelist, axis=1) <=0].shape
