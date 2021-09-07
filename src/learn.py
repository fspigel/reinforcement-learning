from matplotlib.pyplot import hist
import numpy as np
import encoder_decoder as ed
from random import random
from progressbar import progressbar as pb
from pickle import dump, load

from initialize_statelist import initialize_statelist

def initialize_brain(statelist):
    brain = np.zeros(statelist.shape)
    for i, state in enumerate(statelist):
        idx = state==0
        s = np.sum(idx)
        if s > 0:
            brain[i, idx] = 1/s
    return brain

def find_state_id(state, statelist):
    global state_hit_tracker
    # permutations = ed.get_permutations()
    idx = -1
    # for p in permutations:
    try:
        idx1 = np.where((statelist==state).all(axis=1))[0]
    except IndexError as e:
        print(e)
        print(state)
    if len(idx1)==1:
        return idx1[0]
    raise RuntimeError('state not found in statelist: ', state)

def game(statelist, brain1, brain2, state = np.zeros(9)):
    winner = 0
    decisions = []
    history = []
    history_readable = [state]
    starting_step = np.sum(np.abs(state))
    for i in range(9-int(starting_step)):
        brain = brain1 if i%2==0 else brain2
        state, d, idx = move(state, statelist, brain)
        # state = state[p]
        history.append(idx)
        decisions.append(d)
        # p_hist.append(p)
        # state_p = np.copy(state)
        # for p_ in p_hist: 
        #     state_p = state_p[p_]
        # state_p = state_p if i%2!=0 else -state_p
        # history_readable.append(state[p])
        history_readable.append(state if i%2==0 else state*-1)
        if check_victory(state): 
            winner = 1 if i%2==0 else -1
            break
        state = -state
    # if i%2==1: state=-state
    idx = find_state_id(state, statelist)
    history.append(idx)
    return winner, np.array(decisions), np.array(history), np.array(history_readable)

def move(state, statelist, brain):
    idx = find_state_id(state, statelist)

    decisions = brain[idx]
    r = random()
    steps = np.array([np.sum(decisions[:i+1]) for i in range(len(decisions))])
    try:
        decision = np.where(r < steps)[0][0]
    except IndexError:
        print(flush=True)
        print('IndexError encountered')
        print('idx: ', idx)
        print('state:\n', state.reshape(3,3))
        print('brain:\n', brain[idx].reshape(3,3))
        print(r)
        print(decisions)
        print(steps)
        print(np.where(r < steps))
    new_state = np.copy(state)
    new_state[decision] = 1
    return new_state, decision, idx

def get_brain_slice(state, statelist, brain):
    idx = find_state_id(state, statelist)
    return brain[idx].reshape(3,3)

def check_victory(state):
    board = state.reshape(3,3)
    for i in range(3):
        if (board[:,i] == np.array([1,1,1])).all(): return True
        if (board[i,:] == np.array([1,1,1])).all(): return True
    if (np.diag(board) == np.array([1,1,1])).all(): return True
    if (np.diag(board.T) == np.array([1,1,1])).all(): return True
    return False

def update_single(row, d, rate, increase, idx, statelist):
    if (row==1).any(): return
    # if increase: new_value = rate+row[d]-rate*row[d]
    # else: new_value = row[d] * (1-rate)
    if increase: new_value = row[d] + rate
    else: new_value = row[d] - rate

    # regularization
    new_value = np.max([np.min([new_value, 0.999]), 0.001])

    # normalize row
    mag = np.sum(row)-row[d]
    row *= (1-new_value)/mag
    row[d] = new_value

def update(brain, history, decisions, winner, rate, statelist):
    updates = np.zeros(statelist.shape[0], dtype=np.uint8)

    # global updates
    history = history[:-1]
    updates[history] += 1
    if winner == 1:
        losermoves = np.array([i % 2 != 0 for i in range(len(history))])
    else:
        losermoves = np.array([i % 2 == 0 for i in range(len(history))])
    winnermoves = np.array(1+losermoves*-1, dtype=bool)
    for i, [idx, d, good] in enumerate(zip(history, decisions, winnermoves)):
        if i == len(history) - 1:
            _ = 0
            pass
        rate_current = rate if i != len(history) - 1 else 0.9999
        update_single(brain[idx], d, rate_current, good, idx, statelist)

def evaluate(brain, statelist, brain0=None):
    if brain0 is None: brain0 = initialize_brain(statelist)
    count_winner = 0
    count_draw = 0
    count_loser = 0
    zloss_history = []
    rounds = 100
    for i in range(rounds):
        w, _, _, _ = game(statelist, brain, brain0)
        if w == 1: count_winner += 1
        elif w == 0: count_draw += 1
        elif w == -1: 
            count_loser += 1
    for i in range(rounds):
        w, _, _, _ = game(statelist, brain0, brain)
        if w == -1: count_winner += 1
        elif w == 0: count_draw += 1
        elif w == 1:
            count_loser += 1
    w1, d1, l1 = count_winner/(rounds*2), count_draw/(rounds*2), count_loser/(rounds*2)
    count_winner = 0
    count_draw = 0
    count_loser = 0
    for i in range(rounds):
        w = game(statelist, brain, brain)[0]
        if w == 1: count_winner += 1
        elif w == 0: count_draw += 1
        elif w == -1: count_loser += 1
    for i in range(rounds):
        w = game(statelist, brain, brain)[0]
        if w == -1: count_winner += 1
        elif w == 0: count_draw += 1
        elif w == 1: count_loser += 1
    w2, d2, l2 = count_winner/(rounds*2), count_draw/(rounds*2), count_loser/(rounds*2)
    return [w1, d1, l1, w2, d2, l2]

def train(statelist = None, brain = None, rounds = 1_000_000, rate = 0.03):
    if statelist is None: 
        print('boop!')
        statelist = initialize_statelist()
    if brain is None: brain = initialize_brain(statelist)
    brain0 = initialize_brain(statelist)

    evaluation_history = []
    evaluate_step = int(rounds/20)
    for i in pb(range(rounds)):
        if i%evaluate_step == 0: 
            evaluation = evaluate(brain, statelist, brain0)
            evaluation_history.append(evaluation)
            # for loss in evaluation[1][:int(len(evaluation[1])/2)]:
            #     for i in range(100):
            #         w, d, h, _ = game(statelist, brain, brain0, statelist[loss[3]])
            #         if w!= 0: update(brain, h, d, w, rate, statelist)
            # for loss in evaluation[1][int(len(evaluation[1])/2):]:
            #     for i in range(100):
            #         w, d, h, _ = game(statelist, brain0, brain, statelist[loss[3]])
            #         if w!= 0: update(brain, h, d, w, rate, statelist)
        w, d, h, _ = game(statelist, brain, brain)
        if w == 0: continue
        update(brain, h, d, w, rate, statelist)
        w, d, h, _ = game(statelist, brain, brain0)
        if w == 0: continue
        update(brain, h, d, w, rate, statelist)
        w, d, h, _ = game(statelist, brain0, brain)
        if w == 0: continue
        update(brain, h, d, w, rate, statelist)
    return brain, evaluation_history

def train_by_state(statelist = None, brain = None, rate = 0.03):
    if statelist is None: 
        print('boop!')
        statelist = initialize_statelist()
    if brain is None: brain = initialize_brain(statelist)
    brain0 = initialize_brain(statelist)
    statelist_move_counter = np.sum(np.abs(statelist), axis = 1)
    reorder = np.argsort(statelist_move_counter)
    print('now training per state')
    for i in pb(range(len(statelist))):
        current_state = statelist[reorder[i]]
        if np.sum(current_state == -1) < np.sum(current_state == 1): continue
        if np.sum(np.abs(current_state)) == 9: continue
        for j in range(1_000):
            w, d, h, _ = game(statelist, brain, brain, statelist[reorder[i]])
            update(brain, h, d, w, rate, statelist)
    print(evaluate(brain, statelist, brain0))
    return brain

if __name__=='__main__':
    with open('data/statelist', 'rb') as f:
        statelist = load(f)

    brain = initialize_brain(statelist)
    brain0 = initialize_brain(statelist)

    state_hit_tracker = np.zeros(len(statelist), dtype=int)

    # rounds = 1_000_000
    # rate = 0.03

    print(flush=True)
    print(evaluate(brain, statelist), flush=True)

    total_evaluation_history = []
    for rate in [0.01, 0.01, 0.001]:
        brain, evaluation_history = train(statelist=statelist, brain=brain, rounds=20_000, rate=rate)
        # brain = train_by_state(statelist=statelist, brain=brain, rate=rate)
        total_evaluation_history.append(evaluation_history)
        print(evaluate(brain, statelist), flush=True)
        print(np.linalg.norm(brain-brain0))
    
    with open('data/total_evaluation_14', 'wb') as f: dump(total_evaluation_history, f)
    with open('data/brain_14', 'wb') as f: dump(brain, f)
    with open('data/hit_14', 'wb') as f: dump(state_hit_tracker, f)
    print(brain[:10])

    