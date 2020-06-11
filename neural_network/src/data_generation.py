import numpy as np


def sliding_window(a, w=4, o=2, copy=False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def create_rand_seq_permutations(db):
    seq_idx = {}
    for key in db.keys():
        if key not in seq_idx:
            seq_idx[key] = None
        seq_idx[key] = np.random.permutation(np.arange(len(db[key])))

    return seq_idx


def generate_all_data(db, b, targetClas, nClas=7, nTrials=200, nWin=1, S=16, nX=9, nY=9):
    inputs = np.zeros((nWin * nTrials * nClas, S, nX, nY))
    targets = np.zeros(nWin * nTrials * nClas, dtype=int)
    wins = np.zeros(nWin * nTrials * nClas, dtype=int)

    seq_perms = create_rand_seq_permutations(db)
    k = 0
    for trial in range(0, nTrials):
        for clas in targetClas:
            perm = seq_perms[clas][trial]
            tmp = db[clas][perm]

            for win in range(0, nWin):
                inputs[k] = tmp[b[win, :]]
                targets[k] = clas
                wins[k] = win
                k = k + 1
    return inputs, targets, wins


def generate_batch(db, perm, batch_size=4, shuffle=False, classes=None):
    idx = 0

    if classes is None:
        classes = perm.keys()

    while True:
        output_seq = []
        output_idx = np.array([])
        labels = np.array([])

        for key in classes:
            output_idx = np.concatenate(
                [output_idx, perm[key][idx:idx + batch_size]])
            labels = np.concatenate(
                [labels, key * np.ones(batch_size, dtype='int')])
        if shuffle:
            rand_idx = np.random.permutation(np.arange(len(output_idx)))
            output_idx = output_idx[rand_idx]
            labels = labels[rand_idx]

        output_idx = output_idx.astype('int')
        labels = labels.astype('int')

        if len(labels) < 1 or len(output_idx) < 1:
            break

        for i in range(len(labels)):
            output_seq.append(db[labels[i]][output_idx[i]])

        yield np.array(output_seq), labels
        idx += batch_size
