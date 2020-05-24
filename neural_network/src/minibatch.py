import numpy as np


def create_rand_seq_permutations(db):
    seq_idx = {}
    for key in db.keys():
        if key not in seq_idx:
            seq_idx[key] = None
        seq_idx[key] = np.random.permutation(np.arange(len(db[key])))

    return seq_idx


def generate_batch(db, perm, batch_size=4, shuffle=False):
    idx = 0

    while True:
        output_seq = []
        output_idx = np.array([])
        labels = np.array([])
        for key in perm.keys():
            output_idx = np.concatenate(
                [output_idx, perm[key][idx:idx+batch_size]])
            labels = np.concatenate(
                [labels, key*np.ones(batch_size, dtype='int')])
        if shuffle:
            rand_idx = np.random.permutation(np.arange(len(output_idx)))
            output_idx = output_idx[rand_idx]
            labels = labels[rand_idx]

        output_idx = output_idx.astype('int')
        labels = labels.astype('int')

        for i in range(len(labels)):
            output_seq.append(db[labels[i]][output_idx[i]])

        yield output_seq, labels
        idx += batch_size
