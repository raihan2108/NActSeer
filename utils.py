import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg

from os.path import join
from sklearn.model_selection import train_test_split


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    '''if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])'''
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


class SimpleDataIterator(object):
    def __init__(self, train_data, train_lengths, test_data, test_lengths, user_list=None, shuffle=True, mark=True):
        self.train_data = train_data
        self.train_lengths = train_lengths
        self.test_data = test_data
        self.test_lengths = test_lengths
        self.size = train_data.shape[0]
        self.seqlen = train_data.shape[1]
        if user_list is not None:
            self.user_list = user_list
        else:
            self.user_list = None
        self.shuffle_data()
        self.cursor = 0

    def shuffle_data(self):
        perm = np.random.permutation(self.size)
        self.train_data = self.train_data[perm]
        self.train_lengths = self.train_lengths[perm]
        self.test_data = self.test_data[perm]
        self.test_lengths = self.test_lengths[perm]
        if self.user_list is not None:
            self.user_list = self.user_list[perm]
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n - 1 > self.size:
            self.shuffle_data()
            print('enter shuffle data')
        train_res = self.train_data[self.cursor: self.cursor + n]
        train_seqlen = self.train_lengths[self.cursor: self.cursor + n]
        test_res = self.test_data[self.cursor: self.cursor + n]
        test_seqlen = self.test_lengths[self.cursor: self.cursor + n]
        if self.user_list is not None:
            curr_user_list = self.user_list[self.cursor: self.cursor + n]
        self.cursor += n
        if self.user_list is not None:
            return train_res, train_seqlen, test_res, test_seqlen, curr_user_list
        else:
            return train_res, train_seqlen, test_res, test_seqlen


def load_graph(graph_file_name, n_users):
    # edges = list()
    '''with open(graph_file_name, 'r') as raw_file:
        for line in raw_file:
            edges.append(list(map(int, line.split())))'''

    # return edges

    data = pd.read_csv(graph_file_name, sep=' ', header=None, dtype=np.int32)
    rows = data[0]  # Not a copy, just a reference.
    cols = data[1]
    new_rows = np.concatenate([rows, np.arange(0, n_users)])
    new_cols = np.concatenate([cols, np.arange(0, n_users)])
    ones = np.ones(len(rows), np.int32)
    vals = np.concatenate([ones, np.ones(n_users) * 0.00001])
    matrix = sp.csr_matrix((vals, (new_rows, new_cols)))

    return matrix


def load_ids(dataset_dir, id_file_name):
    with open(join(dataset_dir, id_file_name), 'rb') as raw_file:
        user_id, reverse_user_id, item_id, reverse_item_id = pickle.load(raw_file)
    return user_id, reverse_user_id, item_id, reverse_item_id


def gather_user_history_dynamic(act_list, time_list, n_users, n_bins, n_context):
    n_examples = len(act_list)
    n_seq = max(map(len, act_list))
    comp_data = np.zeros((n_examples, n_users, n_seq, n_context), dtype=np.int32)


def load_dataset_dynamic(dataset_dir, activity_file_name, n_users, n_context):
    act_list = list()
    time_list = list()
    user_list = list()

    max_timestamp = -1.0
    min_timestamp = float('inf')
    dataset_name = dataset_dir.split('/')[-1]
    with open(join(dataset_dir, activity_file_name), 'r') as raw_file:
        for line in raw_file:
            t_item_list = list()
            t_time_list = list()
            # _, user_act = line.split()
            user = int(line.split(':')[0])
            entries = line.split()[1:]
            for a_entry in entries:
                item, time_stamp = a_entry.split(':')
                t_item_list.append(int(item.strip()))
                t_time_list.append(int(time_stamp.strip()))

                if min_timestamp > int(time_stamp.strip()):
                    min_timestamp = int(time_stamp.strip())
                if max_timestamp < int(time_stamp.strip()):
                    max_timestamp = int(time_stamp.strip())

            act_list.append(t_item_list)
            time_list.append(t_time_list)
            user_list.append(user)

    print(max_timestamp, min_timestamp)


def prepare_minibatch(tuples, inference=False, options=None):

    '''
    entry = {
            'train_act_seq': train_act_seq,
            'train_time_seq': train_time_seq,
            'train_act_label': train_act_label,
            'train_time_label': train_time_label,
            'test_act_seq': test_act_seq,
            'test_time_seq': test_time_seq,
            'test_act_label': test_act_label,
            'test_time_label': test_time_label,
            'seq_len': len(train_act_seq)
        }
    '''

    train_act_seq = [t['train_act_seq'] for t in tuples]
    train_time_seq = [t['train_time_seq'] for t in tuples]
    train_label = [t['train_act_label'] for t in tuples]
    train_time = [t['train_time_label'] for t in tuples]

    test_act_seq = [t['test_act_seq'] for t in tuples]
    test_time_seq = [t['test_time_seq'] for t in tuples]
    test_label = [t['test_act_label'] for t in tuples]
    test_time = [t['test_time_label'] for t in tuples]

    users = [t['user'] for t in tuples]
    seq_len = [t['seq_len'] for t in tuples]
    n_samples = len(tuples)

    seqs_matrix_train = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, seq in enumerate(train_act_seq):
        seqs_matrix_train[: seq_len[i], i] = seq
    seqs_matrix_train = np.transpose(seqs_matrix_train)

    seqs_matrix_test = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, seq in enumerate(test_act_seq):
        seqs_matrix_test[: seq_len[i], i] = seq
    seqs_matrix_test = np.transpose(seqs_matrix_test)

    times_matrix_train = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, time in enumerate(train_time_seq):
        times_matrix_train[: seq_len[i], i] = time
    times_matrix_train = np.transpose(times_matrix_train)

    times_matrix_test = np.zeros((options['seq_len'], n_samples)).astype('int32')
    for i, time in enumerate(test_time_seq):
        times_matrix_test[: seq_len[i], i] = time
    times_matrix_test = np.transpose(times_matrix_test)

    train_label_vec = np.array(train_label).astype('int32')
    test_label_vec = np.array(test_label).astype('int32')

    train_time_vec = np.array(train_time).astype('int32')
    test_time_vec = np.array(test_time).astype('int32')

    users_vec = np.array(users).astype('int32')
    seq_len_vec = np.array(seq_len).astype('int32')

    return ( seqs_matrix_train, seqs_matrix_test,
        times_matrix_train, times_matrix_test,
        train_label_vec, test_label_vec,
        train_time_vec, test_time_vec,
        users_vec, seq_len_vec
    )


class Loader():
    def __init__(self, data, options=None):
        self.batch_size = options['batch_size']
        self.idx = 0
        self.data = data
        self.shuffle = True
        self.n = len(data)
        self.n_words = options['user_size']
        self.indices = np.arange(self.n, dtype="int32")
        self.options = options

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def reset(self):
        self.idx = 0

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(batch_examples,
                                 inference=False,
                                 options=self.options)


def gather_user_history(act_list, time_list, n_users, n_bins, n_context):
    user_history = np.zeros((n_users, n_bins, n_context), dtype=np.int32)
    for u in range(0, n_users):
        one_act_list = act_list[u]
        one_time_list = time_list[u]
        for t in range(0, n_bins):
            t_list = [i for i, x in enumerate(one_time_list) if x == t]
            loop_t = t - 1
            if loop_t >= 0:
                while len(t_list) < n_context:
                    temp_list = [i for i, x in enumerate(one_time_list) if x == loop_t]
                    t_list = temp_list + t_list
                    loop_t -= 1
                    if loop_t - 1 < 0:
                        break
            if len(t_list) == 0:
                t_list = [0] * n_context
            now_index = t_list[-n_context:]
            begin_ind = now_index[0]
            end_ind = now_index[-1]
            current_history = one_act_list[begin_ind: end_ind + 1]
            if len(current_history) < n_context:
                current_history = [0] * (n_context - len(current_history)) + current_history
            user_history[u, t, :] = current_history

    return user_history


def load_dataset_timestamp(dataset_dir, activity_file_name, n_users, n_context, seq_len):
    act_list = list()
    time_list = list()
    user_list = list()

    max_timestamp = -1.0
    min_timestamp = float('inf')
    dataset_name = dataset_dir.split('/')[-1]
    with open(join(dataset_dir, activity_file_name), 'r') as raw_file:
        for line in raw_file:
            t_item_list = list()
            t_time_list = list()
            # _, user_act = line.split()
            user = int(line.split(':')[0])
            entries = line.split()[1:]
            for a_entry in entries:
                item, time_stamp = a_entry.split(':')
                t_item_list.append(int(item.strip()))
                t_time_list.append(int(time_stamp.strip()))

                if min_timestamp > int(time_stamp.strip()):
                    min_timestamp = int(time_stamp.strip())
                if max_timestamp < int(time_stamp.strip()):
                    max_timestamp = int(time_stamp.strip())

            act_list.append(t_item_list[0: seq_len])
            time_list.append(t_time_list[0: seq_len])
            user_list.append(user)

    print(max_timestamp, min_timestamp)

    new_time_list = list()
    num_bins = 0
    if dataset_name.startswith('flickr'):
        num_bins = 19
    elif dataset_name.startswith('flixster'):
        num_bins = 12
        min_seq_len = 5
    elif dataset_name.startswith('memes'):
        num_bins = 12
        min_seq_len = 5
    elif dataset_name.startswith('gowalla'):
        num_bins = 12
        min_seq_len = 25
    elif dataset_name.startswith('digg'):
        num_bins = 3
        min_seq_len = 10
    elif dataset_name == 'kaggle':
        num_bins = 12
        min_seq_len = 10

    times_bins = np.linspace(min_timestamp, max_timestamp + 1, num=num_bins, dtype=np.int32)
    for a_time_list in time_list:
        temp_time_list = (np.digitize(np.asarray(a_time_list), times_bins) - 1).tolist()
        new_time_list.append(temp_time_list)
    print(len(time_list), len(new_time_list))

    user_history = gather_user_history(act_list, new_time_list, n_users, num_bins, n_context)

    # test_lengths = list(map(len, test_act_list))
    # return train_act_list, test_act_list, train_time_list, test_time_list, train_lengths, user_history
    all_examples = []
    for i in range(0, len(act_list)):

        if len(act_list[i]) < 5:
            continue
        train_act_seq = act_list[i][:-2]
        train_time_seq = new_time_list[i][:-2]

        train_act_label = act_list[i][-2]
        train_time_label = new_time_list[i][-2]

        test_act_seq = act_list[i][1:-1]
        test_time_seq = new_time_list[i][1:-1]

        test_act_label = act_list[i][-1]
        test_time_label = new_time_list[i][-1]

        entry = {
            'train_act_seq': train_act_seq,
            'train_time_seq': train_time_seq,
            'train_act_label': train_act_label,
            'train_time_label': train_time_label,
            'test_act_seq': test_act_seq,
            'test_time_seq': test_time_seq,
            'test_act_label': test_act_label,
            'test_time_label': test_time_label,
            'seq_len': len(train_act_seq),
            'user': user_list[i]
        }

        all_examples.append(entry)

    return all_examples, user_history, num_bins


    '''train_act_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    train_time_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)

    for i in range(0, len(train_act_list)):
        train_act_list_np[i, 0: len(train_act_list[i])] = train_act_list[i]
        train_time_list_np[i, 0: len(train_time_list[i])] = train_time_list[i]

    for i in range(0, len(act_list)):
        part = int((3 / 4) * len(act_list[i]))
        assert part != 0
        train_act_list.append(act_list[i][:part])
        train_time_list.append(new_time_list[i][:part])
        test_act_list.append(act_list[i][part:])
        test_time_list.append(new_time_list[i][part:])

    train_lengths = list(map(len, train_act_list))
    test_lengths = list(map(len, test_act_list))

    print(max(train_lengths), max(test_lengths))

    train_act_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    train_time_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    test_act_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    test_time_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)

    for i in range(0, len(train_act_list)):
        train_act_list_np[i, 0: len(train_act_list[i])] = train_act_list[i]
        train_time_list_np[i, 0: len(train_time_list[i])] = train_time_list[i]

        test_act_list_np[i, 0: len(test_act_list[i])] = test_act_list[i]
        test_time_list_np[i, 0: len(test_time_list[i])] = test_time_list[i]

    train_data = np.stack([train_act_list_np.T, train_time_list_np.T]).T
    test_data = np.stack([test_act_list_np.T, test_time_list_np.T]).T
    print(train_data.shape, test_data.shape)'''

    '''train_data, temp_data, train_lengths, temp_lengths = train_test_split(data, seq_lengths, train_size=0.6)
    test_data, val_data, test_lengths, val_lengths = train_test_split(temp_data, temp_lengths, train_size=0.5)

    print(train_data.shape, test_data.shape, val_data.shape)
    print(len(train_lengths), len(test_lengths), len(val_lengths))

    data_iterator = SimpleDataIterator(train_data, np.asarray(train_lengths),
                    test_data, np.asarray(test_lengths), np.asarray(user_list))
    test_iterator = SimpleDataIterator(test_data, np.asarray(test_lengths))
    val_iterator = SimpleDataIterator(val_data, np.asarray(val_lengths))

    data_ret = dict()
    data_ret['train'] = train_iterator
    data_ret['test'] = test_iterator
    data_ret['val'] = val_iterator'''

    # return data_iterator, user_history, num_bins


def load_dataset(dataset_dir, activity_file_name):
    act_list = list()
    time_list = list()

    with open(join(dataset_dir, activity_file_name), 'r') as raw_file:
        for line in raw_file:
            t_item_list = list()
            t_time_list = list()
            # _, user_act = line.split()
            entries = line.split()[1:]
            for a_entry in entries:
                item, time_stamp = a_entry.split(':')
                t_item_list.append(int(item.strip()))
                t_time_list.append(int(time_stamp.strip()))
            act_list.append(t_item_list)
            time_list.append(t_time_list)

    # seq_lengths = list(map(len, act_list))
    # max_len = max(seq_lengths)

    # train_act_list, val_act_list, train_time_list, val_time_list = train_test_split(act_list, time_list, train_size=0.8)
    # train_seq_lengths = list(map(len, train_act_list))
    # val_seq_lengths = list(map(len, val_act_list))

    train_act_list, train_time_list = list(), list()
    test_act_list, test_time_list = list(), list()

    for i in range(0, len(act_list)):
        part = int((3 / 4) * len(act_list[i]))
        assert part != 0
        train_act_list.append(act_list[i][:part])
        train_time_list.append(time_list[i][:part])
        test_act_list.append(act_list[i][part:])
        test_time_list.append(time_list[i][part:])

    train_lengths = list(map(len, train_act_list))
    test_lengths = list(map(len, test_act_list))

    print(max(train_lengths), max(test_lengths))

    train_act_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    train_time_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    test_act_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)
    test_time_list_np = np.zeros((len(act_list), max(train_lengths)), dtype=np.int32)

    for i in range(0, len(train_act_list)):
        train_act_list_np[i, 0: len(train_act_list[i])] = train_act_list[i]
        train_time_list_np[i, 0: len(train_time_list[i])] = train_time_list[i]

        test_act_list_np[i, 0: len(test_act_list[i])] = test_act_list[i]
        test_time_list_np[i, 0: len(test_time_list[i])] = test_time_list[i]

    train_data = np.stack([train_act_list_np.T, train_time_list_np.T]).T
    test_data = np.stack([test_act_list_np.T, test_time_list_np.T]).T
    print(train_data.shape, test_data.shape)

    '''train_data, temp_data, train_lengths, temp_lengths = train_test_split(data, seq_lengths, train_size=0.6)
    test_data, val_data, test_lengths, val_lengths = train_test_split(temp_data, temp_lengths, train_size=0.5)

    print(train_data.shape, test_data.shape, val_data.shape)
    print(len(train_lengths), len(test_lengths), len(val_lengths))'''

    data_iterator = SimpleDataIterator(train_data, np.asarray(train_lengths), test_data, np.asarray(test_lengths))
    '''test_iterator = SimpleDataIterator(test_data, np.asarray(test_lengths))
    val_iterator = SimpleDataIterator(val_data, np.asarray(val_lengths))

    data_ret = dict()
    data_ret['train'] = train_iterator
    data_ret['test'] = test_iterator
    data_ret['val'] = val_iterator'''

    return data_iterator


def get_logger(log_filename, name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filename, 'w+')
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    # logger.info('Log directory: %s', log_dir)
    return logger


def get_log_dir(log_dir, dataset_name, model_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, '{}_{}.log'.
                    format(dataset_name, model_name))
    return log_file_name
