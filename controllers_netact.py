import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import jaccard_similarity_score

import utils
import metrics2 as metrics
# from models.net_act_old import NetworkActivityModel
# from models.netAct import NetworkActivityModel
from models.net_act import NetworkActivityModel


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def hits_k_ind(yscores, y, user_list, k=10):
    ret = list()
    for y, y_score, user in zip(y, yscores, user_list):
        rank_actual = (-1 * y_score).argsort().argsort()[0]
        if rank_actual < k:
            ret.append(user)

    return ret


class NetActController(object):
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.dataset_name = self._data_kwargs['dataset_dir'].split('/')[-1]
        self.adj_mx = adj_mx
        self.model_params = dict()
        self.model_params['seq_len'] = 30
        self.K = [1, 5, 10, 20, 50, 100]

        model_name = 'net_act_orig'  # self._kwargs['model_name']
        self.log_file_name = utils.get_log_dir(log_dir=self._kwargs['log_dir'],
                            model_name=model_name, dataset_name=self.dataset_name)
        if not os.path.exists(self._kwargs['save_dir']):
            os.makedirs(self._kwargs['save_dir'])
        if not os.path.exists(os.path.join(self._kwargs['save_dir'], self.dataset_name)):
            os.makedirs(os.path.join(self._kwargs['save_dir'], self.dataset_name))
        if not os.path.exists(os.path.join(self._kwargs['save_dir'], self.dataset_name, self._kwargs['model_name'])):
            os.makedirs(os.path.join(self._kwargs['save_dir'], self.dataset_name, self._kwargs['model_name']))

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self.log_file_name, name=__name__, level=log_level)
        self._writer = tf.summary.FileWriter(self.log_file_name)
        self._logger.info(json.dumps(kwargs, indent=2))
        self._saved_file_name = 'best_model.ckpt'

        user_id, reverse_user_id, item_id, reverse_item_id = \
            utils.load_ids(self._data_kwargs['dataset_dir'], self._data_kwargs['ids_file_name'])
        print(len(user_id), len(reverse_user_id), len(item_id), len(reverse_item_id))

        self.n_users = len(user_id)
        self.n_context = self._model_kwargs['context_size']

        data_examples, self.user_history, num_bins = utils.load_dataset_timestamp(self._data_kwargs['dataset_dir'],
                        self._data_kwargs['dataset_name'], self.n_users, self.n_context, self.model_params['seq_len'])
        self.num_bins = num_bins

        self.model_params['batch_size'] = self._data_kwargs['batch_size']
        self.model_params['user_size'] = self.n_users
        self.model_params['item_size'] = len(item_id)
        self.model_params['state_size'] = self._model_kwargs['state_size']
        self.model_params['emb_size'] = self._model_kwargs['emb_size']
        self.model_params['lr'] = self._train_kwargs['base_lr']
        self.model_params['n_bins'] = self.num_bins
        self.model_params['context_size'] = self.n_context
        self.model_params['start_lr'] = len(data_examples) // self._data_kwargs['batch_size']
        self.model_params['min_lr'] = self._train_kwargs['min_learning_rate']
        self.model_params['use_attn'] = self._model_kwargs['use_attn']
        self.model_params['normalize'] = self._model_kwargs['normalize']
        self.model_params['max_diff'] = self._model_kwargs['max_diff']
        if self._model_kwargs['n_samples'] == -1:
            self.model_params['n_samples'] = len(item_id)
        else:
            self.model_params['n_samples'] = self._model_kwargs['n_samples']
        self.model_params['comb'] = self._model_kwargs['comb']

        self.data_iterator = utils.Loader(data_examples, options=self.model_params)

    def initiate_model(self):
        if self._kwargs['model_name'] == 'netact':
            self.model = NetworkActivityModel(self.adj_mx, params=self.model_params)
            # self.model = NetworkActivityModel(self.adj_mx, params=self.model_params)

    def run_train(self):
        print('training Network Activity model')

        self.all_users_5 = list()
        self.all_users_10 = list()
        self.all_users_20 = list()

        with tf.Session() as sess:
            num_batch = len(self.data_iterator)
            # self.data_iterator.size // self._data_kwargs['batch_size']
            self.initiate_model()
            self.model.build_graph()

            self.model_saver = tf.train.Saver()
            if self._kwargs['is_load']:
                self.model_saver.restore(sess, os.path.join(self._kwargs['save_dir'],
                self.dataset_name, self._kwargs['model_name'], self._saved_file_name))
                self._logger.info('Start retraining total # of batch {}'.format(num_batch))
                '''self.user_states = np.load(os.path.join(self._kwargs['save_dir'],
                    self.dataset_name, self._kwargs['model_name'], 'user_state.npy'))'''
            else:
                tf.global_variables_initializer().run(session=sess)
                self._logger.info('Start training total # of batch {}'.format(num_batch))

            score_keys = ['map@{}'.format(k) for k in self.K] + ['hits@{}'.format(k) for k in self.K] + \
                         ['ndcg@{}'.format(k) for k in self.K]
            '''best_scores = {'map@10': 0.0, 'map@50': 0.0, 'map@100': 0.0,
                           'hits@10': 0.0, 'hits@50': 0.0, 'hits@100': 0.0}'''
            best_test_scores = {key: 0.0 for key in score_keys}
            for iter in range(1, self._train_kwargs['n_epochs'] + 1):
                global_cost = 0.0

                for j in range(0, num_batch):
                    '''train_res, train_seqlen, test_res, test_seqlen, user_list = \
                        self.data_iterator.next_batch(n=self._data_kwargs['batch_size'])'''
                    train_input, test_input, train_time, test_time, \
                    train_label, test_label, _, _, user_list, seq_len = self.data_iterator()

                    if train_input.shape[0] < self.data_iterator.batch_size:
                        continue
                    comb_input = np.concatenate([np.expand_dims(train_input, axis=-1),
                                                 np.expand_dims(train_time, axis=-1)], axis=2)
                    fd = {
                        self.model.model_input: comb_input,
                        self.model.model_output: train_label,
                        self.model.seqlen: seq_len,
                        self.model.user_history_ph: self.user_history,
                        self.model.user_list: user_list,
                        self.model.dropout_prob: self._model_kwargs['dropout']
                    }
                    '''self.model.init_state: np.zeros((2, self.model_params['batch_size'],
                                                                self.model_params['state_size']), dtype=np.float32)'''
                    _, total_cost, last_state = sess.run([self.model.train_op, self.model.total_loss,
                                        self.model.final_state], feed_dict=fd)
                    if np.isnan(total_cost):
                        print('cost is nan')
                        exit(1)
                    # self.user_states[:, user_list, :] = last_state
                    global_cost += total_cost

                global_cost /= num_batch
                self._logger.info('Epoch: {:d}, Global Cost: {:.4f}'.format(iter, global_cost))

                if iter % self._train_kwargs['test_interval'] == 0:
                    test_scores, inds_all = self.test_model(sess)
                    if best_test_scores['hits@10'] < test_scores['hits@10']:
                        best_test_scores = test_scores
                        self._logger.info(
                            'Epoch: {:d}, Found New Best Test Performance: {}'.format(iter, json.dumps(test_scores)))
                        self.save_model(sess)
                        self.all_users_5 = inds_all[0].copy()
                        self.all_users_10 = inds_all[1].copy()
                        self.all_users_20 = inds_all[2].copy()
                    else:
                        self._logger.info('Epoch: {:d}, Current Performance: {}'.format(iter, json.dumps(test_scores)))
        with open('pred_user_{}.pkl'.format(self.dataset_name), 'wb+') as write_file:
            pickle.dump([self.all_users_5, self.all_users_10, self.all_users_20], write_file) 

    def test_model(self, sess):
        # self.data_iterator.shuffle_data()
        node_scores = list()
        inds_5, inds_10, inds_20 = list(), list(), list()
        self.data_iterator.reset()
        num_batch = len(self.data_iterator)  # .size // self._data_kwargs['batch_size']

        for j in range(0, num_batch):
            test_batch = self.data_iterator()
            if test_batch[0].shape[0] < self.data_iterator.batch_size:
                continue
            else:
                node_score, inds = self.evaluate_batch(sess, test_batch)
                node_scores.append(node_score)
                inds_5.extend(inds[0])
                inds_10.extend(inds[1])
                inds_20.extend(inds[2])

        final_score = self.get_average_score(node_scores)

        return final_score, tuple((inds_5, inds_10, inds_20))

    def evaluate_batch(self, sess, test_batch):
        y = None
        y_prob = None
        train_input, test_input, train_time, test_time, \
        train_label, test_label, _, _, user_list, seq_len = test_batch
        y_ = test_label
        comb_input = np.concatenate([np.expand_dims(test_input, axis=-1),
                                     np.expand_dims(test_time, axis=-1)], axis=2)
        cand_labels = list()
        assert len(user_list) == len(test_label), "Error in data preprocessing"

        for b in range(self.model_params['batch_size']):
            user_cand = [test_label[b]]
            for ci in range(self.model_params['n_samples'] - 1):
                user_cand.append(random_neq(1, self.model_params['item_size'], [test_label[b]]))

            cand_labels.append(user_cand)
        cand_labels = np.asarray(cand_labels)

        fd_test = {
            self.model.model_input: comb_input,
            self.model.model_output: test_label,
            self.model.seqlen: seq_len,
            self.model.user_history_ph: self.user_history,
            self.model.user_list: user_list,
            self.model.test_item: cand_labels,
            self.model.dropout_prob: 1.0
        }
        '''self.model.init_state: np.zeros((2, self.model_params['batch_size'],
                                    self.model_params['state_size']), dtype=np.float32)'''

        y_prob_ = sess.run(self.model.test_probs, feed_dict=fd_test)
        # if y_prob is None:
        y_prob = y_prob_
        y = y_
        '''else:
            y = np.concatenate((y, y_), axis=0)
            y_prob = np.concatenate((y_prob, y_prob_), axis=0)'''
        # node_score = metrics.portfolio(y_prob, y, k_list=self.K)
        node_score = metrics.Metrics_Sasrec.portfolio(y_prob, y, k_list=self.K)

        ind_5 = hits_k_ind(y_prob, y, user_list, k=5)
        ind_10 = hits_k_ind(y_prob, y, user_list, k=10)
        ind_20 = hits_k_ind(y_prob, y, user_list, k=20)

        return node_score, tuple((ind_5, ind_10, ind_20))

    def get_average_score(self, scores):
        df = pd.DataFrame(scores)
        return dict(df.mean())

    def save_model(self, sess):
        self.model_saver.save(sess, save_path=os.path.join(self._kwargs['save_dir'],
            self.dataset_name, self._kwargs['model_name'], self._saved_file_name))
        '''np.save(file=os.path.join(self._kwargs['save_dir'], self.dataset_name,
                self._kwargs['model_name'], 'user_state.npy'), arr=self.user_states)'''

    def readjust_pred(self, input):
        nz_arr = [input[i] for i in range(0, input.shape[0]) if input[i] > 0]
        z_arr = [0] * (input.shape[0] - len(nz_arr))

        return np.asarray(nz_arr + z_arr)