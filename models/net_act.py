
import numpy as np
import tensorflow as tf

import utils


class NetworkActivityModel(object):
    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def __init__(self, adj_mx, params):
        self.adj_mx = adj_mx
        self.batch_size = params['batch_size']
        self.user_size = params['user_size']
        self.item_size = params['item_size']
        self.state_size = params['state_size']
        self.emb_size = params['emb_size']
        self.num_bins = params['n_bins']
        self.context_size = params['context_size']
        self.max_diffusion_step = params['max_diff']
        self.seq_length = params['seq_len']
        self.lr = params['lr']
        self.training_steps_per_epoch = params['start_lr']
        self.min_lr = params['min_lr']
        self.is_norm = params['normalize']
        self.comb_type = params['comb']
        self.supports = list()
        if params['n_samples'] == -1:
            self.n_samples = self.item_size
        else:
            self.n_samples = params['n_samples']

        self.var_initializer = tf.contrib.layers.xavier_initializer()
        self.model_input = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_length, 2])
        self.model_output = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.user_history_ph = tf.placeholder(tf.int32, shape=[self.user_size, self.num_bins, self.context_size])
        self.user_list = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.seqlen = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.test_item = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_samples])

        self.dropout_prob = tf.placeholder(tf.float32)
        self.conv_output_size = self.state_size  # * 2
        # self.cell_type = tf.nn.rnn_cell.LSTMCell
        self.cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell

        supports = list()
        supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))

        for support in supports:
            self.supports.append(self._build_sparse_matrix(support))
        self.num_matrices = len(self.supports) * self.max_diffusion_step  # + 1    # Adds for x itself.

        self.item_emb = tf.get_variable('item_emb',
                        initializer=self.var_initializer(shape=[self.item_size, self.emb_size]))
        # self.rnn_unit = self.cell_type(self.state_size, activation=tf.nn.tanh)
        self.rnn_unit = self.cell_type(self.state_size, layer_norm=True,
                                 dropout_keep_prob=self.dropout_prob)

        with tf.variable_scope('conv_vars'):
            self.conv_w = tf.get_variable('conv_weights', [self.emb_size * self.num_matrices, self.conv_output_size],
                        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.conv_b = tf.get_variable('conv_b', [self.conv_output_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        self.context_vector = tf.get_variable('context_vector', shape=[self.context_size, self.emb_size, self.emb_size],
                                              dtype=tf.float32, initializer=self.var_initializer)

        self.user_history_emb = tf.nn.embedding_lookup(self.item_emb, tf.cast(self.user_history_ph, dtype=tf.int32),
                                                       name='user_history_emb')

        if params['use_attn']:
            self.use_attn = True
            self.attention_size = self.state_size // 2
            self.W_omega = tf.Variable(tf.random_normal([self.state_size, self.attention_size], stddev=0.1))
            self.b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            self.u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        else:
            self.use_attn = False

    def gconv_overall(self, timestamp, output_size):
        x = self.user_emb[:, timestamp, :]
        x0 = tf.reshape(x, shape=[self.user_size, self.emb_size])
        # x = tf.expand_dims(x0, axis=0)

        for support in self.supports:
            # x1 = tf.sparse_tensor_dense_matmul(support, x0)
            # x = self._concat(x, x1)
            # x = tf.reshape(tf.transpose(x, perm=[1, 2, 0]), shape=[self.user_size, -1])
            # x1 = tf.matmul(x0, self.conv_w)
            # x1 = tf.nn.bias_add(x1, self.conv_b)
            x1 = tf.sparse_tensor_dense_matmul(support, x0)
            x = x1
            if self.max_diffusion_step > 1:
                x = tf.expand_dims(x, axis=0)

            for k in range(2, self.max_diffusion_step + 1):
                x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

            if self.max_diffusion_step > 1:
                x = tf.transpose(x, [1, 0, 2])
                x = tf.reshape(x, [self.user_size, -1])

            x = tf.matmul(x, self.conv_w)
            x = tf.nn.bias_add(x, self.conv_b)

        return tf.reshape(x, [self.user_size, output_size])

    def attention(self, states):
        v = tf.tanh(tf.tensordot(states, self.W_omega, axes=[[2], [0]]) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(states * tf.expand_dims(alphas, -1), 1)
        return output

    def build_graph(self):
        self.user_emb = tf.reshape(tf.tensordot(tf.reshape(self.user_history_emb, [self.user_size, self.num_bins, -1]),
                        tf.reshape(self.context_vector, [-1, self.emb_size]), axes=[2, 0]),
                        [self.user_size, self.num_bins, -1])
        self.user_emb_conv = list()
        for t in range(0, self.num_bins):
            self.user_emb_conv.append(self.gconv_overall(t, output_size=self.state_size))   # * 2
        self.user_emb_conv = tf.transpose(tf.stack(self.user_emb_conv), perm=[1, 0, 2])

        self.rnn_input_emb = tf.nn.embedding_lookup(self.item_emb, tf.cast(self.model_input[:, :, 0], dtype=tf.int32),
                                                    name='item_emb')
        self.rnn_user_emb = list()
        for b in range(0, self.batch_size):
            temp_emb = tf.gather(self.user_emb_conv[self.user_list[b], ...], self.model_input[b, :, 1])
            self.rnn_user_emb.append(temp_emb)
        self.rnn_user_emb = tf.convert_to_tensor(self.rnn_user_emb)

        if self.comb_type == 'add':
            self.comb_rnn_input = tf.add(self.rnn_input_emb, self.rnn_user_emb)
        elif self.comb_type == 'mult':
            self.comb_rnn_input = tf.multiply(self.rnn_input_emb, self.rnn_user_emb)
        else:
            self.comb_rnn_input = tf.concat([self.rnn_input_emb, self.rnn_user_emb], axis=2)

        if self.is_norm:
            self.comb_rnn_input = normalize(self.comb_rnn_input, scope='rnn_in')

        self.init_state = self.rnn_unit.zero_state(self.batch_size, dtype=tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(self.rnn_unit, self.comb_rnn_input,
                                        initial_state=self.init_state, sequence_length=self.seqlen)
        if self.use_attn:
            self.rnn_output_flattened = self.attention(self.rnn_outputs)
        else:
            self.rnn_output_flattened = tf.reshape(self.final_state.h, [-1, self.rnn_outputs.shape[-1].value])
        self.item_loss = self.calc_item_loss(self.rnn_output_flattened,
                    tf.reshape(self.model_output, [-1]), self.item_size)

        tv = tf.trainable_variables()
        self.reg_loss = tf.constant(0.0005) * tf.reduce_mean([tf.nn.l2_loss(v) for v in tv])
        # self.cost += tf.constant(0.0005) * self.reg_loss

        self.total_loss = self.item_loss + self.reg_loss

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        starter_learning_rate = self.lr
        # decay per training epoch
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,
            global_step,
            self.training_steps_per_epoch,
            0.97,
            staircase=True)
        learning_rate = tf.maximum(learning_rate, self.min_lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def calc_item_loss(self, rnn_output, item_label, item_size):
        self.item_out = tf.layers.dense(rnn_output, item_size, trainable=True, use_bias=True,
                        kernel_initializer=self.var_initializer, activation=tf.nn.relu,
                        bias_initializer=tf.constant_initializer(0.25))

        temp_logits = list()
        for b in range(0, self.batch_size):
            temp_logits.append(tf.gather(self.item_out[b, :], self.test_item[b, :]))
        self.test_logits = tf.convert_to_tensor(temp_logits)
        self.test_probs = tf.nn.softmax(self.test_logits)

        if self.is_norm:
            self.item_out = normalize(self.item_out)

        self.probs = tf.nn.softmax(self.item_out)
        self.pred_item = tf.reshape(tf.argmax(self.item_out, axis=1), shape=[self.batch_size, -1])
        one_hot_item = tf.one_hot(tf.cast(item_label, dtype=tf.int32), item_size)
        item_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.item_out, labels=one_hot_item)

        return tf.reduce_sum(item_loss)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs
