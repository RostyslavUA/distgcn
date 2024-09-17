# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
# add the libary path for graph reduction and local search
# sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )

import time
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from multiprocessing import Queue
from copy import deepcopy

import tensorflow as tf
from collections import deque
from gcn.models import GCN_DQN
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
# Settings (FLAGS)
from runtime_config import *
from test_utils import *
from heuristics import *

tf.compat.v1.disable_eager_execution()
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_GEN21_test2', 'test dataset')
flags.DEFINE_float('epsilon', 1.0, 'test dataset')
flags.DEFINE_float('epsilon_min', 0.001, 'test dataset')
# test data path
data_path = FLAGS.datapath
test_datapath = FLAGS.test_datapath
val_mat_names = sorted(os.listdir(data_path))
test_mat_names = sorted(os.listdir(test_datapath))
val_mat_names = [name for name in val_mat_names if f'n{FLAGS.num_nodes}' in name]
test_mat_names = [name for name in test_mat_names if f'n{FLAGS.num_nodes}' in name]

# Some preprocessing
noout = min(FLAGS.diver_num, FLAGS.diver_out) # number of outputs
time_limit = FLAGS.timeout  # time limit for searching
backoff_thresh = 1 - FLAGS.backoff_prob

num_supports = 1 + FLAGS.max_degree
model_func = GCN_DQN
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

from directory import create_result_folder, find_model_folder
outputfolder = create_result_folder(FLAGS, 'dqn')
model_origin = find_model_folder(FLAGS, 'dqn')

# for graph reduction and local search
# api = reducelib()


def add_rnd_q(cns, nIS_vec_local):
    global adj_0

    nIS_vec_local[cns] = 1
    tmp = sp.find(adj_0[cns, :] == 1)
    nIS_vec_local[tmp[1]] = 0
    remain_vec_tmp = (nIS_vec_local == -1)
    adj = adj_0
    adj = adj[remain_vec_tmp, :]
    adj = adj[:, remain_vec_tmp]
    if reduce_graph(adj, nIS_vec_local):
        return True
    return False

def reduce_graph(adj, nIS_vec_local):
    global best_IS_num
    global best_IS_vec
    global bsf_q
    global adj_0
    global q_ct
    global id
    global out_id
    global res_ct

    global wts
    global yy_util
    global best_IS_util

    remain_vec = (nIS_vec_local == -1)

    # reduce graph
    # reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = api.reduce_graph(adj)
    reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = fake_reduce_graph(adj)
    nIS_vec_sub = reduced_node.copy()
    nIS_vec_sub_tmp = reduced_node.copy()
    nIS_vec_sub[nIS_vec_sub_tmp == 0] = 1
    nIS_vec_sub[nIS_vec_sub_tmp == 1] = 0
    reduced_nn = reduced_adj.shape[0]

    # update MIS after reduction
    tmp = sp.find(adj[nIS_vec_sub == 1, :] == 1)
    nIS_vec_sub[tmp[1]] = 0
    nIS_vec_local[remain_vec] = nIS_vec_sub
    nIS_vec_local[nIS_vec_local == 2] = -1
    wts_nn = wts[remain_vec]

    # if the whole graph is reduced, we find a candidate
    if reduced_nn == 0:
        remain_vec_tmp = (nIS_vec_local == -1)
        if np.sum(remain_vec_tmp) == 0:
            # get a solution
            res_ct += 1
            # nIS_vec_local = api.local_search(adj_0, nIS_vec_local)
            nIS_vec_local = fake_local_search(adj_0, nIS_vec_local)
            nIS_util = np.dot(nIS_vec_local, wts)
            # if np.sum(nIS_vec_local) > best_IS_num:
            if nIS_util > best_IS_util:
                best_IS_num = np.sum(nIS_vec_local)
                best_IS_vec = deepcopy(nIS_vec_local)
                best_IS_util = nIS_util
                sio.savemat(os.path.join(outputfolder, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec, 'weights': wts, 'best_util': best_IS_util, 'yy_util': yy_util})
            # print("ID: %03d" % id, "QItem: %03d" % q_ct, "Res#: %03d" % res_ct,
            #       "Current: %d" % (np.sum(nIS_vec_local)), "Best: %d" % best_IS_num, "Reduction")
            print("ID: %03d" % id, "File: {}".format(val_mat_names[id]),
                  "Current: %d" % (np.sum(nIS_vec_local)), "Best: %d" % best_IS_num,
                  "Epsilon: {}".format(dqn_agent.epsilon), "Best Utility: {}".format(best_IS_util),
                  "Y Utility: {}".format(yy_util), "Ratio: {}".format(best_IS_util / yy_util))
            return True
        adj = adj_0
        adj = adj[remain_vec_tmp, :]
        adj = adj[:, remain_vec_tmp]
        wts_nn = wts[remain_vec_tmp]
        bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy(), wts_nn.copy()])
    else:
        bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy(), wts_nn.copy()])

    return False


def weighted_random_graph(N, p, dist, maxWts=1.0):
    graph = nx.generators.random_graphs.fast_gnp_random_graph(N,p)
    if dist.lower() == 'uniform':
        for u in graph:
            graph.nodes[u]['weight'] = np.random.uniform(0,maxWts)
    elif dist.lower() == 'normal_l1':
        for u in graph:
            graph.nodes[u]['weight'] = np.abs(np.random.randn())
    elif dist.lower() == 'normal_l2':
        for u in graph:
            graph.nodes[u]['weight'] = np.square(np.random.randn())

    return graph


class DQNAgent:
    def __init__(self, feature_size=32, architecture='centralized', memory_size=5000):
        self.feature_size = feature_size
        self.architecture = architecture
        self.memory = deque(maxlen=memory_size)
        self.rewards = deque(maxlen=memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = FLAGS.epsilon  # exploration rate
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon_decay = 0.985
        self.learning_rate = FLAGS.learning_rate
        self.sess = tf.compat.v1.Session(config=config)
        self.model = self._build_model()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = model_func(placeholders, input_dim=self.feature_size, architecture=self.architecture,
                           optimizer=FLAGS.optimizer, logging=True)
        return model

    def memorize(self, state, act_vals, solu, wts_nn, reward):
        self.memory.append((state.copy(), act_vals.copy(), solu.copy(), wts_nn.copy(), reward))
        self.rewards.append(reward)

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        features = np.multiply(np.ones([reduced_nn, self.feature_size]), wts_nn)
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)
        state = {"features": features, "support": support}
        if self.architecture == 'decentralized':
            cons_mat = consensus_matrix(adj)
            cons_mat = sparse_to_tuple(cons_mat)
            state = {**state, **{'cons_mat': cons_mat}}
        return state

    def predict(self, state):
        feed_dict_val = construct_feed_dict4pred(state["features"], state["support"], placeholders)
        act_values, action = self.sess.run([self.model.outputs_softmax, self.model.pred], feed_dict=feed_dict_val)
        return act_values, action

    def act(self, state):
        act_values, action = self.predict(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(act_values.size)
        return action  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        losses = []
        for state, act_vals, solu, wts_nn, reward in minibatch:
            target = reward
            # if not done:
            #     act_values, _ = self.predict(next_state)
            #     target = (reward + self.gamma * np.amax(act_values))
            # target_f, _ = self.predict(state)
            target_f = np.reshape(act_vals, (act_vals.size, 1))
            # target_f = -np.ones((act_vals.size, 1))
            target_f[solu] = target + wts_nn
            # Filtering out states and targets for training
            # _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=state)
            # losses.append(loss)
            states.append(state.copy())
            targets_f.append(target_f)

        for i in range(len(states)):
            state = states[i]
            target_f = targets_f[i]
            feed_dict = construct_feed_dict(state['features'], state['support'], target_f, placeholders)
            if self.architecture == 'decentralized':
                feed_dict.update({placeholders['cons_mat']: state['cons_mat']})
            _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
            losses.append(loss)
            # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.nanmean(losses)

    def load(self, name):
        ckpt = tf.train.get_checkpoint_state(name)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded ' + ckpt.model_checkpoint_path)

    def save(self, name):
        self.saver.save(self.sess, os.path.join(name, "model.ckpt"))


def solve_mwis(test=False):
    buffer = deque(maxlen=20)
    q_ct = 0
    best_IS_util = np.array([0.0])
    reduced_nn = adj_0.shape[0]
    reduce_graph(adj_0, -np.ones(nn))

    q_item = bsf_q.pop(0)
    q_ct += 1

    adj = q_item[0]
    remain_vec = deepcopy(q_item[2])
    reduced_adj = q_item[3]
    reverse_mapping = deepcopy(q_item[4])
    remain_nn = adj.shape[0]
    reduced_nn = reduced_adj.shape[0]
    wts_nn = q_item[5]

    # GCN
    state = dqn_agent.makestate(reduced_adj, wts_nn)
    act_vals, act = dqn_agent.predict(state)
    if not test:
        if np.random.rand() <= dqn_agent.epsilon:
            act_vals = np.random.uniform(size=act_vals.size)

    if FLAGS.predict == 'mwis':
        gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
    else:
        gcn_wts = act_vals.flatten()
    # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

    mwis, total_wt = greedy_search(adj, gcn_wts)

    total_wt = np.sum(wts_nn[list(mwis), 0])
    reward = total_wt/(greedy_util.flatten()[0])
    wts_norm = wts_nn[list(mwis), :]/greedy_util.flatten()

    buffer.append((state.copy(), act_vals, list(mwis), wts_norm, reward))

    return total_wt, buffer


def solve_mwis_iterative(test=False):
    # buffer = deque(maxlen=20)
    q_ct = 0
    best_IS_util = np.array([0.0])
    reduced_nn = adj_0.shape[0]
    reduce_graph(adj_0, -np.ones(nn))
    while reduced_nn > 0:
        if len(bsf_q) == 0:
            break
        q_item = bsf_q.pop(0)
        q_ct += 1

        # adj = q_item[0]
        remain_vec = deepcopy(q_item[2])
        reduced_adj = q_item[3]
        # reverse_mapping = deepcopy(q_item[4])
        # remain_nn = adj.shape[0]
        reduced_nn = reduced_adj.shape[0]
        wts_nn = q_item[5]

        # GCN
        state = dqn_agent.makestate(reduced_adj, wts_nn)
        act_vals, act = dqn_agent.predict(state)
        # if not test:
        #     if np.random.rand() <= dqn_agent.epsilon:
        #         act_vals = np.random.uniform(size=act_vals.size)

        if FLAGS.predict == 'mwis':
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        else:
            gcn_wts = act_vals.flatten()

        act = np.argmax(gcn_wts)
        remain_vtx, = np.where(remain_vec)
        cns = remain_vtx[act]

        nIS_vec = deepcopy(q_item[1])
        nIS_vec[cns] = 1
        tmp = sp.find(adj_0[cns, :] == 1)
        nIS_vec[tmp[1]] = 0
        remain_vec_tmp = (nIS_vec == -1)

        adj = adj_0
        adj = adj[remain_vec_tmp, :]
        adj = adj[:, remain_vec_tmp]
        # next_state = dqn_agent.makestate(adj, wts[remain_vec_tmp])
        # reward = wts_nn[act, 0]/(greedy_util.flatten()[0])

        if np.sum(remain_vec_tmp) == 0:
            # get a solution
            # nIS_vec = api.local_search(adj_0, nIS_vec)
            # nIS_vec = fake_local_search(adj_0, nIS_vec)
            best_IS_util = np.dot(nIS_vec, wts)
            break

        if reduce_graph(adj, nIS_vec):
            continue
    return best_IS_util, q_ct


N_bd = FLAGS.feature_size

# Define placeholders
num_nodes = None if FLAGS.architecture == 'centralized' else FLAGS.num_nodes
placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float64, shape=(None, num_nodes)) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float64, shape=(num_nodes, N_bd)), # featureless: #points
    'labels': tf.compat.v1.placeholder(tf.float64, shape=(None, 1)), # 0: not linked, 1:linked
    'labels_mask': tf.compat.v1.placeholder(tf.int64),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int64)  # helper variable for sparse dropout
}
if FLAGS.architecture == 'decentralized':
    placeholders.update({'cons_mat': tf.compat.v1.sparse_placeholder(tf.float64, shape=(None, num_nodes))})

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Create model
dqn_agent = DQNAgent(N_bd, FLAGS.architecture, 5000)
try:
    dqn_agent.load(model_origin)
except:
    print("Unable to load {}".format(model_origin))

best_IS_vec = []
loss_vec = []

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0
batch_size = 100

best_ratio = 1.0
for epoch in range(FLAGS.epochs):
    losses = []
    cnt = 0
    f_ct = 0
    q_totals = []
    p_ratios = []
    newtime = time.time()
    for id in np.random.permutation(len(val_mat_names)):
    # for id in range(10000):
        best_IS_num = -1
        # print(val_mat_names[id])
        mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
        adj_0 = mat_contents['adj']
        if not nx.is_connected(nx.Graph(adj_0)):
            continue
        wts = mat_contents['weights'].transpose()
        # yy_util = mat_contents['mwis_utility']
        # greedy_util = mat_contents['greedy_utility']
        # N = np.random.choice([100,150,200,250,300])
        # p = np.random.choice([0.02,0.05,0.075,0.10,0.15])
        # graph = weighted_random_graph(N, p, 'Uniform')
        # adj_0 = nx.adj_matrix(graph)
        # wts = np.array([graph.nodes[u]['weight'] for u in graph.nodes])
        # wts = np.reshape(wts, (wts.size, 1))
        _, greedy_util = greedy_search(adj_0, wts)
        nn = adj_0.shape[0]
        bsf_q = []
        q_ct = 0
        res_ct = 0
        out_id = -1

        start_time = time.time()
        ss_util, buffer = solve_mwis(test=False)
        p_ratio = ss_util.flatten()/greedy_util.flatten()
        # if p_ratio[0] > 0.9:
        for state, act_vals, solu, wts_nn, reward in buffer:
            dqn_agent.memorize(state, act_vals, solu, wts_nn, reward)
            # if (len(dqn_agent.rewards) < 500) or (np.nanmean(dqn_agent.rewards) <= reward):
            #     dqn_agent.memorize(state, act_vals, solu, wts_nn, reward)
        f_ct += 1
        q_totals.append(len(solu))
        p_ratios.append(p_ratio[0])
        if cnt < batch_size - 1:
            cnt += 1
            continue
        else:
            cnt = 0
            runtime = time.time() - newtime
            newtime = time.time()

        avg_is_size = np.mean(q_totals)

        # # q_totals = []
        # for i in range(1):
        #     loss = dqn_agent.replay(2000)
        # print("Epoch: {}".format(epoch), "ID: %05d" % f_ct, "Avg_IS_Size: {:.4f}".format(avg_is_size),
        #       "Epsilon: {:.6f}".format(dqn_agent.epsilon), "Ratio: {:.6f}".format(p_ratio[0]),
        #       "Loss: {:.6f}".format(loss), "Epoch_Loss: {:.6f}".format(np.mean(losses)), "Epoch_Ratio: {:.6f}".format(np.mean(p_ratios)), "runtime: {:.3f}".format(runtime))
        test_ratio=[]
        for j in range(len(test_mat_names)):
            mat_contents = sio.loadmat(test_datapath + '/' + test_mat_names[j])
            adj_0 = mat_contents['adj']
            wts = mat_contents['weights'].transpose()
            # yy_util = mat_contents['mwis_utility']
            # greedy_util = mat_contents['greedy_utility']
            _, greedy_util = greedy_search(adj_0, wts)
            nn = adj_0.shape[0]
            bsf_q = []
            q_ct = 0
            res_ct = 0
            out_id = -1
            best_IS_util,_ = solve_mwis(test=True)
            # best_IS_util, q_ct = solve_mwis_iterative(test=True)
            # test_ratio.append(best_IS_util / yy_util.flatten()[0])
            test_ratio.append(best_IS_util / greedy_util)

        if np.mean(test_ratio) > best_ratio:
            dqn_agent.save(model_origin)
            best_ratio = np.mean(test_ratio)

        loss = dqn_agent.replay(batch_size)
        # loss = dqn_agent.replay(100)
        if loss is None:
            loss = 1.0
        losses.append(loss)

        print("Epoch: {}".format(epoch),
              "ID: %03d" % f_ct,
              "Train_Ratio: {:.6f}".format(np.mean(p_ratios)),
              "Epsilon: {:.6f}".format(dqn_agent.epsilon),
              "Test_Ratio: {:.6f}".format(np.mean(test_ratio)),
              "Loss: {:.6f}".format(loss),
              "Epoch_Loss: {:.6f}".format(np.mean(losses)),
              "runtime: {:.3f}".format(runtime),
              "mem_val: {:.3f}".format(np.nanmean(dqn_agent.rewards)))
        p_ratios = []

    # dqn_agent.epsilon_min = dqn_agent.epsilon_min * 0.1
    loss_vec.append(np.mean(losses))
    if epoch+1 in epislon_reset:
        epislon_val = epislon_val*0.2
        dqn_agent.epsilon = epislon_val
        # sio.savemat('./%s/%s' % (outputfolder, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec, 'weights': wts, 'best_util': best_IS_util, 'yy_util': yy_util})
print(loss_vec)

