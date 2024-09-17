import tensorflow as tf

flags = tf.compat.v1.flags
# flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('architecture', 'decentralized', 'Model architecture.')  # centralized, decentralized
flags.DEFINE_string('optimizer', 'GD', 'Optimizer.')  # GD, Adam
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_decay', 1.0, 'Initial learning rate.')
flags.DEFINE_integer('num_nodes', 100, 'Desired number of nodes.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('feature_size', 1, 'Number of units input feature.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 1, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 5, 'number of layers.')
flags.DEFINE_integer('num_cons', 100, 'Number of consensus rounds.')

flags.DEFINE_float('backoff_prob', 0.3, 'Probability of saving intermediate node on search tree.')
flags.DEFINE_integer('diver_out', 32, 'Number of predictors in test.')
flags.DEFINE_integer('timeout', 300, 'Total Seconds to Run each case.')
flags.DEFINE_string('datapath','../twin-nphard/data/ER_Graph_Uniform_mixN_mixp_train0', 'Location of Data')
flags.DEFINE_float('snr_db', 10, 'ratio of powers between weights and noise (dB).')
flags.DEFINE_string('training_set', 'IS4SAT', 'Name of training dataset')
flags.DEFINE_integer('greedy', 0, 'Normal: 0, Greedy: 1, Noisy Greedy: 2')
flags.DEFINE_bool('skip', False, 'If skip connection included')
flags.DEFINE_string('wts_init', 'random', 'how to initialize the weights of GCN')
flags.DEFINE_string('snapshot', '', 'snapshot of model')
flags.DEFINE_string('predict', 'mwis', 'direct output: mwis, linear combination: mis')
