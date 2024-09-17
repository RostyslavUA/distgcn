import numpy.random
from inits import *
tf.compat.v1.disable_eager_execution()
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape, seed=42):
    """Dropout for sparse tensors."""
    tf.random.set_seed(seed)
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape, seed=seed, dtype=tf.float64)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = tf.cast(placeholders['dropout'], tf.float64)
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                            name='weights_' + str(i), seed=42)
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(i)] = zeros([input_dim, output_dim],
                                                            name='weights_' + str(i))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        # concated = tf.concat([output, self.act(output)], axis=1)
        # return tf.layers.dense(concated, output.shape[1])
        return self.act(output)


class DGraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(DGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = tf.cast(placeholders['dropout'], tf.float64)
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_nodes = self.support[0].shape[-1]

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if FLAGS.wts_init == 'random':
                    self.vars['weights_' + str(i)] = glorot_dec([self.num_nodes, input_dim, output_dim],
                                                                name='weights_' + str(i), seed=42)
                elif FLAGS.wts_init == 'zeros':
                    self.vars['weights_' + str(i)] = zeros([self.num_nodes, input_dim, output_dim],
                                                            name='weights_' + str(i))
                else:
                    raise NameError('Unsupported wts_init: {}'.format(FLAGS.wts_init))
            if self.bias:
                self.vars['bias'] = zeros([self.num_nodes, output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # Assuming the shape [num_nodes, feat_in]
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, rate=self.dropout)
        if self.sparse_inputs:
            x = tf.sparse.to_dense(x)
        x = tf.expand_dims(x, 1)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = x @ self.vars['weights_' + str(i)]
            else:
                pre_sup = self.vars['weights_' + str(i)]
            pre_sup = tf.squeeze(pre_sup, 1)
            # support = ops.modal_dot(self.support[i], pre_sup)
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        self.output = output
        # bias
        if self.bias:
            output += self.vars['bias']
        # concated = tf.concat([output, self.act(output)], axis=1)
        # return tf.layers.dense(concated, output.shape[1])
        return self.act(output)
