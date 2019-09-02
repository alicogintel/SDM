import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op is None:
        return None
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif init_op == "normal":
        return tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=seed)
    elif init_op == "glorot_normal":
        return tf.contrib.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
    elif init_op == "xavier":
        return tf.contrib.layers.xavier_initializer(seed=seed)
    elif init_op == "orthogonal":
        return tf.orthogonal_initializer()
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def get_emb_partitioner(num_partitions=None, min_slice_size=None, max_partitions=None):
    partitioner = None
    if num_partitions > 1:
        partitioner = tf.fixed_size_partitioner(num_partitions)
    elif min_slice_size is not None and max_partitions is not None:
        partitioner = partitioned_variables.min_max_variable_partitioner(
            max_partitions=max_partitions,
            min_slice_size=min_slice_size << 20)
    return partitioner


def _single_cell(unit_type, num_units, forget_bias, dropout,
                 mode, residual_connection=False):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    logger_list = []
    # Cell Type
    if unit_type == "lstm":
        logger_list.append("  LSTM, forget_bias=%g" % forget_bias)
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == "lstmblock":
        logger_list.append("  LSTM Block, forget_bias=%g" % forget_bias)
        single_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=forget_bias)
    elif unit_type == "lstmfused":
        logger_list.append("  LSTM Block Fused, forget_bias=%g" % forget_bias)
        single_cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        logger_list.append("  GRU")
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        logger_list.append("  Layer Normalized LSTM, forget_bias=%g" % forget_bias)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    dropout = dropout if mode == "train" else 0
    single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
    logger_list.append("  %s " % type(single_cell).__name__)

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
        logger_list.append("  %s" % type(single_cell).__name__)
    logging.info("".join(logger_list))

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, single_cell_fn=None):
    """Create a list of RNN cells."""
    if not single_cell_fn:
        single_cell_fn = _single_cell

    cell_list = []
    for i in range(num_layers):
        logging.info("  cell %d" % i)
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers)
        )
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, attention_window_size, single_cell_fn=None):
    """Create multi-layer RNN cell.

      Args:
        unit_type: string representing the unit type, i.e. "lstm".
        num_units: the depth of each unit.
        num_layers: number of cells.
        num_residual_layers: Number of residual layers from top to bottom. For
          example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
          cells in the returned list will be wrapped with `ResidualWrapper`.
        forget_bias: the initial forget bias of the RNNCell(s).
        dropout: floating point value between 0.0 and 1.0:
          the probability of dropout.  this is ignored if `mode != train`.
        mode: either train/predict
        single_cell_fn: single_cell_fn: allow for adding customized cell.
          When not specified, we default to model_helper._single_cell
      Returns:
        An `RNNCell` instance.
    """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:  # Single layer.
        final_cell = cell_list[0]
    else:                    # Multi layers
        final_cell = tf.contrib.rnn.MultiRNNCell(cell_list)

    #  Attention Wrapper Cell
    if attention_window_size is not None:
        final_cell = tf.contrib.rnn.AttentionCellWrapper(final_cell, attention_window_size)
    return final_cell


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    tf.summary.scalar("grad_norm", gradient_norm)
    tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))

    return clipped_gradients


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """
    batch_range = tf.range(tf.shape(data)[0], dtype=tf.int32)
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def get_optimizer(hparams, _global_step):
    _learning_rate = tf.constant(hparams.learning_rate)
    opt = tf.train.GradientDescentOptimizer(hparams.learning_rate)
    if hparams.optimizer == "sgd":
        _learning_rate = tf.cond(
            _global_step < hparams.start_decay_step,
            lambda: tf.constant(hparams.learning_rate),
            lambda: tf.train.exponential_decay(
                hparams.learning_rate,
                (_global_step - hparams.start_decay_step),
                hparams.decay_steps,
                hparams.decay_factor,
                staircase=True),
            name="learning_rate")
        opt = tf.train.GradientDescentOptimizer(_learning_rate)
    elif hparams.optimizer == "adam":
        assert float(hparams.learning_rate) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
        opt = tf.train.AdamOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'RMSprop':
        opt = tf.train.RMSPropOptimizer(hparams.learning_rate)
    tf.summary.scalar("lr", _learning_rate)
    return opt, _learning_rate


def hash_bucket_embedding(name, bucket_size, dim, use_hashmap=False):
    if use_hashmap:
        id_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name=name, hash_bucket_size=bucket_size, use_hashmap=True)
    else:
        id_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name=name, hash_bucket_size=bucket_size)
    return tf.contrib.layers.embedding_column(sparse_id_column=id_feature, dimension=dim)


def learned_positional_encoding(inputs, max_length, num_units):
    outputs = tf.range(tf.shape(inputs)[1])                # (T_q)
    outputs = tf.where(tf.greater_equal(outputs, max_length), tf.fill(tf.shape(outputs), max_length - 1), outputs)
    outputs = tf.expand_dims(outputs, 0)                   # (1, T_q)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])   # (N, T_q)
    with variable_scope.variable_scope("embeddings") as scope:
        pos_embedding = tf.get_variable(name="pos_embedding", shape=[max_length, num_units],
                                        dtype=tf.float32)
        encoded = tf.nn.embedding_lookup(pos_embedding, outputs)
    return encoded


def pointwise_feedforward(inputs, drop_out, is_training, num_units=None, activation=None):
    # Inner layer
    # outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=activation)
    outputs = tf.layers.dense(inputs, num_units[0], activation=activation)
    outputs = tf.layers.dropout(outputs, drop_out, training=is_training)
    # Readout layer
    # outputs = tf.layers.conv1d(outputs, num_units[1], kernel_size=1, activation=None)
    outputs = tf.layers.dense(outputs, num_units[1], activation=None)

    # drop_out before add&norm
    outputs = tf.layers.dropout(outputs, drop_out, training=is_training)
    # Residual connection
    outputs += inputs
    # Normalize
    outputs = layer_norm(outputs)
    return outputs


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def self_multi_head_attn(inputs, num_units, num_heads, key_masks, dropout_rate, is_training, is_layer_norm=True):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    if num_units is None:
        num_units = inputs.get_shape().as_list[-1]

    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    # (h*N, T_q, T_k)
    align = general_attention(Q_, K_)

    # (h*N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])
    # (h*N, T_q, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(inputs)[1], 1])
    # (h*N, T_q, C/h)
    outputs = soft_max_weighted_sum(align, V_, key_masks, dropout_rate, is_training, future_binding=True)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    # output linear
    outputs = tf.layers.dense(outputs, num_units)

    # drop_out before residual and layernorm
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs  # (N, T_q, C)
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs


def concat_attention(query, key):
    """
    :param query: [batch_size, 1, query_size] -> [batch_size, time, query_size]
    :param key:   [batch_size, time, key_size]
    :return:      [batch_size, 1, time]
        query_size should keep the same dim with key_size
    """
    # TODO: only support 1D attention at present
    # query = tf.tile(query, [1, tf.shape(key)[1], 1])
    # [batch_size, time, q_size+k_size]
    q_k = tf.concat([query, key], axis=-1)
    # [batch_size, time, 1]
    align = tf.layers.dense(q_k, 1, tf.nn.tanh)  # tf.nn.relu old
    # scale (optional)
    align = align / (key.get_shape().as_list()[-1] ** 0.5)
    align = tf.transpose(align, [0, 2, 1])
    return align


def general_attention(query, key):
    """
    :param query: [batch_size, None, query_size]
    :param key:   [batch_size, time, key_size]
    :return:      [batch_size, None, time]
        query_size should keep the same dim with key_size
    """
    # [batch_size, None, time]
    align = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
    # scale (optional)
    align = align / (key.get_shape().as_list()[-1] ** 0.5)
    return align


def self_attention(inputs, num_units, key_masks, dropout_rate, is_training, is_layer_norm=True):
    """
    Args:
      inputs(queries): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    # if num_units is None:
    #     num_units = inputs.get_shape().as_list[-1]

    # (N, T_q, C)
    # Q = tf.layers.dense(inputs, num_units, tf.nn.relu, name='unlinear_trans', reuse=tf.AUTO_REUSE)
    # (N, T_k, C)
    # K = tf.layers.dense(inputs, num_units, tf.nn.relu, name="unlinear_trans", reuse=tf.AUTO_REUSE)

    Q = inputs
    K = inputs
    V = inputs

    align = general_attention(Q, K)
    outputs = soft_max_weighted_sum(align, V, key_masks, dropout_rate, is_training, future_binding=True)

    # Residual connection
    # outputs += inputs  # (N, T_q, C)
    if is_layer_norm:
        # Normalize
        outputs = layer_norm(outputs)  # (N, T_q, C)
    return outputs


def soft_max_weighted_sum(align, value, key_masks, drop_out, is_training, future_binding=False):
    """
    :param align:           [batch_size, None, time]
    :param value:           [batch_size, time, units]
    :param key_masks:       [batch_size, None, time]
                            2nd dim size with align
    :param drop_out:
    :param is_training:
    :param future_binding:  TODO: only support 2D situation at present
    :return:                weighted sum vector
                            [batch_size, None, units]
    """
    # exp(-large) -> 0
    paddings = tf.fill(tf.shape(align), float('-inf'))
    # [batch_size, None, time]
    align = tf.where(key_masks, align, paddings)

    if future_binding:
        length = tf.reshape(tf.shape(value)[1], [-1])
        # [time, time]
        lower_tri = tf.ones(tf.concat([length, length], axis=0))
        # [time, time]
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
        # [batch_size, time, time]
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        # [batch_size, time, time]
        align = tf.where(tf.equal(masks, 0), paddings, align)

    # soft_max and dropout
    # [batch_size, None, time]
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, drop_out, training=is_training)
    # weighted sum
    # [batch_size, None, units]
    return tf.matmul(align, value)


def sequence_feature_mask(columns_to_tensors, feature_columns, seq_len, avg_pooling=False,
                          user_embedding=None, drop_out=0, is_training=True):
    # [batch_size, time, units]
    encoded = layers.sequence_input_from_feature_columns(
        columns_to_tensors=columns_to_tensors,
        feature_columns=feature_columns,
        scope="reuse_embedding"
    )

    # [batch_size, time]
    key_masks = tf.sequence_mask(seq_len, tf.shape(encoded)[1], dtypes.float32)

    if avg_pooling:
        # [batch_size, time, 1]
        key_masks = tf.reshape(key_masks, [-1, tf.shape(encoded)[1], 1])
        encoded = tf.multiply(encoded, key_masks)
        encoded = tf.reduce_sum(encoded, 1) / tf.reshape(tf.cast(seq_len, dtypes.float32), [-1, 1])
    else:
        # [batch_size, 1, time]
        query = tf.tile(user_embedding, [1, tf.shape(encoded)[1], 1])
        align = concat_attention(query, encoded)
        key_masks = tf.cast(key_masks, dtypes.bool)
        # [batch_size, 1, time]
        key_masks = tf.expand_dims(key_masks, 1)
        encoded = soft_max_weighted_sum(align, encoded, key_masks, drop_out, is_training)
        encoded = tf.squeeze(encoded, 1)
    # [batch_size, units]
    return encoded
