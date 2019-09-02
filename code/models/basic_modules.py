import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.nn_impl import sampled_softmax_loss
from model_utils import model_helper
from model_utils.model_helper import extract_axis_1, get_optimizer, layer_norm, sequence_feature_mask, \
    self_multi_head_attn, self_attention, pointwise_feedforward, general_attention, concat_attention, \
    soft_max_weighted_sum, learned_positional_encoding
from tensorflow.python.framework import dtypes


class BasicModules:
    def __init__(self, parser, hparams):
        self.hparams = hparams
        self.parser = parser
        self.num_units = self.hparams.num_units
        self.global_step = tf.train.get_or_create_global_step()
        self.initializer = model_helper.get_initializer(self.hparams.init_op,
                                                        self.hparams.seed,
                                                        self.hparams.init_weight)
        self.kernel_initializer = model_helper.get_initializer(self.hparams.nn_init_op,
                                                               seed=self.hparams.seed)
        self.partitioner = model_helper.get_emb_partitioner(self.hparams.num_partitions,
                                                            self.hparams.min_slice_size,
                                                            self.hparams.ps_num)
        self.dropout = tf.placeholder(tf.float32, name="dropout") \
            if self.hparams.validation else self.hparams.dropout
        self.is_training = self.hparams.mode == 'train'
        self.my_dict = {}

    def dataset_batch(self, params, dataset):
        def _parse_function(example_proto):
            features = tf.parse_single_example(example_proto, features=self.parser.feature_map)
            sparse2dense = {k: tf.sparse_tensor_to_dense(f, default_value=0)
                            for k, f in features.iteritems()
                            if isinstance(f, tf.SparseTensor) and f.dtype != tf.string}
            features.update(sparse2dense)
            sparse2dense = {k: tf.sparse_tensor_to_dense(f, default_value="0")
                            for k, f in features.iteritems()
                            if isinstance(f, tf.SparseTensor) and f.dtype == tf.string}
            features.update(sparse2dense)
            # tf.logging.info(features)
            return self.parser.output_one_example(features)

        # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
        def batching_func(x):
            tf_padded = {"fix": [], "int": tf.cast(0, tf.int64), "str": "0",
                         "var": [None], "str_multi": "43,35,12,54,21"}
            padded_shapes = []
            padded_values = []
            for key in self.parser.input_keys:
                key_1 = key[1]
                key_2 = key[2] if key[0] != "multi_labels" else "str_multi"
                padded_shapes.append(tf_padded[key_1])
                padded_values.append(tf_padded[key_2])

            return x.padded_batch(params['batch_size'],
                                  padded_shapes=tuple(padded_shapes),
                                  padding_values=tuple(padded_values))

        def key_func(src_len, *unused_list):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if self.hparams.max_length > 1:
                bucket_width = (self.hparams.max_length + params['num_buckets'] - 1) // params['num_buckets']
            else:
                bucket_width = 5
            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            bucket_id = tf.cast(bucket_id, tf.int32)
            return tf.to_int64(tf.minimum(params['num_buckets'], bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        dataset = dataset.map(_parse_function, num_parallel_calls=32)
        dataset = dataset.repeat(params["epochs"])
        if params['shuffle']:
            dataset = dataset.shuffle(buffer_size=10000, seed=self.hparams.seed)
        if params['num_buckets'] > 1 and params['mode'] == 'train':
            dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func,
                                                                    window_size=params["batch_size"]))
        elif params['mode'] == 'train' or params['mode'] == "test":
            dataset = batching_func(dataset)
        else:
            dataset = dataset.batch(params["batch_size"])
        dataset = dataset.prefetch(buffer_size=1000)
        return dataset

    def input_fn_dataset(self, file_list, data_type="train"):
        if data_type == "test":
            params = {"mode": "test", "epochs": 1, "shuffle": False, "batch_size": 64, "num_buckets": 0}
        else:
            params = {"mode": self.hparams.mode, "epochs": self.hparams.num_epochs, "shuffle": self.hparams.shuffle,
                      "batch_size": self.hparams.batch_size, "num_buckets": self.hparams.num_buckets}

        with tf.name_scope(data_type + '_input_fn') as scope:
            dataset = tf.data.TFRecordDataset(file_list)
            dataset = self.dataset_batch(params, dataset)
            if self.hparams.validation:
                return dataset
            else:
                iterator = dataset.make_one_shot_iterator()
                return self.parser.output_features(iterator)

    def create_item_embeddings(self, features):
        # soft_max
        with variable_scope.variable_scope("soft_max", values=None, partitioner=self.partitioner) as scope:
            nce_biases = tf.zeros([self.hparams.vocab_size], name='bias')
            nce_weights = tf.get_variable(name='weight', shape=[self.hparams.vocab_size, self.num_units],
                                          dtype=tf.float32, initializer=self.initializer)

        # input item embeddings
        with variable_scope.variable_scope("item_embeddings", partitioner=self.partitioner,
                                           initializer=self.initializer, reuse=tf.AUTO_REUSE) as scope:
            embeddings = self.parser.embedding_columns(feature_type="item")
            if self.hparams.item_id_only:
                encoded = layers.sequence_input_from_feature_columns(
                    columns_to_tensors={"item_emb": features["item_ids"],
                                        "shop_emb": features["shop_ids"],
                                        "brand_emb": features["brand_ids"]},
                    feature_columns=embeddings[0:3], scope="reuse_embedding")
            else:
                encoded = layers.sequence_input_from_feature_columns(
                    columns_to_tensors={"item_emb": features["item_ids"],
                                        "shop_emb": features["shop_ids"],
                                        "cate_emb": features["cate_ids"],
                                        "brand_emb": features["brand_ids"]},
                    feature_columns=embeddings, scope="reuse_embedding")
            if self.hparams.item_fc_trans:
                encoded = tf.layers.dense(encoded, self.num_units, tf.nn.tanh,
                                          kernel_initializer=self.kernel_initializer,
                                          name="item_fc")
        return nce_weights, nce_biases, encoded

    def create_user_embeddings(self, features):
        # input user embedding
        with variable_scope.variable_scope("user_embeddings", partitioner=self.partitioner,
                                           initializer=self.initializer) as scope:
            embeddings_fix = self.parser.embedding_columns(feature_type="user_fix", use_hashmap=True)
            if self.hparams.use_user_id and self.hparams.user_id_only:
                encoded = layers.input_from_feature_columns(
                    columns_to_tensors={"user_id_emb": features["user_id"]},
                    feature_columns=[embeddings_fix[0]])
            else:
                personal_encoded = []
                profile_features = {}
                for fs_name in self.parser.embedding_user_features_fix:
                    profile_features.update({fs_name + "_emb": features[fs_name]})

                profile_encoded = layers.input_from_feature_columns(
                    columns_to_tensors=profile_features,
                    feature_columns=embeddings_fix)

                personal_encoded.append(profile_encoded)
                encoded = tf.concat(personal_encoded, -1)

            if self.hparams.user_fc_trans:
                encoded = tf.layers.dense(encoded, self.num_units, tf.nn.tanh,
                                          kernel_initializer=self.kernel_initializer)
            return encoded

    def create_prefer_embeddings(self, features, user_embedding):
        # input prefer item embeddings
        with variable_scope.variable_scope("item_embeddings", partitioner=self.partitioner,
                                           initializer=self.initializer, reuse=tf.AUTO_REUSE) as scope:
            embeddings = self.parser.embedding_columns(feature_type="item")
            feature_names = ["item", "shop", "brand", "cate"]
            if self.hparams.item_id_only:
                feature_names = feature_names[0:3]
            prefer_outputs = []
            for i in range(len(feature_names)):
                key_emb = feature_names[i] + "_emb"
                value_emb = features["prefer_"+feature_names[i]+"s"]
                value_len = features[feature_names[i]+"s"+"_len"]
                prefer_encoded = sequence_feature_mask({key_emb: value_emb},
                                                       [embeddings[i]],
                                                       value_len,
                                                       avg_pooling=self.hparams.prefer_avg_pooling,
                                                       user_embedding=user_embedding,
                                                       drop_out=self.dropout,
                                                       is_training=self.is_training)
                prefer_outputs.append(prefer_encoded)
            prefer_outputs = tf.concat(prefer_outputs, -1)
            if self.hparams.prefer_fc:
                prefer_outputs = tf.layers.dense(prefer_outputs, self.num_units, tf.nn.tanh,
                                                 kernel_initializer=self.kernel_initializer,
                                                 name="prefer_fc")
            return prefer_outputs

    def create_rnn_encoder(self, seq_len, inputs):
        with tf.variable_scope("encoder", values=None,
                               initializer=model_helper.get_initializer(self.hparams.nn_init_op, seed=self.hparams.seed),
                               partitioner=self.partitioner) as scope:
            cell = model_helper.create_rnn_cell(unit_type=self.hparams.unit_type,
                                                num_units=self.hparams.rnn_hidden_units,
                                                num_layers=self.hparams.num_layers,
                                                num_residual_layers=self.hparams.num_residual_layers,
                                                forget_bias=self.hparams.forget_bias,
                                                dropout=self.dropout,
                                                mode=self.hparams.mode,
                                                attention_window_size=self.hparams.attention_window_size)

            rnn_outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,
                                                         sequence_length=seq_len, inputs=inputs)

            if self.hparams.rnn_layer_norm:
                rnn_outputs = layer_norm(rnn_outputs)

        return rnn_outputs, last_states

    def create_position_encoding(self, inputs):
        with tf.variable_scope('add_pos_encoding', initializer=self.initializer, partitioner=self.partitioner):
            pos_input = learned_positional_encoding(inputs, self.hparams.max_length, self.num_units)
            outputs = inputs + pos_input
            outputs = tf.layers.dropout(outputs, self.dropout, training=self.is_training)
            return outputs

    def create_self_attn(self, key_masks_1d, key_masks_2d, inputs):
        attn_outputs = inputs
        for layer in range(self.hparams.num_multi_head):
            with tf.variable_scope('self_attn_'+str(layer), partitioner=self.partitioner):
                if self.hparams.NARM:
                    attn_outputs = self_attention(attn_outputs, num_units=self.num_units,
                                                  key_masks=key_masks_2d, dropout_rate=self.dropout,
                                                  is_training=self.is_training,
                                                  is_layer_norm=self.hparams.attn_layer_norm)
                else:
                    attn_outputs = self_multi_head_attn(attn_outputs, num_units=self.num_units,
                                                        num_heads=self.hparams.num_heads, key_masks=key_masks_1d,
                                                        dropout_rate=self.dropout, is_training=self.is_training,
                                                        is_layer_norm=self.hparams.attn_layer_norm)
            with tf.variable_scope('ffn_'+str(layer), partitioner=self.partitioner):
                if self.hparams.self_attn_ffn:
                        attn_outputs = pointwise_feedforward(attn_outputs, self.dropout, self.is_training,
                                                             num_units=[self.num_units, self.num_units],  # 4 *
                                                             activation=tf.nn.relu)

        with tf.variable_scope('attn_concat', partitioner=self.partitioner):
            if self.hparams.STAMP:
                inputs = tf.layers.dense(inputs, self.num_units, tf.nn.tanh)
                attn_outputs = tf.layers.dense(attn_outputs, self.num_units, tf.nn.tanh)
                attn_outputs = tf.multiply(attn_outputs, inputs)

            if self.hparams.NARM and not self.hparams.STAMP:
                attn_outputs = tf.concat([attn_outputs, inputs], axis=-1)
                if self.hparams.attn_fc:
                    attn_outputs = tf.layers.dense(attn_outputs, self.num_units)

        return attn_outputs

    def create_user_attn(self, key_masks, inputs, user_embedding_1d, user_embedding_2d):
        """
        Args:
            user_embedding : [batch_size, user_embedding_size]
            inputs :         [batch_size, time, num_units]
            key_masks:       sequence mask, 2D tensor
        Returns:
            outputs :        [batch_size, time, num_units]
        """
        with tf.variable_scope('user_attn', partitioner=self.partitioner):
            # [batch_size, 1, num_units]
            # query = tf.expand_dims(user_embedding, 1)
            key = inputs
            align = None
            if self.hparams.user_attn == 'general':
                query = tf.layers.dense(user_embedding_1d, self.num_units, tf.nn.tanh)
                align = general_attention(query, key)
            elif self.hparams.user_attn == 'concat':
                query = user_embedding_2d
                align = concat_attention(query, key)

            # [batch_size, time, time]
            align = tf.tile(align, [1, tf.shape(inputs)[1], 1])
            outputs = soft_max_weighted_sum(align, key, key_masks, self.dropout, self.is_training, future_binding=True)

            if self.hparams.user_residual:
                outputs += inputs
                # outputs = layer_norm(outputs)

        return outputs

    def create_item_user_input(self, seq_input, user_embedding):
        with tf.variable_scope('item_user_feature', partitioner=self.partitioner):
            # user_embedding = tf.tile(tf.expand_dims(user_embedding, 1), [1, tf.shape(seq_input)[1], 1])
            seq_input = tf.concat([seq_input, user_embedding], axis=-1)
        return seq_input

    def combine_long_short(self, short_rep, long_rep, user_embedding):
        """
        short_rep: [batch_size, time, units]
        long_rep: [batch_size, units]
        user_embedding: [batch_size, units]
        """
        with variable_scope.variable_scope("fusion", partitioner=self.partitioner) as scope:
            long_rep = tf.tile(tf.expand_dims(long_rep, 1), [1, tf.shape(short_rep)[1], 1])
            if self.hparams.fusion_op == "add":
                outputs = long_rep + short_rep
            elif self.hparams.fusion_op == "multiply":
                outputs = tf.multiply(long_rep, short_rep)
            elif self.hparams.fusion_op == "concat":
                outputs = tf.concat([short_rep, long_rep], axis=-1)
                outputs = tf.layers.dense(outputs, self.num_units)
            elif self.hparams.fusion_op == "feature_gated":
                f_input = tf.concat([short_rep, long_rep], -1)
                f = tf.layers.dense(f_input, self.num_units, activation=tf.nn.tanh)
                g_input = tf.concat([short_rep, long_rep], -1)
                g = tf.layers.dense(g_input, self.num_units, activation=tf.sigmoid)
                outputs = tf.multiply(g, short_rep) + tf.multiply(1 - g, f)
                tf.summary.scalar("gate", tf.reduce_mean(g))
            else:
                g_units = self.num_units
                if self.hparams.g_units_one:
                    g_units = 1
                # user_embedding = tf.tile(tf.expand_dims(user_embedding, 1), [1, tf.shape(short_rep)[1], 1])
                g_input = tf.concat([short_rep, long_rep, user_embedding], -1)
                g = tf.layers.dense(g_input, g_units, activation=tf.sigmoid)
                outputs = tf.multiply(g, short_rep) + tf.multiply(1 - g, long_rep)
                tf.summary.scalar("gate", tf.reduce_mean(g))

            return outputs

    def calculate_loss(self, nce_weights, nce_biases, label_split, rnn_outputs_split, target_splits, batch_size,
                       sampled_values=None):

        sampled_loss = sampled_softmax_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=label_split,
            inputs=rnn_outputs_split,
            num_sampled=self.hparams.num_samples,
            num_classes=self.hparams.vocab_size,
            num_true=self.hparams.num_labels,
            sampled_values=sampled_values,
            partition_strategy=self.hparams.partn_strgy
        )

        sampled_loss = tf.reshape(sampled_loss, [batch_size, -1])
        sampled_loss = tf.reduce_sum(sampled_loss * target_splits)

        return sampled_loss

    def create_split_optimizer(self, features, outputs, nce_weights, nce_biases):
        seq_len = tf.cast(features["seq_len"], dtypes.int32)
        batch_size = tf.shape(outputs)[0]

        with tf.variable_scope("loss") as scope:
            rnn_outputs_flat = tf.reshape(outputs, [-1, self.num_units])
            num_labels = self.hparams.num_labels
            if num_labels > 1:
                multi_labels = tf.reshape(features["multi_labels"], [-1])
                multi_labels = tf.string_split(multi_labels, delimiter=",").values
                multi_labels = tf.reshape(multi_labels, [-1, num_labels])
                label_flat = tf.string_to_number(multi_labels, out_type=tf.int64)
            else:
                label_flat = tf.reshape(features["labels"], [-1, 1])
            istarget = tf.sequence_mask(seq_len, tf.shape(outputs)[1], dtype=outputs.dtype)

            rnn_outputs_splits = tf.split(rnn_outputs_flat, num_or_size_splits=self.hparams.split_size,
                                          name="rnn_output_split", axis=0)
            label_splits = tf.split(label_flat, num_or_size_splits=self.hparams.split_size,
                                    name="label_split", axis=0)
            istarget_splits = tf.split(istarget, num_or_size_splits=self.hparams.split_size,
                                       name="istarget_split", axis=0)

            losses = []

            i = 0
            for (rnn_outputs_split, label_split, target_split) in zip(rnn_outputs_splits,
                                                                      label_splits,
                                                                      istarget_splits):
                with tf.variable_scope("loss_" + str(i)) as scope:
                    sampled_loss = self.calculate_loss(nce_weights, nce_biases,
                                                       label_split, rnn_outputs_split,
                                                       target_split, batch_size / self.hparams.split_size)
                    losses.append(sampled_loss)
                i += 1

            all_loss = sum(losses)

            _mean_loss_by_example = all_loss / (tf.to_float(batch_size))
            _mean_loss_by_pos = all_loss / (tf.reduce_sum(istarget))
            if self.hparams.loss_by_example:
                _mean_loss = _mean_loss_by_example
            else:
                _mean_loss = _mean_loss_by_pos
            _mean_loss = tf.check_numerics(_mean_loss, "loss is nan of inf")

        with tf.variable_scope("metrics"):
            tf.summary.scalar("mean_loss_by_example", _mean_loss_by_example)
            tf.summary.scalar("mean_loss_by_pos", _mean_loss_by_pos)
            tf.summary.scalar("train_loss", _mean_loss)
            for i in range(self.hparams.split_size):
                tf.summary.scalar("sample_loss_" + str(i), losses[i])

        with tf.variable_scope("optimizer") as scope:
            params = tf.trainable_variables()
            gradients = tf.gradients(_mean_loss, params,
                                     colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops)
            clipped_gradients = model_helper.gradient_clip(gradients=gradients,
                                                           max_gradient_norm=self.hparams.max_gradient_norm)
            opt, _learning_rate = get_optimizer(self.hparams, self.global_step)
            train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.my_dict.update({
            'learning_rate': _learning_rate,
            'loss': _mean_loss,
            'drop_out': self.dropout
        })

        if self.hparams.validation:
            with tf.variable_scope("validation"):
                last_output = extract_axis_1(outputs, seq_len - 1)
                logits = tf.matmul(last_output, tf.transpose(nce_weights)) + nce_biases
                top_item_ids = tf.nn.top_k(logits, k=self.hparams.topK).indices
                top_item_ids = tf.reshape(top_item_ids, [batch_size, self.hparams.topK])

            self.my_dict.update({
                'top_items': top_item_ids,
                'user_id': features['user_id'],
                'ds': features['ds'],
                "weight": nce_weights.as_tensor(),
                'user_embedding_output': last_output
            })

        return train_op, self.my_dict
