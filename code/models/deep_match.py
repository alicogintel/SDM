import tensorflow as tf
from models.basic_modules import BasicModules
from models.extra_modules import ExtraModules


class DeepMatch(BasicModules, ExtraModules):

    def __init__(self, parser, hparams):
        BasicModules.__init__(self, parser, hparams)
        ExtraModules.__init__(self, parser, hparams)

    def create_sequence_mask(self, features, seq_input):
        with tf.variable_scope('seq_masks', partitioner=self.partitioner):
            seq_len = features['seq_len']
            max_seq_len = tf.shape(seq_input)[1]
            if "shan" in self.hparams.model:
                seq_len = seq_len + 1
                max_seq_len = max_seq_len + 1
            # [batch_size, time]
            key_masks_1d = tf.sequence_mask(seq_len, max_seq_len)
            # [batch_size, time, time]
            key_masks_2d = tf.tile(tf.expand_dims(key_masks_1d, 1), [1, max_seq_len, 1])
        return key_masks_1d, key_masks_2d

    def create_expand_seq_dim(self, input_1d, seq_input):
        with tf.variable_scope('expand_seq_dim', partitioner=self.partitioner):
            max_seq_len = tf.shape(seq_input)[1]
            input_1d = tf.expand_dims(input_1d, 1)
            input_2d = tf.tile(input_1d, [1, max_seq_len, 1])
        return input_1d, input_2d

    def model_fn_train(self, features):
        nce_weights, nce_biases, seq_input = self.create_item_embeddings(features)
        user_embedding_1d, user_embedding_2d, user_embedding = None, None, None

        if "personal" in self.hparams.model:
            user_embedding = self.create_user_embeddings(features)
            user_embedding_1d, user_embedding_2d = self.create_expand_seq_dim(user_embedding, seq_input)
            if self.hparams.input_user_feature:
                seq_input = self.create_item_user_input(seq_input, user_embedding_2d)

        key_masks_1d, key_masks_2d = self.create_sequence_mask(features, seq_input)

        outputs = seq_input
        if "rnn" in self.hparams.model:
            outputs, last_states = self.create_rnn_encoder(features['seq_len'], seq_input)
        elif "dnn" in self.hparams.model:
            outputs = self.average_item_embedding(seq_input)
        elif "ahead_pos" in self.hparams.model:
            outputs = self.create_position_encoding(seq_input)

        if "self_attn" in self.hparams.model:
            outputs = self.create_self_attn(key_masks_1d, key_masks_2d, outputs)
            if "user_attn" in self.hparams.model:
                outputs = self.create_user_attn(key_masks_2d, outputs, user_embedding_1d, user_embedding_2d)

        if "prefer" in self.hparams.model:
            prefer_outputs = self.create_prefer_embeddings(features, user_embedding_1d)
            if "dnn" in self.hparams.model:
                outputs = self.create_dnn(outputs, prefer_outputs, user_embedding)
            elif "shan" in self.hparams.model:
                outputs = self.create_in_shan(outputs, prefer_outputs)
                outputs = self.create_user_attn(key_masks_2d, outputs, user_embedding_1d, user_embedding_2d)
                outputs = self.create_out_shan(outputs)
            else:
                outputs = self.combine_long_short(outputs, prefer_outputs, user_embedding_2d)

        train_op, my_dict = self.create_split_optimizer(features, outputs, nce_weights, nce_biases)
        return train_op, self.global_step, my_dict
