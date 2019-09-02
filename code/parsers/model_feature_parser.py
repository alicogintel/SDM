import tensorflow as tf
from model_utils import model_helper


class ModelFeatureParser(object):
    def __init__(self, hparams):
        self.hparams = hparams
        # tf record input data schema
        self.len_fix_int_keys = [("seq_len", "fix", "int"), ("items_len", "fix", "int"),
                                 ("shops_len", "fix", "int"), ("cates_len", "fix", "int"), ("brands_len", "fix", "int")]

        self.label_var_int_keys = [("labels", "var", "int")]

        self.label_var_str_keys = [("multi_labels", "var", "str")]

        self.item_feature_var_str_keys = [("item_ids", "var", "str"), ("shop_ids", "var", "str"),
                                          ("cate_ids", "var", "str"), ("brand_ids", "var", "str")]

        self.user_feature_var_str_keys = [("prefer_items", "var", "str"), ("prefer_shops", "var", "str"),
                                          ("prefer_cates", "var", "str"), ("prefer_brands", "var", "str")]

        self.user_feature_fix_str_keys = [("user_id", "fix", "str"), ("age", "fix", "str"), ("sex", "fix", "str"),
                                          ("user_lv_cd", "fix", "str"), ("city_level", "fix", "str"),
                                          ("province", "fix", "str"), ("city", "fix", "str"), ("country", "fix", "str")]

        # distinct sparse id feature
        self.embedding_item_features = ["item", "shop", "brand", "cate"]
        self.embedding_user_features_fix = ["user_id", "age", "sex", "user_lv_cd", "city_level", "province", "city", "country"]

        self.input_keys = self.len_fix_int_keys + self.label_var_int_keys + \
                          self.label_var_str_keys + self.item_feature_var_str_keys + \
                          self.user_feature_var_str_keys + self.user_feature_fix_str_keys + [("ds", "fix", "str")]

        tf_feature = {"fix_int": tf.FixedLenFeature([], dtype=tf.int64),
                      "var_int": tf.VarLenFeature(dtype=tf.int64),
                      "var_str": tf.VarLenFeature(dtype=tf.string),
                      "fix_str": tf.FixedLenFeature([], dtype=tf.string)}
        self.feature_map = {}
        if self.hparams.mode == "train":
            for key in self.input_keys:
                self.feature_map.update({key[0]: tf_feature[key[1] + '_' + key[2]]})

    def embedding_columns(self, feature_type, use_hashmap=False):
        sparse_features_emb = []
        embedding_features = {"item": self.embedding_item_features, "user_fix": self.embedding_user_features_fix}
        # item or feature
        for fs_name in embedding_features[feature_type]:
            new_emb = model_helper.hash_bucket_embedding(fs_name+'_emb', self.hparams.bucket_size[fs_name],
                                                         self.hparams.embedding_size[fs_name],
                                                         use_hashmap=use_hashmap)
            sparse_features_emb.append(new_emb)
        return sparse_features_emb

    def output_one_example(self, features):
        if self.hparams.mode == "train":
            example = []
            for key in self.input_keys:
                example.append(features[key[0]])
            return example

    def output_features(self, iterator):
        if self.hparams.mode == "train":
            features = iterator.get_next()
            return {self.input_keys[i][0]: features[i] for i in range(len(self.input_keys))}
