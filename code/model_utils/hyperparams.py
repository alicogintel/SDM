import collections
from model_utils.task_config import TaskConfig

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'ps_num',
    'mode',
    'model',
    'init_op',
    'seed',
    'init_weight',
    'num_partitions',
    'min_slice_size',
    'batch_size',
    'num_units',
    'vocab_size',
    'unit_type',
    'num_layers',
    'num_residual_layers',
    'forget_bias',
    'dropout',
    'num_samples',
    'optimizer',
    'start_decay_step',
    'learning_rate',
    'decay_steps',
    'decay_factor',
    'colocate_gradients_with_ops',
    'max_gradient_norm',
    'last_step',
    'topK',
    'num_epochs',
    'shuffle',
    'loss_by_example',
    'attention_window_size',
    'num_buckets',
    'num_heads',
    'max_length',
    'input_fn',
    'item_fc_trans',
    'user_fc_trans',
    'nn_init_op',
    "bucket_size",
    "embedding_size",
    "self_attn_ffn",
    "split_size",
    "num_labels",
    "softmax",
    "user_residual",
    "partn_strgy",
    "validation",
    "train_len",
    "test_interval",
    "STAMP",
    "NARM",
    "attn_layer_norm",
    "rnn_layer_norm",
    "user_attn",
    "prefer_avg_pooling",
    "rnn_hidden_units",
    "attn_fc",
    "num_multi_head",
    "wait_time",
    "user_id_only",
    "item_id_only",
    "fusion_op",
    "prefer_fc",
    "g_units_one",
    "input_user_feature",
    "use_user_id",
])


def create_hparams(task_config):

    return TrainingHParams(
        # basic
        ps_num=task_config.get_config_as_int("ps_num", 1),
        mode=task_config.get_config("mode", "train"),
        model=task_config.get_config("model", "rnn"),
        num_buckets=task_config.get_config_as_int("num_buckets", 10),
        max_length=task_config.get_config_as_int("max_length", 50),
        input_fn=task_config.get_config("input_fn", "data_set"),
        topK=task_config.get_config_as_int("topK", 20),
        num_epochs=task_config.get_config_as_int("num_epochs", None),
        shuffle=task_config.get_config_as_bool("shuffle", True),
        validation=task_config.get_config_as_bool("validation", True),
        train_len=task_config.get_config_as_int("train_len", None),
        test_interval=task_config.get_config_as_int("test_interval", 1),
        wait_time=task_config.get_config_as_int("wait_time", 1),

        # initializer
        init_op=task_config.get_config("init_op", "uniform"),
        nn_init_op=task_config.get_config("nn_init_op", "orthogonal"),
        seed=task_config.get_config_as_int("seed", 2018),
        init_weight=task_config.get_config_as_float("init_weight", 0.1),

        # embedding partition
        num_partitions=task_config.get_config_as_int("num_partitions", None),
        min_slice_size=task_config.get_config_as_int("min_slice_size", 32),
        bucket_size={
            "item": task_config.get_config_as_int("item_bucket_size", 10000000),
            "cate": task_config.get_config_as_int("cate_bucket_size", 60000),
            "brand": task_config.get_config_as_int("brand_bucket_size", 10000000),
            "shop": task_config.get_config_as_int("shop_bucket_size", 30000000),
            "user_id": task_config.get_config_as_int("user_id_bucket_size", 1000000),
            "age": task_config.get_config_as_int("age_bucket_size", 100),
            "sex": task_config.get_config_as_int("sex_bucket_size", 10),
            "user_lv_cd": task_config.get_config_as_int("user_lv_cd_bucket_size", 100),
            "city_level": task_config.get_config_as_int("city_level_bucket_size", 100),
            "province": task_config.get_config_as_int("province_bucket_size", 1000),
            "city": task_config.get_config_as_int("city_bucket_size", 1000),
            "country": task_config.get_config_as_int("country_bucket_size", 10000)
        },
        embedding_size={
            "item": task_config.get_config_as_int("item_embedding_size", 64),
            "cate": task_config.get_config_as_int("cate_embedding_size", 16),
            "brand": task_config.get_config_as_int("brand_embedding_size", 16),
            "shop": task_config.get_config_as_int("shop_embedding_size", 32),
            "user_id": task_config.get_config_as_int("user_id_embedding_size", 64),
            "age": task_config.get_config_as_int("age_embedding_size", 4),
            "sex": task_config.get_config_as_int("sex_embedding_size", 4),
            "user_lv_cd": task_config.get_config_as_int("user_lv_cd_embedding_size", 4),
            "city_level": task_config.get_config_as_int("city_level_embedding_size", 4),
            "province": task_config.get_config_as_int("province_embedding_size", 4),
            "city": task_config.get_config_as_int("city_embedding_size", 4),
            "country": task_config.get_config_as_int("country_embedding_size", 4)
        },

        # network
        batch_size=task_config.get_config_as_int("batch_size", 256),
        num_units=task_config.get_config_as_int("num_units", 64),
        vocab_size=task_config.get_config_as_int("vocab_size"),
        unit_type=task_config.get_config("unit_type", "lstm"),
        num_layers=task_config.get_config_as_int("num_layers", 2),
        num_residual_layers=task_config.get_config_as_int("num_residual_layers", 1),
        forget_bias=task_config.get_config_as_float("forget_bias", 1.0),
        dropout=task_config.get_config_as_float("dropout", 0.2),
        num_samples=task_config.get_config_as_int("num_samples", 2000),
        attention_window_size=task_config.get_config_as_int("attention_window_size", None),
        num_heads=task_config.get_config_as_int("num_heads", 8),
        item_fc_trans=task_config.get_config_as_bool("item_fc_trans", False),
        user_fc_trans=task_config.get_config_as_bool("user_fc_trans", False),
        self_attn_ffn=task_config.get_config_as_bool("self_attn_ffn", False),
        user_residual=task_config.get_config_as_bool("user_residual", False),
        STAMP=task_config.get_config_as_bool("STAMP", False),
        NARM=task_config.get_config_as_bool("NARM", False),
        attn_layer_norm=task_config.get_config_as_bool("attn_layer_norm", True),
        rnn_layer_norm=task_config.get_config_as_bool("rnn_layer_norm", False),
        user_attn=task_config.get_config("user_attn", "general"),
        prefer_avg_pooling=task_config.get_config_as_bool("prefer_avg_pooling", False),
        rnn_hidden_units=task_config.get_config_as_int("rnn_hidden_units", 64),
        attn_fc=task_config.get_config_as_bool("attn_fc", False),
        num_multi_head=task_config.get_config_as_int("num_multi_head", 1),
        user_id_only=task_config.get_config_as_bool("user_id_only", False),
        item_id_only=task_config.get_config_as_bool("item_id_only", False),
        fusion_op=task_config.get_config("fusion_op", "gated"),
        prefer_fc=task_config.get_config_as_bool("prefer_fc", True),
        g_units_one=task_config.get_config_as_bool("g_units_one", False),
        input_user_feature=task_config.get_config_as_bool("input_user_feature", False),
        use_user_id=task_config.get_config_as_bool("use_user_id", True),

        # optimizer
        optimizer=task_config.get_config("optimizer", "adam"),
        start_decay_step=task_config.get_config_as_int("start_decay_step", 1600000),
        learning_rate=task_config.get_config_as_float("learning_rate", 1),
        decay_steps=task_config.get_config_as_int("decay_steps", 100000),
        decay_factor=task_config.get_config_as_float("decay_factor", 0.98),
        colocate_gradients_with_ops=task_config.get_config_as_bool("colocate_gradients_with_ops", True),
        max_gradient_norm=task_config.get_config_as_float("max_gradient_norm", 5.0),
        loss_by_example=task_config.get_config_as_bool("loss_by_example", False),
        last_step=task_config.get_config_as_int("last_step", 32000000),
        split_size=task_config.get_config_as_int("split_size", 1),
        num_labels=task_config.get_config_as_int("num_labels", 1),
        softmax=task_config.get_config("softmax", "sampled_softmax"),
        partn_strgy=task_config.get_config("partn_strgy", "mod")
    )


def create_flags(flags):
    flags.DEFINE_string("checkpointDir", "./", "checkpoint_dir")
    flags.DEFINE_string("model", "rnn,self_attn,personal,user_attn,prefer", "model")
    flags.DEFINE_string("mode", "train", "mode")
    flags.DEFINE_string("unit_type", "gru", "unit_type")
    flags.DEFINE_string("num_epochs", 10, "num_epochs")
    flags.DEFINE_string("batch_size", 256, "batch_size")
    flags.DEFINE_string("num_samples", 2000, "num_samples")
    flags.DEFINE_integer("split_size", 1, "split_size, batch split size, splited_samples share neg_samples")
    flags.DEFINE_integer("last_step", 15000000, "last_step")
    flags.DEFINE_string("user_id_embedding_size", 64, "user_id_embedding_size")
    flags.DEFINE_string("num_buckets", 1, "num_buckets")
    flags.DEFINE_string("shuffle", True, "shuffle")
    flags.DEFINE_string("loss_by_example", False, "loss_by_example")
    flags.DEFINE_string("user_residual", True, "user layer residual")
    flags.DEFINE_integer("vocab_size", 157371, "size of item pool")

    flags.DEFINE_string("learning_rate", 0.001, "learning_rate")
    flags.DEFINE_string("start_decay_step", 16000000, "start_decay_step")
    flags.DEFINE_string("decay_steps", 100000, "decay_steps")
    flags.DEFINE_string("decay_factor", 0.95, "decay_factor")
    flags.DEFINE_string("optimizer", "adagrad", "optimizer")
    flags.DEFINE_string("max_gradient_norm", 5.0, "max_gradient_norm")
    flags.DEFINE_integer("num_labels", 5, "multi labels")
    flags.DEFINE_string("softmax", "sampled_softmax", "softmax layer")
    flags.DEFINE_string("partn_strgy", "div", "for inference or not")
    flags.DEFINE_string("validation", True, "validation or not")
    flags.DEFINE_integer("train_len", 1430824, "sample lens")
    flags.DEFINE_string("item_fc_trans", False, "itemid+general repre")
    flags.DEFINE_string("self_attn_ffn", False, "self_attn_ffn")
    flags.DEFINE_integer("test_interval", 1, "test_interval")

    flags.DEFINE_string("STAMP", False, "short term priority")
    flags.DEFINE_string("NARM", False, "neural attentive")
    flags.DEFINE_integer("num_heads", 4, "heads num for attention")
    flags.DEFINE_string("attn_layer_norm", True, "layer_norm attention")
    flags.DEFINE_string("rnn_layer_norm", False, "rnn_layer_norm")
    flags.DEFINE_string("user_attn", "general", "user attention layer choice")
    flags.DEFINE_string("prefer_avg_pooling", False, "prefer features avg pooling, otherwise user attn")
    flags.DEFINE_integer("rnn_hidden_units", 64, "rnn hidden size")
    flags.DEFINE_integer("num_layers", 1, "rnn layer num")
    flags.DEFINE_integer("num_residual_layers", 0, "residual layer num")
    flags.DEFINE_integer("item_embedding_size", 64, "residual layer num")
    flags.DEFINE_integer("num_units", 64, "softmax embedding size")
    flags.DEFINE_string("attn_fc", False, "attention fc")
    flags.DEFINE_integer("num_multi_head", 1, "number of transformers")
    flags.DEFINE_integer("wait_time", 1, "chief worker waiting time")
    flags.DEFINE_string("user_id_only", False, "only user id feature")
    flags.DEFINE_string("item_id_only", False, "only item id feature")
    flags.DEFINE_string("fusion_op", "gated", "fusion operation")
    flags.DEFINE_string("prefer_fc", True, "long rep fc to units")
    flags.DEFINE_string("g_units_one", False, "if scalar gate")
    flags.DEFINE_string("input_user_feature", False, "user feature added to input layer")
    flags.DEFINE_string("use_user_id", True, "user id feature")

    return flags


def create_task_config(FLAGS, conf_file_path):
    FLAGS._parse_flags()
    task_config = TaskConfig(FLAGS.__flags, conf_file_path)
    return task_config
