import os
import random
import sys
import tensorflow as tf
import traceback
import numpy as np

currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + os.sep + '../')
sys.path.append(currentPath + os.sep + '../..')

from parsers.model_feature_parser import ModelFeatureParser
from model_utils.hyperparams import create_hparams, create_flags, create_task_config
from tensorflow.python.platform import tf_logging as logging
from train.utils import parent_directory
from models.deep_match import DeepMatch

flags = tf.app.flags
FLAGS = create_flags(flags).FLAGS


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Parse config parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conf_file_path = os.path.join(os.path.join(parent_directory(current_dir), 'config/task_config.json'))
    logging.info("will use task conf file %s" % conf_file_path)
    task_config = create_task_config(FLAGS, conf_file_path)
    hparams = create_hparams(task_config=task_config)
    print hparams

    parser = ModelFeatureParser(hparams)
    model = DeepMatch(parser, hparams)

    # start the training
    try:
        run_validating(hparams=hparams, model=model)
    except Exception, e:
        logging.error("catch a exception: %s" % e.message)
        logging.error("exception is: %s" % traceback.format_exc())
        raise Exception("terminate process!")


def run_validating(hparams, model):
    acc_keys = ["user_id", "ds", "user_embedding_output"]

    #  user defined function
    #  you should write your own code here for reading and writing data
    train_file = get_your_train_files()
    test_file = get_your_test_files()
    writer = open_your_test_result_file()

    if not train_file or len(train_file) == 0 or not test_file or len(test_file) == 0:
        logging.error("End training directly since no train files or test files!")
        return

    logging.info("current_train_file: {}".format(train_file))
    logging.info("current_test_file: {}".format(test_file))

    checkpointDir = FLAGS.checkpointDir
    if not tf.gfile.Exists(checkpointDir):
        tf.gfile.MakeDirs(checkpointDir)
        with tf.gfile.FastGFile(os.path.join(checkpointDir, "hyperparams"), 'w') as f:
            f.write(str(hparams))
            f.flush()
            f.close()

    train_data = model.input_fn_dataset(train_file, data_type="train")
    test_data = model.input_fn_dataset(test_file, data_type="test")

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    train_init_op = iterator.make_initializer(train_data)
    test_init_op = iterator.make_initializer(test_data)

    features = model.parser.output_features(iterator)

    train_op, global_step, my_dict = model.model_fn_train(features)

    steps_per_epoch = hparams.train_len // hparams.batch_size
    test_interval = hparams.test_interval
    epochs = test_interval

    config = tf.ConfigProto()

    chief_only_hooks = [tf.train.StepCounterHook()]
    drop_zero_dict = {
        my_dict['drop_out']: 0.0
    }

    drop_dict = {
        my_dict['drop_out']: hparams.dropout
    }
    summary_dir = os.path.join(FLAGS.checkpointDir, 'train')

    with tf.train.MonitoredTrainingSession(chief_only_hooks=chief_only_hooks, config=config) as sess:
        train_writer = tf.summary.MetricsWriter(summary_dir, sess.graph)
        step_ = 0
        sess.run(train_init_op, feed_dict=drop_zero_dict)
        while step_ < steps_per_epoch * hparams.num_epochs + 5:
            _, loss_, step_, lr_ = sess.run([train_op, my_dict["loss"], global_step, my_dict['learning_rate']],
                                            feed_dict=drop_dict)
            train_writer.add_scalar("loss", loss_, step_)
            train_writer.add_scalar("learning_rate", lr_, step_)
            if random.randint(1, 200) == 1:
                logging.info("[Epoch {}] {}_sampled_mean_loss: {}".format(epochs, step_, loss_))
            if step_ >= steps_per_epoch * epochs:
                logging.info("[Epoch {}] Testing...".format(epochs))
                sess.run(test_init_op, feed_dict=drop_zero_dict)
                weight = sess.run(my_dict['weight'], feed_dict=drop_zero_dict)
                logging.info(weight.shape)
                test_batch_counter = 0
                try:
                    while True:
                        test_batch_counter += 1
                        user_id, ds, user_vector = sess.run([my_dict[j] for j in acc_keys], feed_dict=drop_zero_dict)
                        user_id = user_id.tolist()
                        ds = ds.tolist()
                        arr = np.matmul(user_vector, np.transpose(weight))
                        indices = np.argpartition(arr, -hparams.topK, axis=1)[:, -hparams.topK:]
                        for num, p in enumerate(zip(user_id, indices, ds)):
                            writer.write([p[0], ','.join(map(str, p[1])), epochs, p[2]])
                except tf.errors.OutOfRangeError:
                    logging.info("[Epoch {}] test batch counter {}..".format(epochs, test_batch_counter))
                    pass
                sess.run(train_init_op, feed_dict=drop_zero_dict)
                logging.info("[Epoch {}] Back to train...".format(epochs))
                epochs += test_interval

    logging.info("*" * 20 + "End training.")


if __name__ == '__main__':
    tf.app.run()
