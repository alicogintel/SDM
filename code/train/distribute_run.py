import os
import random
import sys
import tensorflow as tf
import time
import traceback
import numpy as np

currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + os.sep + '../')
sys.path.append(currentPath + os.sep + '../..')

from parsers.model_feature_parser import ModelFeatureParser
from model_utils.hyperparams import create_hparams, create_flags, create_task_config
from tensorflow.python.platform import tf_logging as logging
from train.utils import parent_directory, get_current_volume_files
from model_utils import model_helper
from models.deep_match import DeepMatch

flags = tf.app.flags
FLAGS = create_flags(flags).FLAGS


def main(unused_argv):
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    tf.logging.set_verbosity(tf.logging.INFO)

    # construct the servers
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count = len(worker_spec)
    ps_count = len(ps_spec)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

    # join the ps server
    if FLAGS.job_name == "ps":
        server.join()

    # Parse config parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conf_file_path = os.path.join(os.path.join(parent_directory(current_dir), 'config/task_config.json'))
    logging.info("will use task conf file %s" % conf_file_path)
    task_config = create_task_config(FLAGS, conf_file_path)
    task_config.add_if_not_contain("ps_num", ps_count)
    hparams = create_hparams(task_config=task_config)
    print hparams

    parser = ModelFeatureParser(hparams)
    model = DeepMatch(parser, hparams)

    # start the training
    try:
        run_validating(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster,
                       is_chief=is_chief, target=server.target, hparams=hparams, model=model)
    except Exception, e:
        logging.error("catch a exception: %s" % e.message)
        logging.error("exception is: %s" % traceback.format_exc())
        raise Exception("terminate process!")


def run_validating(worker_count, task_index, cluster, is_chief, target, hparams, model):
    # acc_keys = ["user_id", "top_items"]
    acc_keys = ["user_id", "ds", "user_embedding_output"]
    volume = FLAGS.volumes
    volume = volume.split(",")
    train_file_list = tf.gfile.ListDirectory(volume[0])
    test_file_list = tf.gfile.ListDirectory(volume[1])
    if not train_file_list or len(train_file_list) == 0 or not test_file_list or len(test_file_list) == 0:
        logging.error("End training directly since no train files or test files!")
        return

    train_file = get_current_volume_files(volume[0], train_file_list, task_index, worker_count)
    test_file = get_current_volume_files(volume[1], test_file_list, task_index, worker_count)
    logging.info("current_train_file: {}".format(train_file))
    logging.info("current_test_file: {}".format(test_file))

    checkpointDir = FLAGS.checkpointDir
    if is_chief:
        if not tf.gfile.Exists(checkpointDir):
            tf.gfile.MakeDirs(checkpointDir)
            with tf.gfile.FastGFile(os.path.join(checkpointDir, "hyperparams"), 'w') as f:
                f.write(str(hparams))
                f.flush()
                f.close()

    with tf.device(model_helper.get_device_str(task_index)):
        train_data = model.input_fn_dataset(train_file, data_type="train")
        test_data = model.input_fn_dataset(test_file, data_type="test")

        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        train_init_op = iterator.make_initializer(train_data)
        test_init_op = iterator.make_initializer(test_data)

        features = model.parser.output_features(iterator)

    with tf.device(tf.train.replica_device_setter(
            worker_device=model_helper.get_device_str(task_index), cluster=cluster)):
        train_op, global_step, my_dict = model.model_fn_train(features)

    steps_per_epoch = hparams.train_len // hparams.batch_size
    test_interval = hparams.test_interval
    epochs = test_interval

    writer = tf.python_io.TableWriter(FLAGS.output_test_result, slice_id=task_index)
    config = tf.ConfigProto()

    on_hold = tf.Variable(tf.constant(worker_count), dtype=tf.int32)
    subtract_op = tf.assign_add(on_hold, -1)
    add_op = tf.assign_add(on_hold, 1)

    chief_only_hooks = [tf.train.StepCounterHook()]
    drop_zero_dict = {
        my_dict['drop_out']: 0.0
    }

    drop_dict = {
        my_dict['drop_out']: hparams.dropout
    }
    summary_dir = os.path.join(FLAGS.checkpointDir, 'train')

    with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief,
                                           chief_only_hooks=chief_only_hooks, config=config) as sess:
        if is_chief:
            train_writer = tf.summary.MetricsWriter(summary_dir, sess.graph)
        step_ = 0
        sess.run(train_init_op, feed_dict=drop_zero_dict)
        while step_ < steps_per_epoch * hparams.num_epochs + 5:
            step_, on_hold_ = sess.run([global_step, on_hold], feed_dict=drop_zero_dict)
            if on_hold_ == worker_count:
                _, loss_, step_, lr_ = sess.run([train_op, my_dict["loss"], global_step, my_dict['learning_rate']],
                                                feed_dict=drop_dict)
                if is_chief:
                    train_writer.add_scalar("loss", loss_, step_)
                    train_writer.add_scalar("learning_rate", lr_, step_)
                if random.randint(1, 200) == 1:
                    logging.info("[Epoch {}] {}_sampled_mean_loss: {}".format(epochs, step_, loss_))
            else:
                logging.info("[Epoch {}] Waiting for other workers...".format(epochs))
                time.sleep(5)
            if step_ >= steps_per_epoch * epochs:
                logging.info("[Epoch {}] Testing...".format(epochs))
                sess.run(subtract_op, feed_dict=drop_zero_dict)
                sess.run(test_init_op, feed_dict=drop_zero_dict)
                weight = sess.run(my_dict['weight'], feed_dict=drop_zero_dict)
                logging.info(weight.shape)
                test_batch_counter = 0
                try:
                    while True:
                        test_batch_counter += 1
                        user_id, ds, user_vector = sess.run([my_dict[j] for j in acc_keys],
                                                            feed_dict=drop_zero_dict)
                        user_id = user_id.tolist()
                        ds = ds.tolist()
                        arr = np.matmul(user_vector, np.transpose(weight))
                        indices = np.argpartition(arr, -hparams.topK, axis=1)[:, -hparams.topK:]
                        for num, p in enumerate(zip(user_id, indices, ds)):
                            writer.write([p[0], ','.join(map(str, p[1])), epochs, p[2]], [0, 1, 2, 3])
                except tf.errors.OutOfRangeError:
                    # logging.info(e.message)
                    # logging.info("[Epoch {}] Testing Finished...".format(epochs))
                    logging.info("[Epoch {}] test batch counter {}..".format(epochs, test_batch_counter))
                    pass
                sess.run(train_init_op, feed_dict=drop_zero_dict)
                sess.run(add_op, feed_dict=drop_zero_dict)
                logging.info("[Epoch {}] Back to train...".format(epochs))
                epochs += test_interval
        sess.run(subtract_op, feed_dict=drop_zero_dict)
        if is_chief:
            count = 100
            on_hold_ = sess.run(on_hold, feed_dict=drop_zero_dict)
            while on_hold_ > 0 or count > 0:
                logging.info("[Epoch {} Chief] On hold value {}".format(epochs, on_hold_))
                time.sleep(hparams.wait_time)
                on_hold_ = sess.run(on_hold, feed_dict=drop_zero_dict)
                count -= 1
    writer.close()
    logging.info("*" * 20 + "End training.")


if __name__ == '__main__':
    tf.app.run()
