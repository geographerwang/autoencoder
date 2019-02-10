import tensorflow as tf
from data.dataset import _get_training_data, _get_test_data
from DAE import DAE
import numpy as np
import time
import os


tf.app.flags.DEFINE_string('tf_records_train_path', 
                           'data/ml-1m/train/',
                           'Path of the training data.')

tf.app.flags.DEFINE_string('tf_records_test_path', 
                           'data/ml-1m/test/',
                           'Path of the test data.')

tf.app.flags.DEFINE_string('output_dir',
                           '.',
                           'Path of the model.')

tf.app.flags.DEFINE_integer('num_epoch', 500,
                            'Number of training epochs.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'Size of the training batch.')

tf.app.flags.DEFINE_float('learning_rate', 0.0005,
                          'Learning_Rate')

tf.app.flags.DEFINE_boolean('l2_reg', False,
                            'L2 regularization.')

tf.app.flags.DEFINE_float('lambda_', 0.01,
                          'Wight decay factor.')

tf.app.flags.DEFINE_integer('num_v', 3952,
                            'Number of visible neurons(Number of movies the users rated.)')

tf.app.flags.DEFINE_integer('num_h', 128,
                            'Number of hidden neurons.)')

tf.app.flags.DEFINE_integer('num_samples', 5953,
                            'Number of training samples (Number of users, who gave a rating).')

FLAGS = tf.app.flags.FLAGS


def main(_):
    '''Building the graph, opening of a session and starting the training od the neural network.'''

    num_batches = int(FLAGS.num_samples/FLAGS.batch_size)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.output_dir, "runs", timestamp))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with tf.Graph().as_default():

        train_data, train_data_infer = _get_training_data(FLAGS)
        test_data = _get_test_data(FLAGS)

        iter_train = train_data.make_initializable_iterator()
        iter_train_infer = train_data_infer.make_initializable_iterator()
        iter_test = test_data.make_initializable_iterator()

        x_train = iter_train.get_next()
        x_train_infer = iter_train_infer.get_next()
        x_test = iter_test.get_next()

        model = DAE(FLAGS)

        train_op, train_loss_op = model._optimizer(x_train)
        pred_op, test_loss_op = model._validation_loss(x_train_infer, x_test)

        with tf.Session() as sess:
            loss_var = tf.Variable(0.0)
            tf.summary.scalar("loss", loss_var)
            write_op = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            # Summaries for loss and accuracy
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

            for epoch in range(FLAGS.num_epoch):
                train_loss = 0
                test_loss = 0

                sess.run(iter_train.initializer)
                sess.run(iter_train_infer.initializer)
                sess.run(iter_test.initializer)

                # training
                for _ in range(num_batches):
                    _, train_loss_ = sess.run([train_op, train_loss_op])
                    train_loss += train_loss_
                summaries = sess.run(write_op, {loss_var: train_loss/num_batches})
                train_summary_writer.add_summary(summaries, epoch)

                # validation
                for _ in range(FLAGS.num_samples):
                    _, test_loss_ = sess.run([pred_op, test_loss_op])
                    test_loss += test_loss_
                summaries = sess.run(write_op, {loss_var: test_loss/FLAGS.num_samples})
                test_summary_writer.add_summary(summaries, epoch)

                print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'%(epoch, (train_loss/num_batches), (test_loss/FLAGS.num_samples)))


if __name__ == "__main__":

    tf.app.run()
