from Templete_utils import *
from glob import glob
import os
from time import time

class GAN(object):
    def __init__(self, sess, args):
        self.model_name = 'DatasetAPI'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset

        self.batch_size = args.batch_size * args.gpu_num

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.augment_flag = args.augment_flag
        self.augment_size = self.img_size + (30 if self.img_size == 256 else 15)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.num_batches = min(len(self.trainA_dataset), len(self.trainB_dataset)) // self.batch_size

    def network(self, x):
        return x

    def build_model(self):
        """ Input Image"""
        Image_Data_Class = ImageData(self.batch_size, self.img_size, self.img_ch, self.augment_flag)


        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)

        trainA = trainA.map(Image_Data_Class.image_processing).shuffle(10000).prefetch(self.batch_size).batch(self.batch_size).repeat()


        trainA_iterator = trainA.make_initializable_iterator()
        self.trainA_init_op = trainA_iterator.initializer

        self.data_A = trainA_iterator.get_next()

        loss = self.network(self.data_A)


        self.train_op = tf.train.AdamOptimizer(0.1, beta1=0.5, beta2=0.999).minimize(loss)



    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        self.trainA_init_op.run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        for epoch in range(start_epoch, 200):
            for idx in range(start_batch_id, self.num_batches):

                loss = self.sess.run(self.train_op)


                # display training status
                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss))

                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
            self.save(self.checkpoint_dir, counter)



    @property
    def model_dir(self):
        return "{}_{}".format(
            self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
