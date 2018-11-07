import tensorflow as tf
import numpy as np
import os, time
from scipy.misc import imresize
from model import generator,discriminator

isTraining = tf.placeholder(dtype=tf.bool)

batch_size = 9
learning_rate = 1e-4
epochs = 10
model = 'new_model'
ndims = 128,128
train_images = np.load('./data/train.npy',mmap_mode='r')[:10000]

highres = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
lowres = tf.placeholder(tf.float32, shape=(None, 64, 64, 3))

#create Networks
g=generator(lowres,ndims,is_training=isTraining)

d_real = discriminator(highres, is_training = isTraining)
d_fake  = discriminator(g, is_training = isTraining ,reuse = True)

#compute Loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake))) * 0.1 + tf.reduce_mean(tf.abs(tf.subtract(g, highres)))

#Variables
d_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
g_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

#optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_training_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_training_vars)


if not os.path.isdir('models'):
    os.mkdir('models')

#Session Configuration
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

#Training
print('Started Training')
training_start_time = time.time()
for epoch in range(epochs):
    g_losses_this_epoch = []
    d_losses_this_epoch = []
    iters = int(train_images.shape[0]/batch_size)
    epoch_start_time = time.time()
    for iter in range(iters):
        X = train_images[iter*batch_size:(iter+1)*batch_size] / 255.0
        Z = np.array([imresize(img, size=(64, 64, 3)) for img in X]) 
        loss_g_this_iter, loss_d_this_iter,  _ = sess.run([g_loss,d_loss,g_optimizer], {lowres: Z, highres: X, isTraining: True})
        sess.run([d_loss,d_optimizer], {highres: X, lowres: Z, isTraining: True})
        if iter%25 == 0:
          print('Epoch: %d, Passed iterations=%d/%d, Generative cost=%.9f, Discriminative cost=%.9f' % (epoch, iter*batch_size, iters*batch_size, loss_g_this_iter, loss_d_this_iter))

    saver.save(sess, '/'.join(['models', str(epoch), model]))

    epoch_end_time = time.time()
    total_epoch_time = time.strftime("%H:%M:%S", time.gmtime(int(epoch_end_time - epoch_start_time)))

saver.save(sess, '/'.join(['models', model, model]))
            
print('Model \'%s\' saved in: \'%s/\'' % (model, '/'.join(['models', model])))
training_end_time = time.time()
total_training_time = time.strftime("%H:%M:%S", time.gmtime(int(training_end_time - training_start_time)))

print("Training Complete")
print('Total time taken for training: {0} epochs was '.format(str(epochs)) + total_training_time)

sess.close()
