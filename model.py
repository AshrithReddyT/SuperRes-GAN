import tensorflow as tf

def generator(x, new_dims, is_training=True, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:            
            scope.reuse_variables()
        conv1 = conv2d(x, output_dim=32, stride=1)
        conv1 = lrelu(conv1)
        
        conv2 = conv2d(conv1, output_dim=128, stride=1)
        conv2 = lrelu(conv2)

        conv3 = conv2d(conv2, output_dim=128, stride=1)
        conv3 = lrelu(conv3)

        resize = tf.image.resize_images(conv3, size=new_dims)

        conv4 = conv2d(resize, output_dim=128, stride=1)
        conv4 = lrelu(conv4)

        conv5 = conv2d(conv4, output_dim=64, stride=1)
        conv5 = lrelu(conv5)

        conv6 = conv2d(conv5, output_dim=3, stride=1)
        out = tf.nn.sigmoid(conv6)

    return out

def discriminator(images, is_training=True, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
 
        conv1 = conv2d(images, output_dim=64, kernel=7, stride=1)
        conv1 = lrelu(conv1)
        
        conv2 = conv2d(conv1, output_dim=64, kernel=7, stride=2)
        conv2 = lrelu(conv2)
            
        conv3 = conv2d(conv2, output_dim=32, stride=2)
        conv3 = lrelu(conv3)

        conv4 = conv2d(conv3, output_dim=1, stride=2)
        conv4 = lrelu(conv4)

        conv4 = tf.contrib.layers.flatten(conv4)
        out = dense(conv4 , 1)
    
    return out

def conv2d(x, output_dim, kernel=3, stride=2, padding='SAME'):
    return tf.layers.conv2d(x, output_dim, [kernel, kernel], strides=(stride, stride), padding=padding)

def dense(x,output_size, activation=tf.nn.relu):
    return tf.layers.dense(x , output_size , activation)

def lrelu(x, threshold=0.01):
    return tf.maximum(x, threshold*x)
