
from argparse import ArgumentParser

parser = ArgumentParser()
add_arg = parser.add_argument
add_arg('--model' , default='default', help='Name of the trained model to use.')
add_arg('--batch-size', default=50, type=int, help='Number of images provided at each test iteration.')
add_arg('--input', default='test/', help='Output dir set in \'prepare.py\'.')
add_arg('--idims', default='64x64', help='Input image dimensions.')
add_arg('--odims', default='128x128', help='Output image dimensions.')
args = parser.parse_args()

from os import makedirs
from scipy.misc import imsave, imresize, imread
from skimage import io
from glob import glob
from time import strftime
from model import generator
import tensorflow as tf
import numpy as np

class SuperRes:
    def __init__(self): 
        print('Reading test images...')
        files = sorted(glob(args.input + '/*.*'))
        self.dataset = np.array([imread(file) for file in files])
        print('Done.')
        self.model = args.model
        self.dataset_size = self.dataset.shape[0]
        self.batch_size = min(self.dataset_size, args.batch_size)
        self.out_path = '/'.join(['output_images', strftime('%Y%m%d-%H%M%S')])
        self.idims = list(map(int, args.idims.split('x')))[::-1]
        self.odims = list(map(int, args.odims.split('x')))[::-1]
        print('Converting from {} to {}'.format(self.idims, self.odims))

    def test(self):
        sml_x = tf.placeholder(tf.float32, [None,  self.idims[0], self.idims[1], 3])
        odims = tf.placeholder(tf.int32, [2])
        gener_x = generator(sml_x, odims, is_training=False, reuse=False)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            try:
                saver.restore(sess, '/'.join(['models', self.model, self.model]))
            except Exception as e:
                print('Model could not be restored. Exiting.\nError: ' + e)
                exit()
            makedirs(self.out_path)
            print('Performing super resolution ...')
            for idx in range(0, self.dataset_size, self.batch_size):
                start, end = idx, min(idx + self.batch_size, self.dataset_size)
                batch = range(start, end)
                batch_big = self.dataset[batch] / 255.0
                batch_sml = np.array([imresize(img, size=(self.idims[0], self.idims[1], 3)) for img in batch_big])
                superres_imgs = sess.run(gener_x, feed_dict={sml_x: batch_sml, odims: self.odims})
                interpolated_imgs = np.array([imresize(img, size=superres_imgs.shape[1:])/255.0 for img in batch_sml])
                images = np.concatenate((interpolated_imgs, superres_imgs), 2)
                for i, image in enumerate(images):
                    imsave('%s/%d.png' % (self.out_path, start+i), image)
                print('%d/%d saved successfully.' % (end, self.dataset_size))

if __name__ == '__main__':
    superres = SuperRes()
    superres.test()
