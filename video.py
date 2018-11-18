import subprocess
import argparse
from glob import glob
from os import makedirs,path,remove
from shutil import rmtree
import cv2
from PIL import Image
from scipy.misc import imsave, imresize, imread
import numpy as np
import tensorflow as tf
from model import generator

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg('--video', default='./test1.mp4', type=str, help='Video File Name')
add_arg('--output', default='./output.mp4', type=str, help='Output Video File Name')

args = parser.parse_args()
command = "ffmpeg -i "+args.video + " audio.mp3"
subprocess.call(command, shell=True)

model = 'new_model'

if not path.exists('./frames'):
    makedirs('./frames')

if not path.exists('./frames_out'):
    makedirs('./frames_out')
fps=0

def split():
    vidcap = cv2.VideoCapture(args.video)
    global fps
    fps=vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("./frames/%d.jpg" %count, image)   
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def stitch():
    for i in range(num_frames):
        img1 = cv2.imread('./frames_out/{0}.jpg'.format(i))
        if(i == 0):
            height, width, layers = img1.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter('./super.mp4', fourcc, fps, (width, height))
        video.write(img1)
    cv2.destroyAllWindows()
    video.release()


def superres():
    sml_x = tf.placeholder(tf.float32, [None,  None,  None, 3])
    ndims = tf.placeholder(tf.int32, [2])
    gener_x = generator(sml_x, ndims, is_training=False, reuse=False)
    init = tf.global_variables_initializer()
    dataset = np.empty(shape=(num_frames,64,64,3),dtype=np.uint8)
    for i in range(num_frames):
        im = Image.open('./frames/%d.jpg'%i)
        dataset[i] = np.asarray(im)
    
    batch_size=20
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, '/'.join(['models', model, model]))
        except Exception as e:
            print('Model coult not be restored. Exiting.\nError: ' + e)
            exit()
        
        print('Saving test results ...')
        start = 0
        for idx in range(0, num_frames, batch_size):
            start, end = idx, min(idx + batch_size, num_frames)
            batch = range(start, end)
            dims = dataset[0].shape
            print(batch)
            highdims = [dims[0]*2, dims[1]*2]
            print(highdims)
            batch_big = dataset[batch] 
            print(batch_big.shape)
            superres_imgs = sess.run(
                gener_x, feed_dict={sml_x: batch_big, ndims: highdims})
            for idx, superres_img in enumerate(superres_imgs):
                imsave('./frames_out/%d.jpg' % (start+idx), superres_img)
            start += batch_size
            print('%d/%d saved successfully.' % \
                        (min(start, num_frames), num_frames))


split()
frames = glob('./frames/*.jpg')
num_frames = len(frames)
print(num_frames)
superres()
stitch()
command = "ffmpeg -i super.mp4 -i audio.mp3 -c copy -map 0:v:0 -map 1:a:0 "+ args.output
subprocess.call(command, shell=True)

remove('./audio.mp3')
remove('./super.mp4')
rmtree('./frames')
rmtree('./frames_out')
print('Successfully Increased Resoluion by 2x')
