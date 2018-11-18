import subprocess
import argparse
from glob import glob
from os import makedirs,path,remove
from shutil import rmtree
import cv2
from scipy.misc import imsave, imresize, imread
import numpy as np
import tensorflow as tf
from model import generator
import sys
import time

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--model' , default='default', help='Name of the trained model to use.')
add_arg('--batch-size', default=10, type=int, help='Number of images provided at each test iteration.')
add_arg('--video', default='./test1.mp4', type=str, help='Video File Name')
add_arg('--output', default='./output.mp4', type=str, help='Output Video File Name')
add_arg('--idims', default='64x64', help='Input image dimensions.')
add_arg('--odims', default='128x128', help='Output image dimensions.')

args = parser.parse_args()
command = "ffmpeg -i "+args.video + " audio.mp3"
subprocess.call(command, shell=True)

model = args.model
idims = list(map(int, args.idims.split('x')))[::-1]
odims = list(map(int, args.odims.split('x')))[::-1]

fps = 0

def video_gen(video):
    vidcap = cv2.VideoCapture(video)
    global fps
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    image = imresize(np.array(image), size=(idims[0], idims[1], 3))
    batch = np.array([])
    while success:
        batch = np.append(batch, np.array([image]), axis=0) if batch.shape[0] > 0 else np.array([image])
        success, image = vidcap.read()
        if success:
            image = imresize(np.array(image), size=(idims[0], idims[1], 3))
        if batch.shape[0] == args.batch_size:
            yield batch
            batch = np.array([])

def superres():
    sml_x = tf.placeholder(tf.float32, [None,  None,  None, 3])
    ndims = tf.placeholder(tf.int32, [2])
    gener_x = generator(sml_x, ndims, is_training=False, reuse=False)
    init = tf.global_variables_initializer()
    dataset_gen = video_gen(args.video)
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, '/'.join(['models', model, model]))
        except Exception as e:
            print('Model could not be restored. Exiting.\nError: ' + e)
            exit()
        
        print('Saving test results ...')
        i = 0
        start_time = time.time()
        cv2.namedWindow("preview")
        for batch_big in dataset_gen:
            i += 1
            highdims = odims
            superres_imgs = sess.run(
                gener_x, feed_dict={sml_x: batch_big, ndims: highdims})
            if i == 1:
                height, width, layers = odims[0], odims[1], 3
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                print('Using fps: ' + str(fps))
                video = cv2.VideoWriter('./super.mp4', fourcc, fps, (width, height))
            for img in superres_imgs:
                sys.stdout.write('FPS: '+ str(int(1.0*i/(time.time()-start_time)))+'\r')
                img = (img*255).astype(np.uint8)
                video.write(img)
                cv2.imshow('preview', img)
                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
                if key == 32:
                    cv2.waitKey(0)
        cv2.destroyAllWindows()
        video.release()

superres()
command = "ffmpeg -i super.mp4 -i audio.mp3 -c copy -map 0:v:0 -map 1:a:0 "+ args.output
subprocess.call(command, shell=True)

remove('./audio.mp3')
remove('./super.mp4')
print('Successfully Increased Resolution by 2x')
