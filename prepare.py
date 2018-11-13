import argparse
from glob import glob
from random import shuffle
import os, sys
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--train', default='./train/', type=str, help='Path to training images')
add_arg('--test', default='./test/SR_testing_datasets/', type=str, help='Path to testing images')
add_arg('--output', default='./data/', type=str, help='npy\'s will be stored here.')

args = parser.parse_args()


if __name__ == '__main__':
    train_path = args.output + 'train.npy'
    test_path = args.output + 'test.npy'
    train_files = glob(args.train + '/*.jpeg')
    test_files = glob(args.test+ '/**/*.*', recursive=True)
    training_set_size = len(train_files)
    testing_set_size = len(test_files)
    print(testing_set_size)
    shuffle(train_files)
    shuffle(test_files)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    train_imgs = np.empty(shape=(training_set_size, 128, 128, 3), dtype=np.uint8)
    print('Preparing Training Data...')
    for i,infile in enumerate(train_files):
        if i % 1000 == 0:
            print(str(i)+ '/' +str(training_set_size))
        try:
            im = Image.open(infile)
            rgbimg = Image.new('RGB', im.size)
            rgbimg.paste(im)
            img = rgbimg.resize((128,128))
            train_imgs[i] = np.asarray(img)
        except IOError:
            print("cannot prepare '%s'" % infile)
    np.save(train_path,train_imgs)
    
    print('Training Data Done.')
    test_imgs = np.empty(shape=(testing_set_size, 128, 128, 3), dtype=np.uint8)
    print('Preparing Testing Data...')
    for i,infile in enumerate(test_files):
        try:
            im = Image.open(infile)
            rgbimg = Image.new('RGB', im.size)
            rgbimg.paste(im)
            img = rgbimg.resize((128,128))
            test_imgs[i] = np.asarray(img)
        except IOError:
            print("cannot prepare '%s'" % infile)
    print('Testing Data Done.')
    print(test_imgs.shape)
    np.save(test_path,test_imgs)
