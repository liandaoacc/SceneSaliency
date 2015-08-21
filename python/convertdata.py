# -*- coding: utf-8 -*-
import yaml
import logging
import logging.config
import logging.handlers
import argparse
import os.path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import skimage.io, skimage.color, skimage.transform, skimage.exposure

def setuplogging():
    # set log system
    with open('logging.conf', 'r') as f:
        conf = yaml.load(f)
    logging.config.dictConfig(conf)
    
def parseArgument():
    parser = argparse.ArgumentParser(description='Prepare the data.')
    parser.add_argument('--gray', action = 'store_true',                         
                        help = 'When this option is on, treat images \
                        as grayscale ones')
    parser.add_argument('--shuffle', action = 'store_true',                         
                        help = 'When this option is on, randomly \
                        shuffle the order of images and their labels')
    parser.add_argument('--check_size', action = 'store_true',                         
                        help = 'When this option is on, check that \
                        image and saliency have the same size')
    parser.add_argument('--resize_imageW', metavar = 'image width', type = int, 
                        nargs = '?', const = 96, default = 96, 
                        help = 'Width images are resized to')
    parser.add_argument('--resize_imageH', metavar = 'image height', type = int, 
                        nargs = '?', const = 96, default = 96, 
                        help = 'Height images are resized to')
    parser.add_argument('--normalize_image', metavar = 'normalization range', 
                        type = int, nargs = 2, default = [0, 255],
                        help = 'Range images are normalizated to')
    parser.add_argument('--resize_saliencyW', metavar = 'saliency width', 
                        type = int, nargs = '?', const = 48, default = 48, 
                        help = 'Width saliencys are resized to')
    parser.add_argument('--resize_saliencyH', metavar = 'saliency height', 
                        type = int, nargs = '?', const = 48, default = 48, 
                        help = 'Height saliency are resized to')
    parser.add_argument('--normalize_saliency', metavar = 'normalization range', 
                        type = int, nargs = 2, default = [0, 1],
                        help = 'Range saliencys are normalizated to')
    parser.add_argument('--imagefolder', metavar = 'str', 
                        type = str, required = True, 
                        help = 'imagefolder is the root folder \
                        that holds all the images')        
    parser.add_argument('--saliencyfolder', metavar = 'str', 
                        type = str, required = True, 
                        help = 'saliencyfolder is the root folder \
                        that holds all the ground truth saliency map')
    parser.add_argument('--listfile', metavar = 'str', 
                        type = str, required = True, 
                        help = 'listfile should be a list \
                        of files, in the format as  \
                        subfolder1/file1.JPEG subfolder1/file1.mat')
    parser.add_argument('outfile', type = str, 
                        help = 'outfile is the name of the output file')
    args = parser.parse_args()
    return args

def getfilelist(imagefolder, saliencyfolder, listfile, shuffle):
    with open(listfile, 'r') as f:
        filelist = []
        for line in f:
            line = line.strip()
            files = line.split('\t')
            files[0] = os.path.join(imagefolder, files[0])
            files[1] = os.path.join(saliencyfolder, files[1])
            filelist.append(files)
        filelist_np =  np.array(filelist)
        if(shuffle):
            np.random.shuffle(filelist_np)
        return filelist_np

def checknormalizerange(normalize_image, normalize_saliency):
    logger_root = logging.getLogger()
    
    if((normalize_image[0] < 0) or (normalize_image[1] < 0) or 
       (normalize_image[0] == normalize_image[1])):
        logger_root.info('normalize_image is illegal, use default [0 255]')
        normalize_image = [0, 255]
    if(normalize_image[0] > normalize_image[1]):
        temp = normalize_image[0]
        normalize_image[0] = normalize_image[1]
        normalize_image[1] = temp
    
    if((normalize_saliency[0] < 0) or (normalize_saliency[1] < 0) or 
       (normalize_saliency[0] == normalize_saliency[1])):
        logger_root.info('normalize_saliency is illegal, use defualt [0 1]')
        normalize_saliency = [0, 1]
    if(normalize_saliency[0] > normalize_saliency[1]):
        temp = normalize_saliency[0]
        normalize_saliency[0] = normalize_saliency[1]
        normalize_saliency[1] = temp

def checkresize(resize_imageW, resize_imageH, resize_saliencyW, resize_saliencyH):
    logger_root = logging.getLogger()
    
    if(resize_imageW < 0):
        logger_root.info('resize_imageW is illegal, use default 96')
        resize_imageW = 96
    if(resize_imageH < 0):
        logger_root.info('resize_imageH is illegal, use default 96')
        resize_imageH = 96
    if(resize_saliencyW < 0):
        logger_root.info('resize_saliencyW is illegal, use default 48')
        resize_saliencyW = 48
    if(resize_saliencyH < 0):
        logger_root.info('resize_saliencyH is illegal, use default 48')
        resize_saliencyH = 48       

def readimage(imagename, gray):
    logger_root = logging.getLogger()
    image = skimage.io.imread(imagename)
    if(image.shape[2] < 3):
        logger_root.info('%s is not rgb of rgba'%(imagename))
        return None   
    if(gray):
        image = skimage.color.rgb2gray(image)
        return image
    if(image.shape[2] == 4):
        logger_root.info('%s is rgba'%(imagename))
        image = image[:,:,0:3]
    return image

def normalizeimage(image, normalize_range):
    image = skimage.exposure.rescale_intensity(image, out_range = normalize_range)
    # range [0.0 255.0] means image is uint8
    if(normalize_range[0] == 0 and normalize_range[1] == 255):
        return image.astype(np.uint8)
    return image.astype(np.float32)
    
def processimage(image, resize_W, resize_H, normalize_range):
    image = skimage.transform.resize(image, (resize_H, resize_W))
    image = skimage.exposure.rescale_intensity(image, out_range = normalize_range)
    # range [0.0 255.0] means image is uint8
    if(normalize_range[0] == 0 and normalize_range[1] == 255):
        return image.astype(np.uint8)
    return image.astype(np.float32)

# scikit image, 2D multichannel is (row, col, ch), 2D grayscale is (row, col)
# caffe, blob is (N, ch, row, col)
def hwc2chw(image, gray):
    if(gray):
        size = image.shape + tuple([1])
        image = image.reshape(size)
    return image.transpose((2, 0, 1))

def chw2hwc(image, gray):
    if(gray):
        return image[0]
    return image.transpose((1, 2, 0))

def readsaliency(saliencyname):
    mat_content = scipy.io.loadmat(saliencyname)
    saliency = mat_content['I']
    return saliency

def normalizesaliency(saliency, normalize_range):
    saliency = skimage.exposure.rescale_intensity(saliency, 
                                                  out_range = normalize_range)
    # range [0.0 255.0] means image is uint8
    if(normalize_range[0] == 0 and normalize_range[1] == 255):
        return saliency.astype(np.uint8)
    return saliency.astype(np.float32)
    
def processsaliency(saliency, resize_W, resize_H, normalize_range):
    saliency = skimage.transform.resize(saliency, (resize_H, resize_W))
    saliency = skimage.exposure.rescale_intensity(saliency, 
                                                  out_range = normalize_range)
    # range [0.0 255.0] means image is uint8
    if(normalize_range[0] == 0 and normalize_range[1] == 255):
        return saliency.astype(np.uint8)
    return saliency.astype(np.float32)

def hw2l(saliency):
    return saliency.flatten()
    
def l2hw(saliency, resize_saliencyW, resize_saliencyH):
    return saliency.reshape((resize_saliencyH, resize_saliencyW))
                                                        
def convertimageandmat(files, gray, check_size, resize_imageW, resize_imageH, 
                    resize_saliencyW, resize_saliencyH, normalize_image, 
                    normalize_saliency):
    logger_root = logging.getLogger()
    
    image = readimage(files[0], gray)
    saliency = readsaliency(files[1])

    if(check_size):
        if(image.shape[0:2] != saliency.shape):
            logger_root.error('size of %s is not equal to size %s'
            %(files[0], files[1]))
            return None
    image = processimage(image, resize_imageW, resize_imageH, normalize_image)
    saliency = processsaliency(saliency, resize_saliencyW, resize_saliencyH, 
                               normalize_saliency)
    image = hwc2chw(image, gray)
    saliency = hw2l(saliency)

    return image, saliency

def convertimage(imagename, gray, resize_W, resize_H, normalize_image):
    image = readimage(imagename, gray)
    image = processimage(image, resize_W, resize_H, normalize_image)
    image = hwc2chw(image, gray)
    return image
 
def convertsaliency(saliencyname, resize_W, resize_H, normalize_saliency):
    saliency = readsaliency(saliencyname)
    saliency = processsaliency(saliency, resize_W, resize_H, normalize_saliency)
    saliency = hw2l(saliency)
    return saliency
    
if __name__ == '__main__':
    setuplogging()
    # parse argv
    args = parseArgument()
    filelist = getfilelist(args.imagefolder, args.saliencyfolder, 
                           args.listfile, args.shuffle)
    image1, saliency1 = convertimageandmat(filelist[0], args.gray,
                                           args.check_size, 
                                           args.resize_imageW, 
                                           args.resize_imageH, 
                                           args.resize_saliencyW, 
                                           args.resize_saliencyH, 
                                           args.normalize_image, 
                                           args.normalize_saliency)
    image1 = chw2hwc(image1, args.gray)
    saliency1 = l2hw(saliency1, args.resize_saliencyW, args.resize_saliencyH)
    
    image2 = convertimage(filelist[0][0], args.gray, args.resize_imageW, 
                          args.resize_imageH, args.normalize_image)
    image2 = chw2hwc(image2, args.gray)
    
    saliency2 = convertsaliency(filelist[0][1], args.resize_saliencyW, 
                                args.resize_saliencyH, args.normalize_saliency)
    saliency2 = l2hw(saliency2, args.resize_saliencyW, args.resize_saliencyH)

    plt.figure()
    skimage.io.imshow(image1)
    plt.figure()
    skimage.io.imshow(saliency1)
    plt.figure()
    skimage.io.imshow(image2)
    plt.figure()
    skimage.io.imshow(saliency2)
    
    skimage.io.show()