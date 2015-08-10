import yaml
import logging
import logging.config
import logging.handlers
import argparse
import os.path
import sys
import numpy as np
import scipy.io
import skimage.io, skimage.color, skimage.transform
import h5py

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
                        all the item have the same size')
    parser.add_argument('--resize_width', metavar = 'width', type = int, 
                        nargs = '?', const = 96, default = 96, 
                        help = 'Width images are resized to')
    parser.add_argument('--resize_height', metavar = 'height', type = int, 
                        nargs = '?', const = 96, default = 96, 
                        help = 'Height images are resized to')
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
 
def readimageandmat(files, gray, check_size, resize_width, resize_height):
    image = skimage.io.imread(files[0])
    mat_content = scipy.io.loadmat(files[1])
    saliency = mat_content['I']
    
    if(check_size):
        if(image.shape[0:2] != saliency.shape):
            logger_root = logging.getLogger()
            logger_root.error('size of %s is not equal to size %s'
            %(files[0], files[1]))
            return None
    
    # make sure resize_width and resize_height is good
    resize_width = resize_width if resize_width > 0 else 96
    resize_height = resize_height if resize_height > 0 else 96
    
    if(resize_width > 0 and resize_height > 0):
        image = skimage.transform.resize(image, (resize_height, resize_width))
        saliency = skimage.transform.resize(saliency, 
                                            (resize_height, resize_width))

    if(gray):
        image = skimage.color.rgb2gray(image)
    image = image.astype(np.float32)
    saliency = saliency.astype(np.float32)
    return image, saliency

    
if __name__ == '__main__':
    
    setuplogging()
    
    # parse argv
    args = parseArgument()
    
    if args.resize_width <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_width should be greater than 0')
        sys.exit(1)
    if args.resize_height <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_height should be greater than 0')
        sys.exit(1)    
    
    filelist = getfilelist(args.imagefolder, args.saliencyfolder, 
                           args.listfile, args.shuffle)
    
    # get the shape for h5py dataset
    imageHWC, saliency = readimageandmat(filelist[0], args.gray, 
                                         args.check_size, args.resize_width, 
                                         args.resize_height)
    imageCHW = imageHWC.transpose((2, 0, 1))
    N = len(filelist)
    datashape = tuple([N]) + imageCHW.shape
    labelshape = tuple([N, 1]) + saliency.shape
    
    # write to h5py dataset
    with h5py.File("mytraining.hdf5") as f:
        dset = f.create_dataset(name = 'data', shape = datashape, 
                         dtype = np.dtype(np.float32), 
                         compression="gzip", compression_opts=4)
        lset = f.create_dataset(name = 'label', shape = labelshape, 
                         dtype = np.dtype(np.float32), 
                         compression="gzip", compression_opts=4)
        for i in range(N):
            imageHWC, saliency = readimageandmat(filelist[i], args.gray, 
                                         args.check_size, args.resize_width, 
                                         args.resize_height)
            imageCHW = imageHWC.transpose((2, 0, 1))
            dset[i] = imageCHW
            lset[i] = saliency