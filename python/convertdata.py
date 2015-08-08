import logging
import logging.config
import argparse
import os.path
import numpy as np
import scipy.io
import skimage.io
import skimage.color
import h5py


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
        #TODO:
        a = 1
    print image.shape, saliency.shape
    image_gray = skimage.color.rgb2gray(image)
    skimage.io.imshow(image_gray)
       
if __name__ == '__main__':
    args = parseArgument()
    logging.config.dictConfig('logging1.conf')
    filelist = getfilelist(args.imagefolder, args.saliencyfolder, 
                           args.listfile, args.shuffle)
    readimageandmat(filelist[0], args.gray, args.check_size, 
                    args.resize_width, args.resize_height)

            


