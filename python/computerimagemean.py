# -*- coding: utf-8 -*-
import numpy as np
import convertdata as cvd
import logging
import h5py

def computerimagemean(args):
    logger_root = logging.getLogger()
    filelist = cvd.getfilelist(args.imagefolder, args.saliencyfolder, 
                               args.listfile, args.shuffle)
    imagelist = filelist[:,0]
    image = cvd.convertimage(imagelist[0], args.gray, args.resize_imageW, 
                             args.resize_imageH, args.normalize_image)
    N = len(imagelist)
    imagemean = np.zeros(image.shape, dtype = np.float64)
    for i in range(N):
        image = cvd.convertimage(imagelist[i], args.gray, 
                                         args.resize_imageW, 
                                         args.resize_imageH, 
                                         args.normalize_image)
        imagemean += image
        if((i + 1) % 100 == 0):
            logger_root.info('Processed %d files.' %(i + 1))
    if((i + 1) % 100 != 0):
        logger_root.info('Processed %d files.' %(i + 1))
    imagemean /= (N + 1)
    # range [0.0 255.0] means image is uint8
    if(args.normalize_image[0] == 0 and args.normalize_image[1] == 255):
        return imagemean.astype(np.uint8)
    return imagemean.astype(np.float32)
    
def computerandsavemean(args):
    imagemean = computerimagemean(args)
    meanshape = tuple([1]) + imagemean.shape
    # write to h5py dataset
    with h5py.File('datasetmean.hdf5') as f:
        mset = f.create_dataset(name = 'mean', shape = meanshape, 
                                dtype = np.dtype(np.float32), 
                                compression = "gzip", compression_opts = 4)
        mset[0] = imagemean

if __name__ == '__main__':
    cvd.setuplogging()
    # parse argv
    args = cvd.parseArgument()
    computerandsavemean(args)