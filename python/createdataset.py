# -*- coding: utf-8 -*-
import convertdata as cvd
import logging
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import h5py

def showdata(args):
    cvd.checkresize(args.resize_imageW, args.resize_imageH, 
                    args.resize_saliencyW, args.resize_saliencyH)
    cvd.checknormalizerange(args.normalize_image, 
                            args.normalize_saliency)
    filelist = cvd.getfilelist(args.imagefolder, args.saliencyfolder, 
                               args.listfile, args.shuffle)
    
    # get the shape for h5py dataset
    image, saliency = cvd.convertimageandmat(filelist[14], args.gray, 
                                             args.check_size, 
                                             args.resize_imageW, 
                                             args.resize_imageH,
                                             args.resize_saliencyW, 
                                             args.resize_saliencyH, 
                                             args.normalize_image, 
                                             args.normalize_saliency)
    
    with h5py.File('datasetmean.hdf5') as f:
        mset = f.get('mean')
        imagemean = mset[0]
    mean_c = imagemean.mean(1).mean(1)
    for i in range(imagemean.shape[0]):
        imagemean[i].fill(mean_c[i])
    image = image.astype(np.float32)
    image -= imagemean
    image = cvd.chw2hwc(image, args.gray)
    image = cvd.normalizeimage(image, args.normalize_image) 
    saliency = cvd.l2hw(saliency, args.resize_saliencyW, args.resize_saliencyH)
    plt.figure()
    skimage.io.imshow(image)
    plt.figure()
    skimage.io.imshow(saliency)
    skimage.io.show()
        
def createdata(args):
    logger_root = logging.getLogger()
    cvd.checkresize(args.resize_imageW, args.resize_imageH, 
                    args.resize_saliencyW, args.resize_saliencyH)
    cvd.checknormalizerange(args.normalize_image, 
                            args.normalize_saliency)
    filelist = cvd.getfilelist(args.imagefolder, args.saliencyfolder, 
                               args.listfile, args.shuffle)
    
    # get the shape for h5py dataset
    image, saliency = cvd.convertimageandmat(filelist[0], args.gray, 
                                             args.check_size, 
                                             args.resize_imageW, 
                                             args.resize_imageH,
                                             args.resize_saliencyW, 
                                             args.resize_saliencyH, 
                                             args.normalize_image, 
                                             args.normalize_saliency)
                                         
    N = len(filelist)
    datashape = tuple([N]) + image.shape
    labelshape = tuple([N]) + saliency.shape
    
    with h5py.File('datasetmean.hdf5') as f:
        mset = f.get('mean')
        imagemean = mset[0]
    mean_c = imagemean.mean(1).mean(1)
    for i in range(imagemean.shape[0]):
        imagemean[i].fill(mean_c[i])
        
    # write to h5py dataset
    with h5py.File(args.outfile) as f:
        dset = f.create_dataset(name = 'data', shape = datashape, 
                                dtype = np.dtype(np.float32), 
                                compression = "gzip", compression_opts = 4)
        lset = f.create_dataset(name = 'label', shape = labelshape, 
                                dtype = np.dtype(np.float32), 
                                compression = "gzip", compression_opts = 4)
        for i in range(N):
            image, saliency = cvd.convertimageandmat(filelist[i],
                                                     args.gray,
                                                     args.check_size, 
                                                     args.resize_imageW, 
                                                     args.resize_imageH, 
                                                     args.resize_saliencyW, 
                                                     args.resize_saliencyH, 
                                                     args.normalize_image, 
                                                     args.normalize_saliency)
            image = image.astype(np.float32)
            image -= imagemean
            image = cvd.chw2hwc(image, args.gray)
            image = cvd.normalizeimage(image, args.normalize_image)
            image = cvd.hwc2chw(image, args.gray)
            dset[i] = image
            lset[i] = saliency
            if((i + 1) % 100 == 0):
                logger_root.info('Processed %d files.' %(i + 1))
        if((i + 1) % 100 != 0):
            logger_root.info('Processed %d files.' %(i + 1))
    
if __name__ == '__main__':
    cvd.setuplogging()
    # parse argv
    args = cvd.parseArgument()
    createdata(args)