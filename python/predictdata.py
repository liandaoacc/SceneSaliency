# -*- coding: utf-8 -*-
import sys
import logging
import convertdata
import numpy as np
import skimage.io, skimage.color, skimage.transform

scenesaliency_root = '/home/humt/SoftwareProgram/SceneSaliency/Sources/SceneSaliency/'
caffe_root = scenesaliency_root + 'caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
             
if __name__ == '__main__':
    
    convertdata.setuplogging()
    
    # parse argv
    args = convertdata.parseArgument()
    
    if args.resize_imageW <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_imageW should be greater than 0')
        sys.exit(1)
    if args.resize_imageH <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_imageH should be greater than 0')
        sys.exit(1)   

    if args.resize_saliencyW <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_saliencyW should be greater than 0')
        sys.exit(1)
    if args.resize_saliencyH <= 0:
        logger_root = logging.getLogger()
        logger_root.error('resize_saliencyH should be greater than 0')
        sys.exit(1)    
        
    filelist = convertdata.getfilelist(args.imagefolder, args.saliencyfolder,
                                       args.listfile, args.shuffle)
         
    imageHWC, saliency = convertdata.readimageandmat(filelist[8], 
                                                     args.gray, args.check_size, 
                                                     args.resize_imageW, 
                                                     args.resize_imageH,
                                                     args.resize_saliencyW, 
                                                     args.resize_saliencyH)                      
    imageCHW = imageHWC.transpose((2, 0, 1))
    saliency = saliency.flatten()
    
    input_arrays = np.zeros([1,3,96,96],np.float32)
    input_arrays[0] = imageCHW
    
    caffe.set_mode_cpu()
    net = caffe.Net(scenesaliency_root + 'model/deploy.prototxt',
                    scenesaliency_root + 'model/snapshot_iter_2000.caffemodel',
                    caffe.TEST)
    net.blobs['data'].data[...] = input_arrays
    blobs = [(k, v.data.shape) for k, v in net.blobs.items()]
    out = net.forward()
    output_arrays = out['fc6']
    
    