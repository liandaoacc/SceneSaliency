# - Try to find Caffe
#
# The following variables are optionally searched for defaults
#  Caffe_ROOT_DIR:            Base directory where all Caffe components are found
#
# The following are set after configuration is done:
#
#   Caffe_INCLUDE_DIRS - Caffe include directories
#   Caffe_LIBRARIES    - libraries to link against
#   Caffe_DEFINITIONS  - a list of definitions to pass to compiler
#
#   Caffe_HAVE_CUDA    - signals about CUDA support
#   Caffe_HAVE_CUDNN   - signals about cuDNN support

find_package(OpenCV REQUIRED)
set(Caffe_ROOT_DIR "" CACHE PATH "Folder contains Caffe")

find_path(Caffe_CONF_DIRS NAMES CaffeConfig.cmake
                             PATHS ${Caffe_ROOT_DIR} ${Caffe_ROOT_DIR}/share/Caffe)
                             
include(${Caffe_CONF_DIRS}/CaffeConfig.cmake)