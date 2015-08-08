find_path(Matio_INCLUDE_DIR NAMES matio.h PATHS /usr/include /usr/local/include DOC "Path to matio include directory")
find_library(Matio_LIBRARY NAMES matio PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib DOC "Path to libmatio.so")

set(Matio_INCLUDE_DIR ${Matio_INCLUDE_DIR})
set(Matio_LIBRARY ${Matio_LIBRARY})
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Matio DEFAULT_MSG Matio_INCLUDE_DIR Matio_LIBRARY)
 
if(MATIO_FOUND)
  mark_as_advanced(Matlab_DIR Matio_INCLUDE_DIR Matio_LIBRARY)
endif(MATIO_FOUND)