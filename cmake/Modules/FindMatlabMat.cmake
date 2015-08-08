# This module looks for MatlabMat library
# Defines variables:
#    Matlab_DIR    - Matlab root dir
#    Matlab_INCLUDE_DIR - path to matlab include directory
#    Matlab_LIBRARIES - path to matlab library

if(MSVC)
  foreach(__ver "9.30" "7.14" "7.11" "7.10" "7.9" "7.8" "7.7")
    get_filename_component(__matlab_root "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\${__ver};MATLABROOT]" ABSOLUTE)
    if(__matlab_root)
      break()
    endif()
  endforeach()
endif()

if(APPLE)
  foreach(__ver "R2014b" "R2014a" "R2013b" "R2013a" "R2012b" "R2012a" "R2011b" "R2011a" "R2010b" "R2010a")
    if(EXISTS /Applications/MATLAB_${__ver}.app)
      set(__matlab_root /Applications/MATLAB_${__ver}.app)
      break()
    endif()
  endforeach()
endif()

if(UNIX)
   execute_process(COMMAND which matlab OUTPUT_STRIP_TRAILING_WHITESPACE
                   OUTPUT_VARIABLE __out RESULT_VARIABLE __res)

   if(__res MATCHES 0) # Suppress `readlink` warning if `which` returned nothing
     execute_process(COMMAND which matlab  COMMAND xargs readlink
                     COMMAND xargs dirname COMMAND xargs dirname COMMAND xargs echo -n
                     OUTPUT_VARIABLE __matlab_root OUTPUT_STRIP_TRAILING_WHITESPACE)
   endif()
endif()


find_path(Matlab_DIR NAMES bin/mex bin/mexext PATHS ${__matlab_root}
                     DOC "Matlab directory" NO_DEFAULT_PATH)

find_path(Matlab_INCLUDE_DIR NAMES mat.h mex.h PATHS ${Matlab_DIR}/extern/include DOC "Path to matlab include directory" NO_DEFAULT_PATH)
find_library(Matlab_mat NAMES mat PATHS ${Matlab_DIR}/bin/glnxa64 DOC "Path to libmat.so" NO_DEFAULT_PATH)
find_library(Matlab_mx NAMES mx PATHS ${Matlab_DIR}/bin/glnxa64  DOC "Path to libmx.so" NO_DEFAULT_PATH)
find_library(Matlab_ut NAMES ut PATHS ${Matlab_DIR}/bin/glnxa64  DOC "Path to libut.so" NO_DEFAULT_PATH)
  
set(Matlab_INCLUDE_DIR ${Matlab_INCLUDE_DIR})
list(APPEND Matlab_LIBRARIES ${Matlab_mat} ${Matlab_mx} ${Matlab_ut})
  
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MatlabMat DEFAULT_MSG Matlab_DIR Matlab_INCLUDE_DIR Matlab_LIBRARIES)
 
if(MATLABMAT_FOUND)
  mark_as_advanced(Matlab_DIR Matlab_INCLUDE_DIR Matlab_LIBRARIES)
endif(MATLABMAT_FOUND)