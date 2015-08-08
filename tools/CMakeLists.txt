# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# ---[ Includes
find_package(Caffe)

if(${Caffe_HAVE_CUDA})
  message(STATUS "Using CUDA")
else(${Caffe_HAVE_CUDA})
  message(STATUS "No using CUDA")
endif(${Caffe_HAVE_CUDA})

if(${Caffe_HAVE_CUDNN})
  message(STATUS "Using CUDNN")
else(${Caffe_HAVE_CUDNN})
  message(STATUS "No using CUDNN")
endif(${Caffe_HAVE_CUDNN})

find_package(Matio)

add_definitions(${Caffe_DEFINITIONS})
include_directories(${Caffe_INCLUDE_DIRS} ${Matio_INCLUDE_DIR})

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # target
  add_executable(${name} ${source})
 
  target_link_libraries(${name} ${Caffe_LIBRARIES} ${Matio_LIBRARY})
  
  project_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  project_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
  project_set_solution_folder(${name} tools)

  # Install
  install(TARGETS ${name} DESTINATION bin)
endforeach(source)
