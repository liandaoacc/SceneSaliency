################################################################################################
# Short command for setting defeault target properties
# Usage:
#   project_default_properties(<target>)
function(project_default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX ${Project_DEBUG_POSTFIX}
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endfunction()

################################################################################################
# Short command for setting runtime directory for build target
# Usage:
#   project_set_runtime_directory(<target> <dir>)
function(project_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()

################################################################################################
# Short command for setting solution folder property for target
# Usage:
#   project_set_solution_folder(<target> <folder>)
function(project_set_solution_folder target folder)
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "${folder}")
  endif()
endfunction()