# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
mark_as_advanced(CMAKE_CONFIGURATION_TYPES)

if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

# --[ If user doesn't specify build type then assume release
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX TRUE)
endif()

# ---[ Solution folders
project_option(USE_PROJECT_FOLDERS "IDE Solution folders" (MSVC_IDE OR CMAKE_GENERATOR MATCHES Xcode) )

if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ Set debug postfix
set(Project_DEBUG_POSTFIX "-d")

set(Project_POSTFIX "")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(Project_POSTFIX ${Project_DEBUG_POSTFIX})
endif()