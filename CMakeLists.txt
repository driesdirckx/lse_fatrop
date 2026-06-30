cmake_minimum_required(VERSION 3.14)
project(lse_fatrop_pointmass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# --------------------------------------------------------------------------- #
# Locate fatrop.
#
# Two supported modes (the default is the self-contained source build):
#
#  (A) SOURCE BUILD (default): point FATROP_ROOT at a checkout of fatrop and we
#      add it as a subdirectory. BLASFEO is fetched + built automatically via
#      fatrop's WITH_BUILD_BLASFEO option, so there is nothing else to install.
#
#  (B) INSTALLED PACKAGE: configure with -DUSE_INSTALLED_FATROP=ON and we call
#      find_package(fatrop). Requires that fatrop *and* blasfeo are installed
#      and visible to CMake (CMAKE_PREFIX_PATH).
# --------------------------------------------------------------------------- #
option(USE_INSTALLED_FATROP "Use an installed fatrop via find_package" OFF)
set(FATROP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fatrop"
    CACHE PATH "Path to the fatrop source checkout (source-build mode)")

if(USE_INSTALLED_FATROP)
  find_package(fatrop REQUIRED)
else()
  if(NOT EXISTS "${FATROP_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
      "FATROP_ROOT='${FATROP_ROOT}' does not contain a fatrop checkout.\n"
      "Set -DFATROP_ROOT=/path/to/fatrop, or use -DUSE_INSTALLED_FATROP=ON.")
  endif()
  # Turn off the parts of fatrop we don't need and let it fetch BLASFEO.
  set(WITH_BUILD_BLASFEO     ON  CACHE BOOL "" FORCE)
  set(BUILD_TESTS            OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES         OFF CACHE BOOL "" FORCE)
  set(BUILD_WITH_C_INTERFACE OFF CACHE BOOL "" FORCE)
  add_subdirectory(${FATROP_ROOT} ${CMAKE_BINARY_DIR}/fatrop_build)
endif()

# --------------------------------------------------------------------------- #
# Our executable.
# --------------------------------------------------------------------------- #
add_executable(point_mass_obstacle src/main.cpp)
target_include_directories(point_mass_obstacle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# LSE (log-sum-exp) obstacle example: same dynamics, smooth-max constraint with
# analytical Jacobian/Hessian (src/lse.hpp, src/point_mass_lse_ocp.hpp).
add_executable(point_mass_lse_obstacle src/main_lse.cpp)
target_include_directories(point_mass_lse_obstacle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# The exported/installed target is fatrop::fatrop; the in-tree target is fatrop.
if(USE_INSTALLED_FATROP)
  target_link_libraries(point_mass_obstacle PRIVATE fatrop::fatrop)
  target_link_libraries(point_mass_lse_obstacle PRIVATE fatrop::fatrop)
else()
  target_link_libraries(point_mass_obstacle PRIVATE fatrop)
  target_link_libraries(point_mass_lse_obstacle PRIVATE fatrop)
endif()
