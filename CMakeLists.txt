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
# Two supported modes (the default is the pre-installed package):
#
#  (A) INSTALLED PACKAGE (default): fatrop + BLASFEO were built and installed
#      under FATROP_INSTALL_PREFIX (see README §4). We add that prefix to
#      CMAKE_PREFIX_PATH and call find_package(blasfeo) then find_package(fatrop)
#      (fatropConfig references the imported 'blasfeo' target, so blasfeo must be
#      found first).
#
#  (B) SOURCE BUILD: configure with -DUSE_INSTALLED_FATROP=OFF and point
#      FATROP_ROOT at a fatrop checkout; we add it as a subdirectory and let it
#      fetch + build BLASFEO. Slower (compiles fatrop too) but needs no install.
# --------------------------------------------------------------------------- #
option(USE_INSTALLED_FATROP "Use an installed fatrop via find_package" ON)
set(FATROP_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../fatrop/install"
    CACHE PATH "Install prefix of fatrop + blasfeo (installed-package mode)")
set(FATROP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fatrop"
    CACHE PATH "Path to the fatrop source checkout (source-build mode)")

if(USE_INSTALLED_FATROP)
  set(CMAKE_PREFIX_PATH "${FATROP_INSTALL_PREFIX}" ${CMAKE_PREFIX_PATH})
  # fatropConfig.cmake references the imported 'blasfeo' target without a
  # find_dependency(), so blasfeo must be discovered first.
  find_package(blasfeo REQUIRED)
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
  # fatrop's own CMake refers to its headers via ${CMAKE_SOURCE_DIR}/include, which
  # under add_subdirectory points at THIS project instead of fatrop. Inject the
  # real fatrop include dir (inherited by the fatrop subdirectory targets) so its
  # sources — and ours — can find <fatrop/...> headers.
  include_directories(${FATROP_ROOT}/include)
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
