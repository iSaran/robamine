cmake_minimum_required(VERSION 2.8.3)

project(rlrl)

# Use C++  for initialise an std::vector without a constructor
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Build in release mode
set(CMAKE_BUILD_TYPE Release)

find_package(cmake_modules REQUIRED)
find_package(catkin REQUIRED
  autharl_core
)

set(RLRL_INCLUDE_DIRS
  include
  core/include
  policy/include
)

catkin_package(
  INCLUDE_DIRS
    ${RLRL_INCLUDE_DIRS}
  LIBRARIES
    rlrl_core
    rlrl_policy
  CATKIN_DEPENDS
    autharl_core
#  DEPENDS system_lib
)

include_directories(
  ${RLRL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_subdirectory(core)
add_subdirectory(policy)
