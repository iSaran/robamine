# Set the name of the library
set(RLRL_LIB_NAME rlrl_core)

# List the source files of the library
set(SRC_FILES
  src/action.cpp
  src/mdp.cpp
  src/policy.cpp
  src/reward.cpp
  src/rollout.cpp
  src/state.cpp
)

# List other libraries that need to be linked
set(OTHER_LIBS_TO_LINK)

# Build the library and the tests
add_library(${RLRL_LIB_NAME} ${SRC_FILES})

# set_target_properties(${AUTHARL_LIB_NAME} PROPERTIES VERSION ${${PROJECT_NAME}_VERSION})

target_link_libraries(${RLRL_LIB_NAME}
  ${catkin_LIBRARIES}
  ${OTHER_LIBS_TO_LINK})

add_dependencies(${RLRL_LIB_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS})
