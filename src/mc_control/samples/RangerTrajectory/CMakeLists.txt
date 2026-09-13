#
# Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
#

# The controller configuration is configured twice with different state search
# paths. The installed copy points at the install prefix, as every other FSM
# sample does. The build-tree copy points at the build tree, because the ticker
# configs under rolling-contact-report/config and the lifecycle test below run
# straight out of build/ and would otherwise fail to find MetaTasks.so.
mc_rtc_set_prefix(STATES mc_controller/fsm/states)
set(RANGER_TRAJECTORY_STATES_LIBRARIES "${MC_STATES_RUNTIME_DESTINATION_PREFIX}")
set(RANGER_TRAJECTORY_STATES_FILES "${MC_STATES_RUNTIME_DESTINATION_PREFIX}/data")
configure_file(
  etc/RangerTrajectory.in.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/install/RangerTrajectory.yaml
)
configure_file(
  etc/RangerTrajectory.in.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/install/RangerTrajectory_TVM.yaml
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/install/RangerTrajectory.yaml
              ${CMAKE_CURRENT_BINARY_DIR}/install/RangerTrajectory_TVM.yaml
        DESTINATION ${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc/
)

set(RANGER_TRAJECTORY_STATES_LIBRARIES
    "${PROJECT_BINARY_DIR}/src/mc_control/fsm/states"
)
set(RANGER_TRAJECTORY_STATES_FILES
    "${PROJECT_SOURCE_DIR}/src/mc_control/fsm/states/data"
)
configure_file(
  etc/RangerTrajectory.in.yaml ${CMAKE_CURRENT_BINARY_DIR}/etc/RangerTrajectory.yaml
)
configure_file(
  etc/RangerTrajectory.in.yaml
  ${CMAKE_CURRENT_BINARY_DIR}/etc/RangerTrajectory_TVM.yaml
)

add_subdirectory(src)

if(BUILD_TESTING)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)

  add_executable(testRangerTrajectoryControllerLifecycle test_controller_lifecycle.cpp)
  target_include_directories(
    testRangerTrajectoryControllerLifecycle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_compile_definitions(
    testRangerTrajectoryControllerLifecycle
    PRIVATE
      RANGER_TRAJECTORY_CONFIG_PATH="${CMAKE_CURRENT_BINARY_DIR}/etc/RangerTrajectory.yaml"
      RANGER_TRAJECTORY_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:ranger_mini_v3>"
      RANGER_TRAJECTORY_OBSERVER_MODULE_PATH="$<TARGET_FILE_DIR:EncoderObserver>"
      BOOST_TEST_DYN_LINK
      BOOST_TEST_MAIN
  )
  target_link_libraries(
    testRangerTrajectoryControllerLifecycle PRIVATE ranger_trajectory
                                                    Boost::unit_test_framework
  )
  add_dependencies(
    testRangerTrajectoryControllerLifecycle ranger_mini_v3 EncoderObserver
    BodySensorObserver MetaTasks
  )
  add_test(NAME RangerTrajectoryControllerLifecycle
           COMMAND testRangerTrajectoryControllerLifecycle
  )
  set_tests_properties(
    RangerTrajectoryControllerLifecycle
    PROPERTIES TIMEOUT 300 ENVIRONMENT
               "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
  )

  # The control points committed in etc/RangerTrajectory.in.yaml are generated;
  # this fails if the file and the generator have drifted apart.
  add_test(
    NAME RangerTrajectoryWaypointsUpToDate
    COMMAND
      "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/scripts/make-trajectory-waypoints.py"
      --yaml "${CMAKE_CURRENT_SOURCE_DIR}/etc/RangerTrajectory.in.yaml" --check
  )
  set_tests_properties(RangerTrajectoryWaypointsUpToDate PROPERTIES TIMEOUT 60)

  # Smoke test through the real ticker path: MCGlobalController loads the
  # controller module, merges the configuration and plays the FSM. Five seconds
  # only covers the first state, which is enough to catch a configuration that
  # no longer loads.
  add_test(
    NAME RangerTrajectoryTicker
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-ranger-trajectory.yaml"
      --run-for 5 --no-sync
  )
  set_tests_properties(
    RangerTrajectoryTicker
    PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" TIMEOUT 120 ENVIRONMENT
               "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
  )

  # Same reason as the RollingContact fixtures: MCGlobalController merges
  # ~/.config/mc_rtc/mc_rtc.yaml before the -f file, so a developer profile
  # could otherwise override MainRobot and the enabled controllers here.
  mc_rtc_isolate_test_home(RangerTrajectoryControllerLifecycle RangerTrajectoryTicker)
endif()
