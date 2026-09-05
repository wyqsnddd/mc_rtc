#
# Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
#

add_controller(
  "rolling_contact_controller"
  "${CMAKE_CURRENT_SOURCE_DIR}/mc_rolling_contact_controller.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mc_rolling_contact_controller.h"
)

# Keyboard teleoperation is optional so the CPU rolling-contact controller
# remains buildable on systems that do not install RoboticsUtils. The
# interactive Ranger example checks this capability at runtime.
find_package(RoboticsUtils QUIET CONFIG)
if(TARGET RoboticsUtils::Keyboard)
  # RoboticsUtils 0.1.0 builds KeyboardCapture together with its GEOS-backed
  # utility source but omits libgeos_c from the exported target. Supply that
  # transitive dependency until the external package exports it itself.
  find_library(ROBOTICSUTILS_GEOS_C_LIBRARY NAMES geos_c)
endif()
if(TARGET RoboticsUtils::Keyboard AND ROBOTICSUTILS_GEOS_C_LIBRARY)
  target_link_libraries(
    rolling_contact_controller
    PRIVATE RoboticsUtils::Keyboard "${ROBOTICSUTILS_GEOS_C_LIBRARY}"
  )
  target_compile_definitions(
    rolling_contact_controller PRIVATE MC_ROLLING_CONTACT_HAS_KEYBOARD=1
  )
endif()

install(
  DIRECTORY etc/
  DESTINATION ${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc/
)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
  add_executable(
    testRollingContactControllerLifecycle test_controller_lifecycle.cpp
  )
  target_compile_definitions(
    testRollingContactControllerLifecycle
    PRIVATE
      ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/rolling_contact_description"
      ROLLING_CONTACT_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:rolling_contact>"
  )
  target_link_libraries(
    testRollingContactControllerLifecycle
    PRIVATE rolling_contact_controller Boost::unit_test_framework
  )
  add_dependencies(testRollingContactControllerLifecycle rolling_contact)
  target_compile_definitions(
    testRollingContactControllerLifecycle
    PRIVATE BOOST_TEST_DYN_LINK BOOST_TEST_MAIN
  )
  add_test(
    NAME RollingContactControllerRepeatedLifecycle
    COMMAND testRollingContactControllerLifecycle
  )
  set_tests_properties(
    RollingContactControllerRepeatedLifecycle
    PROPERTIES
      TIMEOUT 60
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
  )
  set(ROLLING_CONTACT_TEST_ROOT "${PROJECT_SOURCE_DIR}")
  set(ROLLING_CONTACT_CHECKER
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/scripts/check-controller-log.py"
  )
  add_test(
    NAME RollingContactCPULinkage
    COMMAND
      "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/scripts/audit-cpu-linkage.py"
      --library-dir "$<TARGET_FILE_DIR:mc_rbdyn>"
      --library-dir "$<TARGET_FILE_DIR:rolling_contact_controller>"
      --library-dir "${PROJECT_BINARY_DIR}/deps/tasks-system-install/lib"
      --library-dir "/usr/local/lib"
      --output "${PROJECT_BINARY_DIR}/rolling-contact-cpu-linkage.json"
      "$<TARGET_FILE:mc_rbdyn>"
      "$<TARGET_FILE:mc_solver>"
      "$<TARGET_FILE:rolling_contact>"
      "$<TARGET_FILE:rolling_contact_controller>"
      "$<TARGET_FILE:testRollingContactControllerLifecycle>"
  )
  set_tests_properties(
    RollingContactCPULinkage
    PROPERTIES
      TIMEOUT 30
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1"
  )
  add_test(
    NAME RollingContactControllerDifferential
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-differential.yaml"
      --run-for 5 --no-sync
  )
  set_tests_properties(
    RollingContactControllerDifferential
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      TIMEOUT 60
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
      FIXTURES_SETUP RollingContactDifferentialLog
  )
  add_test(
    NAME RollingContactControllerDifferentialLog
    COMMAND
      "${Python3_EXECUTABLE}" "${ROLLING_CONTACT_CHECKER}"
      /tmp/mc-rtc-rolling-differential-RollingContact-latest.bin --cycles 1000
      --scenario forward --backend Tasks --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
  )
  set_tests_properties(
    RollingContactControllerDifferentialLog
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      FIXTURES_REQUIRED RollingContactDifferentialLog
      TIMEOUT 30
  )

  add_test(
    NAME RollingContactControllerFourSteering
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-four-steering.yaml"
      --run-for 5 --no-sync
  )
  set_tests_properties(
    RollingContactControllerFourSteering
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      TIMEOUT 60
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
      FIXTURES_SETUP RollingContactFourSteeringLog
  )
  add_test(
    NAME RollingContactControllerFourSteeringLog
    COMMAND
      "${Python3_EXECUTABLE}" "${ROLLING_CONTACT_CHECKER}"
      /tmp/mc-rtc-rolling-four-steering-RollingContact-latest.bin --cycles 1000
      --scenario crab --backend Tasks --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
  )
  set_tests_properties(
    RollingContactControllerFourSteeringLog
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      FIXTURES_REQUIRED RollingContactFourSteeringLog
      TIMEOUT 30
  )

  add_test(
    NAME RollingContactControllerTVMDifferential
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-differential-tvm.yaml"
      --run-for 5 --no-sync
  )
  set_tests_properties(
    RollingContactControllerTVMDifferential
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      TIMEOUT 60
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
      FIXTURES_SETUP RollingContactTVMDifferentialLog
  )
  add_test(
    NAME RollingContactControllerTVMDifferentialLog
    COMMAND
      "${Python3_EXECUTABLE}" "${ROLLING_CONTACT_CHECKER}"
      /tmp/mc-rtc-rolling-differential-tvm-RollingContact_TVM-latest.bin --cycles 1000
      --scenario forward --backend TVM --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
  )
  set_tests_properties(
    RollingContactControllerTVMDifferentialLog
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      FIXTURES_REQUIRED RollingContactTVMDifferentialLog
      TIMEOUT 30
  )

  add_test(
    NAME RollingContactControllerTVMFourSteering
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-four-steering-tvm.yaml"
      --run-for 5 --no-sync
  )
  set_tests_properties(
    RollingContactControllerTVMFourSteering
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      TIMEOUT 60
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
      FIXTURES_SETUP RollingContactTVMFourSteeringLog
  )
  add_test(
    NAME RollingContactControllerTVMFourSteeringLog
    COMMAND
      "${Python3_EXECUTABLE}" "${ROLLING_CONTACT_CHECKER}"
      /tmp/mc-rtc-rolling-four-steering-tvm-RollingContact_TVM-latest.bin --cycles 1000
      --scenario crab --backend TVM --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
  )
  set_tests_properties(
    RollingContactControllerTVMFourSteeringLog
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      FIXTURES_REQUIRED RollingContactTVMFourSteeringLog
      TIMEOUT 30
  )
endif()
