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
      RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH="${PROJECT_SOURCE_DIR}/src/mc_robots/ranger_mini_v3_description"
      RANGER_MINI_V3_ROBOT_MODULE_PATH="$<TARGET_FILE_DIR:ranger_mini_v3>"
      ROLLING_CONTACT_OBSERVER_MODULE_PATH="$<TARGET_FILE_DIR:EncoderObserver>"
  )
  target_link_libraries(
    testRollingContactControllerLifecycle
    PRIVATE rolling_contact_controller Boost::unit_test_framework
  )
  add_dependencies(
    testRollingContactControllerLifecycle ranger_mini_v3 EncoderObserver
    BodySensorObserver VelocityAidedTiltObserver
  )
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
  # SMK-13: long-horizon closed loop. Four 60 s rollouts (12000 cycles each,
  # both chassis, both terrains) are about as expensive as the whole rest of
  # this binary, so the Boost case carries a `disabled()` decorator and never
  # runs in the default RollingContactControllerRepeatedLifecycle invocation.
  # Boost re-enables a disabled unit when --run_test names it, which is how the
  # entry below reaches it. Opt-in, following the ORC-08 pattern:
  #   cmake -S . -B build -DROLLING_CONTACT_LONG_HORIZON_SMOKE=ON
  # then run ctest -L rolling-contact-slow.
  option(ROLLING_CONTACT_LONG_HORIZON_SMOKE
         "Register the SMK-13 long-horizon closed-loop smoke test with CTest" OFF
  )
  if(ROLLING_CONTACT_LONG_HORIZON_SMOKE)
    add_test(
      NAME RollingContactLongHorizonSMK13
      COMMAND
        testRollingContactControllerLifecycle
        --run_test=LongHorizonClosedLoopSMK13 --log_level=message
    )
    set_tests_properties(
      RollingContactLongHorizonSMK13
      PROPERTIES
        TIMEOUT 300
        LABELS "rolling-contact-slow"
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
    )
    mc_rtc_isolate_test_home(RollingContactLongHorizonSMK13)
  endif()

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
      "$<TARGET_FILE:ranger_mini_v3>"
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

  # The chassis tilt estimator end to end: synthesised IMU in
  # Ticker::simulate_sensors(), VelocityAidedTiltObserver, and the controller's
  # attitude monitor. Twelve seconds because the yaw command has to carry the
  # chassis a useful way around the cone about the tilted terrain normal before
  # --tilt-excursion-min is satisfied; see the configuration's header for why a
  # level chassis would make the bound vacuous.
  add_test(
    NAME RollingContactControllerFourSteeringTilt
    COMMAND
      $<TARGET_FILE:mc_rtc_ticker> -f
      "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-four-steering-tilt.yaml"
      --run-for 12 --no-sync
  )
  set_tests_properties(
    RollingContactControllerFourSteeringTilt
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      TIMEOUT 90
      ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
      FIXTURES_SETUP RollingContactFourSteeringTiltLog
  )
  add_test(
    NAME RollingContactControllerFourSteeringTiltLog
    COMMAND
      "${Python3_EXECUTABLE}" "${ROLLING_CONTACT_CHECKER}"
      /tmp/mc-rtc-rolling-four-steering-tilt-RollingContact-latest.bin --cycles 2400
      --scenario ackermann_left --backend Tasks --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
      # 0.5 degrees against a measured 0.028 peak: an order of margin, so this
      # fails on a broken frame or a dead sensor rather than on filter tuning.
      # The 10 degree excursion floor is measured at 20.0.
      --tilt-error-limit 0.5 --tilt-excursion-min 10.0 --tilt-settle 2.0
  )
  set_tests_properties(
    RollingContactControllerFourSteeringTiltLog
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      FIXTURES_REQUIRED RollingContactFourSteeringTiltLog
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

  # ORC-08: closed-loop cross-check against an independent simulator. The
  # simulator is the real mc_mujoco binary, not the CPU-headless runner of
  # run-mujoco-suite.py -- that runner injects MuJoCo's true contact forces into
  # the estimator, which is exactly the ground truth that once hid a wheel
  # detachment from it. Opt-in because it needs mc_mujoco installed, writes
  # hundreds of megabytes of log per run and takes about a minute; the default
  # suite must stay fast. Enable with
  #   cmake -S . -B build -DROLLING_CONTACT_MC_MUJOCO_CROSS_CHECK=ON
  # then run ctest -L rolling-contact-mujoco.
  option(ROLLING_CONTACT_MC_MUJOCO_CROSS_CHECK
         "Register the ORC-08 mc_mujoco closed-loop cross-check with CTest" OFF
  )
  if(ROLLING_CONTACT_MC_MUJOCO_CROSS_CHECK)
    find_program(MC_MUJOCO_EXECUTABLE mc_mujoco REQUIRED)
    add_test(
      NAME RollingContactMuJoCoCrossCheckORC08
      COMMAND
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/rolling-contact-report/scripts/cross-check-mc-mujoco.py"
        --mc-mujoco "${MC_MUJOCO_EXECUTABLE}" --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
        --config
        "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-four-steering.yaml"
        --timesteps 0.005 --window 5.0 --wall-timeout 8.0 --torque-control --output
        "${PROJECT_BINARY_DIR}/rolling-contact-orc08.json"
    )
    set_tests_properties(
      RollingContactMuJoCoCrossCheckORC08
      PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        TIMEOUT 600
        LABELS "rolling-contact-mujoco"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
    )
    # The same cross-check in mc_mujoco's own default actuation regime, which is
    # what the acceptance command `mc_mujoco -f ...` uses. Registering both is
    # the point: the two regimes disagree by a factor of twelve on the
    # controller/simulator discrepancy (0.0234 m under joint PD against
    # 0.2953 m under torque control), and running only one of them is how a
    # regime divergence goes unnoticed. The bounds here are the tight ones the
    # PD regime actually achieves.
    add_test(
      NAME RollingContactMuJoCoCrossCheckPDORC08
      COMMAND
        "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/rolling-contact-report/scripts/cross-check-mc-mujoco.py"
        --mc-mujoco "${MC_MUJOCO_EXECUTABLE}" --utils "$<TARGET_FILE_DIR:mc_bin_to_log>"
        --config
        "${PROJECT_SOURCE_DIR}/rolling-contact-report/config/mc_rtc-four-steering.yaml"
        --timesteps 0.005 --window 5.0 --wall-timeout 8.0 --position-discrepancy 0.06
        --minimum-tracked-fraction 0.9 --output
        "${PROJECT_BINARY_DIR}/rolling-contact-orc08-pd.json"
    )
    set_tests_properties(
      RollingContactMuJoCoCrossCheckPDORC08
      PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        TIMEOUT 600
        LABELS "rolling-contact-mujoco"
        SKIP_RETURN_CODE 77
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON"
    )
    mc_rtc_isolate_test_home(
      RollingContactMuJoCoCrossCheckORC08 RollingContactMuJoCoCrossCheckPDORC08
    )
  endif()

  # The ticker fixtures below construct a full MCGlobalController, which merges
  # ~/.config/mc_rtc/mc_rtc.yaml *before* the -f file. A developer profile can
  # therefore override MainRobot, the enabled controllers and every RollingContact
  # option these fixtures leave unset, and LoadUserConfiguration cannot be turned
  # off from the -f file because it is read before that file is merged. Pin an
  # empty home so no user configuration reaches them.
  mc_rtc_isolate_test_home(
    RollingContactControllerRepeatedLifecycle
    RollingContactCPULinkage
    RollingContactControllerDifferential
    RollingContactControllerDifferentialLog
    RollingContactControllerFourSteering
    RollingContactControllerFourSteeringLog
    RollingContactControllerTVMDifferential
    RollingContactControllerTVMDifferentialLog
    RollingContactControllerTVMFourSteering
    RollingContactControllerTVMFourSteeringLog
  )
endif()
