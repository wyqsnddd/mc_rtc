#include "mc_rolling_contact_controller.h"

#include <mc_observers/ObserverLoader.h>
#include <mc_rbdyn/RobotLoader.h>

#include <boost/test/unit_test.hpp>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

mc_rbdyn::RobotModulePtr robotModule(const std::string & name)
{
  static const bool configured = []()
  {
    mc_rbdyn::RobotLoader::clear();
    mc_rbdyn::RobotLoader::update_robot_module_path({ROLLING_CONTACT_ROBOT_MODULE_PATH});
    return true;
  }();
  (void)configured;
  return mc_rbdyn::RobotLoader::get_robot_module(name);
}

void loadObserverModules()
{
  static const bool configured = []()
  {
    mc_observers::ObserverLoader::clear();
    mc_observers::ObserverLoader::update_module_path({ROLLING_CONTACT_OBSERVER_MODULE_PATH});
    return true;
  }();
  (void)configured;
}

mc_rtc::Configuration controllerConfiguration(const std::string & scenario)
{
  mc_rtc::Configuration config;
  auto settings = config.add("RollingContact");
  settings.add("scenario", scenario);
  if(scenario == "infeasible_soft" || scenario == "incompatible" || scenario == "steering_rate"
     || scenario == "mode_cycle")
  {
    settings.add("longitudinal", "soft");
  }
  return config;
}

/** ObserverPipelines config used by the closed-loop tests below: Encoder runs
 * before BodySensor because BodySensorObserver::run() reads
 * realRobot().X_b1_b2(sensor.parentBody(), floatingBaseBody), which depends
 * on the joint state Encoder's update() just refreshed via
 * forwardKinematics() (mc_observers/BodySensorObserver.cpp:89-134,
 * mc_observers/EncoderObserver.cpp:169-170).
 *
 * Unlike the report configs under rolling-contact-report/config/ (which use
 * "position: encoderValues" / "velocity: encoderVelocities" because
 * mc_mujoco publishes a real encoder packet every tick), this test harness
 * constructs MCRollingContactController directly and never populates
 * robot().encoderValues()/encoderVelocities() - there is no simulator or
 * ticker doing that here. EncoderObserver::run() throws on an empty encoder
 * array in "encoderValues"/"encoderVelocities" mode (see
 * mc_observers/EncoderObserver.cpp:88-97), so use "control" mode instead: it
 * mirrors robot().mbc().q/alpha, which
 * MCRollingContactController::syncControlRobotFromSensors() has already
 * synchronized from the mocked sensors earlier in the same run(), so Encoder
 * is still genuinely exercised ahead of BodySensor in the pipeline.
 */
mc_rtc::Configuration observerPipelineConfig(bool withBodySensor = true)
{
  std::string yaml = "ObserverPipelines:\n"
                     "  - name: RollingContactPipeline\n"
                     "    gui: false\n"
                     "    observers:\n"
                     "      - type: Encoder\n"
                     "        update: true\n"
                     "        position: control\n"
                     "        velocity: control\n";
  if(withBodySensor)
  {
    yaml += "      - type: BodySensor\n"
           "        update: true\n"
           "        bodySensor: FloatingBase\n"
           "        method: sensor\n"
           "        updatePose: true\n"
           "        updateVel: true\n";
  }
  return mc_rtc::Configuration::fromYAMLData(yaml);
}

/** Build a closed-loop Ranger controller with the Encoder + BodySensor
 * observer pipeline attached, mirroring how MCGlobalController wires
 * ObserverPipelines for a real deployment: createObserverPipelines() runs
 * right after construction and reset() runs before resetObserverPipelines()
 * (see MCGlobalController::initController(), mc_global_controller.cpp:353-368
 * and :968). Pass withBodySensor=false only for the mutation test: dropping
 * BodySensor from the pipeline must make realRobot()'s pose/twist stop
 * tracking the sensor.
 */
std::unique_ptr<mc_control::MCRollingContactController> makeClosedLoopController(const std::string & scenario,
                                                                                 bool withBodySensor = true)
{
  loadObserverModules();
  auto config = controllerConfiguration(scenario);
  config("RollingContact").add("closedLoopFeedback", true);
  config.load(observerPipelineConfig(withBodySensor));
  auto controller = std::make_unique<mc_control::MCRollingContactController>(
      robotModule("RollingContactRangerMiniV3"), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller->createObserverPipelines(config);
  controller->reset({controller->robot().mbc().q});
  return controller;
}

/** Run the observer pipeline for one cycle, then the controller, mirroring
 * MCGlobalController::run(): runObserverPipelines() happens before
 * controller_->run() every tick (mc_global_controller.cpp:809,813). A caller
 * that skips this and calls controller.run() directly restores a stale free
 * joint from realRobot() every cycle (see the comment on the restore block in
 * MCRollingContactController::run()).
 */
bool stepClosedLoop(mc_control::MCRollingContactController & controller)
{
  controller.runObserverPipelines();
  return controller.run();
}

void exercise(mc_control::MCController::Backend backend, const std::string & robot, int repetitions)
{
  const auto config = controllerConfiguration("hold");
  for(int replay = 0; replay < repetitions; ++replay)
  {
    mc_control::MCRollingContactController controller(robotModule(robot), 0.005, config, backend);
    controller.reset({controller.robot().mbc().q});
    BOOST_REQUIRE(controller.run());
  }
}

/** Ranger Mini V3 on the "hold" script: no scripted twist, so the controller
 * follows whatever setCommandedTwist() last received. Open loop on purpose:
 * closedLoopFeedback would overwrite the floating base from the (never
 * updated) FloatingBase sensor every cycle and the chassis could not move.
 */
std::unique_ptr<mc_control::MCRollingContactController> makeRangerController()
{
  const auto config = controllerConfiguration("hold");
  auto controller = std::make_unique<mc_control::MCRollingContactController>(
      robotModule("RollingContactRangerMiniV3"), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller->reset({controller->robot().mbc().q});
  return controller;
}

/** Same as makeRangerController(), but lets the caller add or override keys
 * under the "RollingContact" settings block before construction (e.g.
 * twistWeight, baseOrientationWeight) - mirrors the config().add(...) pattern
 * makeClosedLoopController() uses for closedLoopFeedback.
 */
std::unique_ptr<mc_control::MCRollingContactController> makeRangerController(
    const std::function<void(mc_rtc::Configuration &)> & configure)
{
  auto config = controllerConfiguration("hold");
  auto settings = config("RollingContact");
  configure(settings);
  auto controller = std::make_unique<mc_control::MCRollingContactController>(
      robotModule("RollingContactRangerMiniV3"), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller->reset({controller->robot().mbc().q});
  return controller;
}

/** Chassis displacement between two floating-base poses, expressed in the
 * START body frame.
 *
 * sva::PTransformd::rotation() is the world-to-body map E_0_b, so
 * E_0_b * d_world are the world displacement's coordinates in the body frame.
 */
/** Temporarily point standard input at a pseudo-terminal.
 *
 * The "keyboard" scenario refuses to start unless stdin is interactive, and a
 * test binary run by CTest has a pipe there. Rather than adding a test-only
 * escape hatch to the controller, hand it a real tty for the duration of the
 * test so the production guard runs exactly as it does for an operator.
 */
struct ScopedPseudoTerminalStdin
{
  ScopedPseudoTerminalStdin()
  {
    master_ = ::posix_openpt(O_RDWR | O_NOCTTY);
    if(master_ < 0 || ::grantpt(master_) != 0 || ::unlockpt(master_) != 0) { return; }
    const char * name = ::ptsname(master_);
    if(!name) { return; }
    slave_ = ::open(name, O_RDWR | O_NOCTTY);
    if(slave_ < 0) { return; }
    savedStdin_ = ::dup(STDIN_FILENO);
    if(savedStdin_ < 0 || ::dup2(slave_, STDIN_FILENO) < 0) { return; }
    active_ = true;
  }

  ~ScopedPseudoTerminalStdin()
  {
    if(savedStdin_ >= 0)
    {
      ::dup2(savedStdin_, STDIN_FILENO);
      ::close(savedStdin_);
    }
    if(slave_ >= 0) { ::close(slave_); }
    if(master_ >= 0) { ::close(master_); }
  }

  ScopedPseudoTerminalStdin(const ScopedPseudoTerminalStdin &) = delete;
  ScopedPseudoTerminalStdin & operator=(const ScopedPseudoTerminalStdin &) = delete;

  bool active() const noexcept { return active_ && ::isatty(STDIN_FILENO) == 1; }

private:
  int master_ = -1;
  int slave_ = -1;
  int savedStdin_ = -1;
  bool active_ = false;
};

Eigen::Vector3d chassisMotion(const sva::PTransformd & start, const sva::PTransformd & end)
{
  return start.rotation() * (end.translation() - start.translation());
}

/** Signed yaw travelled between two floating-base poses.
 *
 * With E = world-to-body, the body-to-world attitude is E^T, so the rotation
 * the chassis performed in the world is R = E_end^T * E_start. The mirrored
 * product E_end * E_start^T is its inverse and reports the opposite sign.
 */
double chassisYaw(const sva::PTransformd & start, const sva::PTransformd & end)
{
  const Eigen::Matrix3d relative = end.rotation().transpose() * start.rotation();
  return std::atan2(relative(1, 0), relative(0, 0));
}

/** Drive a changing FloatingBase sensor trajectory through a closed-loop
 * Ranger controller for a few cycles and check that realRobot() - not a copy
 * of the control robot - tracks it.
 *
 * This is the discriminator between "state observation" and the old
 * hand-rolled mirror it replaced: a mirror would make realRobot() equal
 * whatever the QP predicted for the control robot's free joint that cycle,
 * which has no reason to equal the sensor. Here the sensor value changes
 * every cycle and is asserted against directly, both at the public posW()/
 * velW() API (world frame) and at the internal mbc().alpha[0] storage (body
 * frame, RBDyn's Free-joint convention), so a pipeline that silently stopped
 * updating realRobot() - e.g. because BodySensor was dropped, see the
 * withBodySensor parameter - is caught either way.
 *
 * Pass withBodySensor=false only to mutation-test this check: BOOST_CHECK
 * (not BOOST_REQUIRE) is used for the tracking assertions so a dropped
 * BodySensor observer is reported as failures rather than aborting the test
 * body, letting every cycle's mismatch show up in the log.
 */
void driveObserverPipelineAndCheckRealRobotTracksSensor(bool withBodySensor)
{
  auto controller = makeClosedLoopController("hold", withBodySensor);
  auto & sensor = controller->robot().data()->bodySensors[
      controller->robot().data()->bodySensorsIndex.at("FloatingBase")];

  for(int cycle = 0; cycle < 10; ++cycle)
  {
    const double t = static_cast<double>(cycle);
    // A changing pose/twist per cycle: if realRobot() were a stale copy (or
    // simply never updated) it would not track this motion.
    const Eigen::Vector3d position{0.05 * t, -0.02 * t, 0.16};
    const Eigen::Quaterniond orientation{Eigen::AngleAxisd(0.05 * t, Eigen::Vector3d::UnitZ())};
    const Eigen::Vector3d linearVelocity{0.05, -0.02, 0.0};
    const Eigen::Vector3d angularVelocity{0.0, 0.0, 0.05};
    sensor.position(position);
    sensor.orientation(orientation);
    sensor.linearVelocity(linearVelocity);
    sensor.angularVelocity(angularVelocity);
    if(cycle == 0) { controller->resetObserverPipelines(); }
    BOOST_REQUIRE(stepClosedLoop(*controller));

    const auto & real = controller->realRobot();
    BOOST_CHECK_SMALL((real.posW().translation() - position).norm(), 1e-9);
    BOOST_CHECK_SMALL((real.posW().rotation() - orientation.toRotationMatrix()).norm(), 1e-9);
    BOOST_CHECK_SMALL((real.velW().linear() - linearVelocity).norm(), 1e-9);
    BOOST_CHECK_SMALL((real.velW().angular() - angularVelocity).norm(), 1e-9);

    // The sensor orientation is the world-to-body rotation E_0_b (same
    // convention as posW().rotation(), see the comments on the other
    // closed-loop tests above), and BodySensorObserver's X_s_fb collapses to
    // Identity for this robot (FloatingBase's parent body is "chassis" and
    // X_b_s is Identity), so rotating the world-frame sensor velocity by
    // that same E_0_b predicts exactly what realRobot.velW(velW_) must have
    // written into mbc().alpha[0].
    const Eigen::Matrix3d worldToBody = orientation.toRotationMatrix();
    const Eigen::Vector3d expectedBodyAngular = worldToBody * angularVelocity;
    const Eigen::Vector3d expectedBodyLinear = worldToBody * linearVelocity;
    const auto & alpha0 = real.mbc().alpha[0];
    BOOST_REQUIRE_EQUAL(alpha0.size(), 6u);
    const Eigen::Vector3d actualBodyAngular{alpha0[0], alpha0[1], alpha0[2]};
    const Eigen::Vector3d actualBodyLinear{alpha0[3], alpha0[4], alpha0[5]};
    BOOST_CHECK_SMALL((actualBodyAngular - expectedBodyAngular).norm(), 1e-9);
    BOOST_CHECK_SMALL((actualBodyLinear - expectedBodyLinear).norm(), 1e-9);
    BOOST_TEST_MESSAGE("[observer-tracking] cycle=" << cycle << " |dpos|="
                                                     << (real.posW().translation() - position).norm()
                                                     << " |dvel|=" << (real.velW().linear() - linearVelocity).norm());
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(RollingContactControllerRepeatedLifecycle)
{
  exercise(mc_control::MCController::Backend::Tasks, "RollingContactDifferential", 4);
  exercise(mc_control::MCController::Backend::Tasks, "RollingContactFourSteering", 2);
  exercise(mc_control::MCController::Backend::Tasks, "RollingContactRangerMiniV3", 4);
  exercise(mc_control::MCController::Backend::TVM, "RollingContactDifferential", 4);
  exercise(mc_control::MCController::Backend::TVM, "RollingContactFourSteering", 2);
  exercise(mc_control::MCController::Backend::TVM, "RollingContactRangerMiniV3", 4);
}

BOOST_AUTO_TEST_CASE(RollingContactControllerScenarioMatrix)
{
  const std::array<std::string, 11> differential = {"hold",          "reverse",     "forward",
                                                     "turn_left",     "turn_right",  "circle_left",
                                                     "circle_right",  "sinusoid",    "unequal_radii",
                                                     "infeasible_soft", "mode_cycle"};
  const std::array<std::string, 10> steering = {"hold",           "forward",       "reverse",
                                                 "crab",           "ackermann_left", "ackermann_right",
                                                 "pure_yaw",       "steering_rate", "incompatible",
                                                 "mode_cycle"};
  for(const auto backend : {mc_control::MCController::Backend::Tasks, mc_control::MCController::Backend::TVM})
  {
    for(const auto & scenario : differential)
    {
      const auto config = controllerConfiguration(scenario);
      mc_control::MCRollingContactController controller(
          robotModule("RollingContactDifferential"), 0.005, config, backend);
      controller.reset({controller.robot().mbc().q});
      for(int cycle = 0; cycle < 20; ++cycle) { BOOST_REQUIRE(controller.run()); }
      BOOST_CHECK_EQUAL(controller.datastore().call<std::string>("RollingContact::GetBackend"),
                        backend == mc_control::MCController::Backend::Tasks ? "Tasks" : "TVM");
    }
    for(const auto & scenario : steering)
    {
      const auto config = controllerConfiguration(scenario);
      mc_control::MCRollingContactController controller(
          robotModule("RollingContactRangerMiniV3"), 0.005, config, backend);
      controller.reset({controller.robot().mbc().q});
      for(int cycle = 0; cycle < 20; ++cycle) { BOOST_REQUIRE(controller.run()); }
      BOOST_CHECK_EQUAL(controller.datastore().call<std::string>("RollingContact::GetBackend"),
                        backend == mc_control::MCController::Backend::Tasks ? "Tasks" : "TVM");
    }
  }
}

BOOST_AUTO_TEST_CASE(FourSteeringTracksCommandedTwistSigns)
{
  // Each commanded twist must move the chassis in the commanded direction.
  // Before the QP rate rows the keyboard path needed an explicit sign mirror
  // and two empirical scale factors to get anywhere near this.
  struct Case
  {
    const char * name;
    Eigen::Vector3d twist; // vx, vy, omega in the chassis frame
  };
  const std::vector<Case> cases = {{"forward", {0.3, 0.0, 0.0}},   {"backward", {-0.3, 0.0, 0.0}},
                                   {"crab-left", {0.0, 0.3, 0.0}}, {"crab-right", {0.0, -0.3, 0.0}},
                                   {"yaw-positive", {0.0, 0.0, 0.5}}, {"yaw-negative", {0.0, 0.0, -0.5}}};

  for(const auto & test : cases)
  {
    auto controller = makeRangerController();
    controller->setCommandedTwist(test.twist);

    const sva::PTransformd start = controller->robot().posW();
    for(int cycle = 0; cycle < 400; ++cycle) { BOOST_REQUIRE_MESSAGE(controller->run(), test.name); }
    const sva::PTransformd end = controller->robot().posW();

    const Eigen::Vector3d motion = chassisMotion(start, end);
    const double yaw = chassisYaw(start, end);
    BOOST_TEST_MESSAGE("[twist] " << test.name << " dx=" << motion.x() << " dy=" << motion.y()
                                  << " dyaw=" << yaw);

    // 400 cycles at dt = 5 ms is 2 s, so a perfectly tracked 0.3 m/s command
    // travels 0.6 m and a 0.5 rad/s command turns 1.0 rad. Observed here:
    // 0.593 m forward/backward, 0.535 m of crab (plus a 0.061 m forward
    // excursion while the hinges swing to +/-pi/2) and 0.919 rad of yaw. The
    // bounds are set to a third of the ideal value, which is roughly half of
    // every observation and still far above the 0.10 m / 0.37 rad the QP
    // produced before the rate rows carried their weight.
    if(std::abs(test.twist.x()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(motion.x() * test.twist.x() > 0.0, test.name << " x sign: " << motion.x());
      BOOST_CHECK_MESSAGE(std::abs(motion.x()) > 0.2, test.name << " x magnitude: " << motion.x());
    }
    if(std::abs(test.twist.y()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(motion.y() * test.twist.y() > 0.0, test.name << " y sign: " << motion.y());
      BOOST_CHECK_MESSAGE(std::abs(motion.y()) > 0.2, test.name << " y magnitude: " << motion.y());
    }
    if(std::abs(test.twist.z()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(yaw * test.twist.z() > 0.0, test.name << " yaw sign: " << yaw);
      BOOST_CHECK_MESSAGE(std::abs(yaw) > 0.3, test.name << " yaw magnitude: " << yaw);
    }
  }
}

BOOST_AUTO_TEST_CASE(FourSteeringCommandChangeKeepsResidualsBounded)
{
  // Switching between commands must not need a transition grace period: the QP
  // arbitrates steering and drive together, so no wheel receives drive torque
  // against a stale contact direction.
  auto controller = makeRangerController();
  const std::vector<Eigen::Vector3d> sequence = {
      {0.3, 0.0, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.3, 0.0}, {0.3, 0.0, 0.4}};

  // Observed over the whole sequence: the worst lateral slip is 7.9e-2 m/s, on
  // the very first cycles of the forward -> pure-yaw step, where the chassis is
  // still translating at 0.3 m/s while the hinges swing to +/-0.94 rad. It is a
  // decaying transient: every phase ends at 3.7e-18, 4.3e-4, 8.7e-5 and 4.7e-2
  // m/s respectively. The bounds below sit ~2x above those observations.
  //
  // Mutation-tested, so these are not decorative. Withholding the steering-rate
  // reference from the QP makes run() itself fail; tripling the steering
  // convergence time takes the worst slip to 1.63 m/s (213 failed assertions);
  // and handing the QP the pre-projection rolling rate - a real bug this test
  // caught during review - takes it to 0.29 m/s.
  double worst = 0.0;
  for(const auto & twist : sequence)
  {
    controller->setCommandedTwist(twist);
    for(int cycle = 0; cycle < 200; ++cycle)
    {
      BOOST_REQUIRE(controller->run());
      worst = std::max(worst, controller->maxLateralResidual());
      BOOST_CHECK_LT(controller->maxLateralResidual(), 0.15);
    }
    // The transient must decay inside the phase, not merely stay bounded.
    BOOST_CHECK_LT(controller->maxLateralResidual(), 0.1);
    BOOST_TEST_MESSAGE("[phase-end] " << twist.transpose() << " res=" << controller->maxLateralResidual());
  }
  BOOST_TEST_MESSAGE("[residual] worst lateral slip over the sequence: " << worst);
}

BOOST_AUTO_TEST_CASE(TwistWeightAxesAreUnitNormalizedNotInterchangeable)
{
  // eq:planar-twist-weight weights the planar twist error with a diagonal
  // matrix W_xi = diag(w_vx, w_vy, w_omega), not a scalar, because vx/vy
  // (m/s) and omega (rad/s) are not comparable: a weight of "1" in
  // (m/s)^-2 units and a weight of "1" in (rad/s)^-2 units penalize
  // physically different things that only look alike because both numbers
  // happen to be 1. This test proves twistWeight's three entries reach the
  // QP as independent, genuine per-axis multipliers: not silently ignored,
  // not merged into one effective scalar, and not interchangeable with each
  // other.
  //
  // The primary checks below read back basePositionTask_'s and
  // baseOrientationTask_'s dimWeight() directly through the
  // RollingContact::GetBase{Position,Orientation}DimWeight datastore calls -
  // exactly the vector SetPointTaskCommon::computeQC
  // (Tasks/src/QPTasks.cpp:67-70) multiplies elementwise into that task's QP
  // Hessian - instead of inferring the weight's effect from a closed-loop
  // trajectory.
  //
  // A physical-trajectory design was tried first and abandoned. By default,
  // baseOrientationTask_/basePositionTask_ have no real authority to lose in
  // the first place: fourSteering_ (Ranger) unconditionally sets
  // trackRotatingRates = true with rollingRateWeight/steeringRateWeight
  // defaulting to 1000/dt^2 = 4e7, and the no-slip "longitudinal" row
  // defaults to Hard (a genuine equality constraint, not a weighted task -
  // see RollingContactConstraint.h). Both pin every wheel's spin to the
  // kinematic feed-forward regardless of task weight, at a coefficient
  // respectively ~80000x and effectively infinite times
  // baseOrientationWeight/basePositionWeight's ~500 default, so starving
  // twistWeight.z() moved the tracked yaw by only ~1e-6 rad regardless of
  // the starving factor. Softening the rolling constraint
  // (longitudinal: soft, rate weights zeroed) restored real leverage for
  // omega, but the matching x-axis scenario turned out to be confounded by a
  // genuine vx/yaw coupling in the QP - boosting twistWeight.x() moved yaw by
  // 0.3-0.7 rad at every magnitude tried, so the trajectory-divergence
  // metric could not isolate the x-axis effect the way it could for omega.
  // Direct dimWeight() introspection sidesteps all of this: it is exact,
  // deterministic, and (unlike a closed-loop metric) exercises exactly the
  // code this task changed - the config-to-dimWeight() wiring - without
  // also depending on the rolling-contact QP's unrelated internal balance.
  //
  // Non-vacuousness (would fail if twistWeight's entries were forced equal,
  // ignored, swapped between tasks, or collapsed into one scalar): w_vx=2,
  // w_vy=3, w_omega=5 are three distinct values, so any such bug changes at
  // least one component of the two expected vectors below.
  {
    auto controller = makeRangerController(
        [](mc_rtc::Configuration & settings) { settings.add("twistWeight", Eigen::Vector3d{2.0, 3.0, 5.0}); });
    const Eigen::Vector3d positionDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionDimWeight");
    const Eigen::Vector3d orientationDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBaseOrientationDimWeight");
    BOOST_TEST_MESSAGE("[twistWeight] position dimWeight=" << positionDimWeight.transpose());
    BOOST_TEST_MESSAGE("[twistWeight] orientation dimWeight=" << orientationDimWeight.transpose());
    // basePositionTask_: [w_vx, w_vy, *] - z (chassis height) is not part of
    // the planar twist and always keeps weight 1.
    BOOST_CHECK_SMALL((positionDimWeight - Eigen::Vector3d{2.0, 3.0, 1.0}).norm(), 1e-9);
    // baseOrientationTask_: [*, *, w_omega] - roll/pitch are not part of the
    // planar twist and always keep weight 1.
    BOOST_CHECK_SMALL((orientationDimWeight - Eigen::Vector3d{1.0, 1.0, 5.0}).norm(), 1e-9);
  }

  // Default (no twistWeight key at all) must reproduce today's behaviour
  // exactly: both dimWeight vectors stay at Ones(), same as before
  // twistWeight_ existed.
  {
    auto controller = makeRangerController([](mc_rtc::Configuration &) {});
    const Eigen::Vector3d positionDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionDimWeight");
    const Eigen::Vector3d orientationDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBaseOrientationDimWeight");
    BOOST_CHECK_SMALL((positionDimWeight - Eigen::Vector3d::Ones()).norm(), 1e-9);
    BOOST_CHECK_SMALL((orientationDimWeight - Eigen::Vector3d::Ones()).norm(), 1e-9);
  }

  // The deprecated bare-scalar form broadcasts to all three axes, then
  // follows the same per-task split as the vector form.
  {
    auto controller = makeRangerController([](mc_rtc::Configuration & settings) { settings.add("twistWeight", 4.0); });
    const Eigen::Vector3d positionDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionDimWeight");
    const Eigen::Vector3d orientationDimWeight =
        controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBaseOrientationDimWeight");
    BOOST_CHECK_SMALL((positionDimWeight - Eigen::Vector3d{4.0, 4.0, 1.0}).norm(), 1e-9);
    BOOST_CHECK_SMALL((orientationDimWeight - Eigen::Vector3d{1.0, 1.0, 4.0}).norm(), 1e-9);
  }

  // Closed-loop confirmation, on the one axis (omega) that produces an
  // isolated, unconfounded signal (see the block comment above): starving
  // twistWeight.z() alone must move the closed-loop yaw well away from
  // baseline while leaving x comparatively close to it. This is a
  // supplementary sanity check that the dimWeight() wiring has genuine
  // closed-loop consequence, on top of the exact checks above.
  {
    // M_PI is not guaranteed by the C++ standard.
    constexpr double pi = 3.14159265358979323846;
    const double k = (pi / 180.0) * (pi / 180.0);

    auto run = [](const Eigen::Vector3d & twistWeight)
    {
      auto controller = makeRangerController(
          [&](mc_rtc::Configuration & settings)
          {
            settings.add("twistWeight", twistWeight);
            // Move the no-slip row into the weighted objective (still at its
            // default rollingWeight_ = 1000) instead of a hard equality
            // constraint, and silence the separately-dominant rate rows so
            // the base tasks' weight is contending against something of
            // comparable magnitude. See the block comment above.
            settings.add("longitudinal", "soft");
            settings.add("rollingRateWeight", 0.0);
            settings.add("steeringRateWeight", 0.0);
          });
      const sva::PTransformd start = controller->robot().posW();
      for(int cycle = 0; cycle < 400; ++cycle)
      {
        // A *steady* commanded twist gives the base tasks nothing to
        // arbitrate: the wheel-kinematic reference and the base tasks target
        // the same twist, so in steady state they agree and the weight is
        // moot. They genuinely disagree only while the steering hinges are
        // still slewing towards steeringWheelReference()'s answer
        // (first-order convergence, time constant
        // steeringTimeConstant_ = 0.15 s = 30 cycles at dt = 5 ms). Flipping
        // the commanded yaw every 15 cycles - half the convergence time -
        // keeps the hinges perpetually mid-slew, so this disagreement
        // persists for the whole run instead of decaying after 0.15 s.
        const double yaw = ((cycle / 15) % 2 == 0) ? 0.8 : -0.8;
        controller->setCommandedTwist({0.3, 0.0, yaw});
        BOOST_REQUIRE(controller->run());
      }
      const sva::PTransformd end = controller->robot().posW();
      return std::make_pair(chassisMotion(start, end), chassisYaw(start, end));
    };

    const auto baseline = run({1.0, 1.0, 1.0});
    const auto omegaStarved = run({1.0, 1.0, k});
    BOOST_TEST_MESSAGE("[twistWeight] baseline     dx=" << baseline.first.x() << " dy=" << baseline.first.y()
                                                         << " dyaw=" << baseline.second);
    BOOST_TEST_MESSAGE("[twistWeight] omegaStarved dx=" << omegaStarved.first.x() << " dy="
                                                         << omegaStarved.first.y() << " dyaw=" << omegaStarved.second);
    // Starving omega alone must move yaw measurably away from baseline...
    //
    // The bound was 0.2 rad while updateModes() still let the QP's own contact
    // multiplier demote a wheel. Under this adversarial command that happened
    // constantly: instrumenting the two runs showed 2854 of 3200 wheel-samples
    // (89%) estimated sliding or detached, on flat ground, with no disturbance
    // and with the first detachment reported at a healthy 183.9 N and 147 N of
    // friction margin. The chassis was free to swing because its contacts kept
    // being switched off, and both runs' net yaw was an order of magnitude
    // larger for it (baseline -0.157 rad, omegaStarved +0.144 rad).
    //
    // With that spurious detachment removed all four wheels stay rolling, the
    // chassis is properly held by four rolling contacts, and the whole yaw
    // excursion is correspondingly smaller: baseline -0.00748 rad,
    // omegaStarved -0.03225 rad, difference 0.02476 rad. The weight still has
    // a genuine, isolated closed-loop consequence - it is 3.3x the residual
    // x-axis crosstalk below, and the run is deterministic to the bit, so the
    // 0.01 rad bound keeps a 2.5x margin. It is deliberately not tuned back up
    // by re-enabling the mode chatter it used to ride on.
    BOOST_CHECK_GT(std::abs(baseline.second - omegaStarved.second), 0.01);
    // ...while leaving x comparatively close to baseline: the effect stays
    // on the axis whose weight actually changed.
    BOOST_CHECK_LT(std::abs(baseline.first.x() - omegaStarved.first.x()), 0.1);
  }
}

BOOST_AUTO_TEST_CASE(FourSteeringMirrorsForwardPlusNegativeYaw)
{
  // Regression test for the driveAcceleration 20->100 rewrite regression:
  // FourSteeringTracksCommandedTwistSigns only exercises pure axes and
  // FourSteeringCommandChangeKeepsResidualsBounded only exercises one mixed
  // command, forward-plus-*positive*-yaw (0.3, 0, +0.4). Neither covers
  // forward-plus-*negative*-yaw, which is exactly the gap that let the
  // regression through: at driveAcceleration=100 the +yaw case tracks fine
  // while the mirrored -yaw case detaches two wheels 2 cycles in and the
  // chassis barely moves. A mirror-symmetric robot commanded with mirrored
  // twists must produce mirrored trajectories; this pins that invariant
  // directly rather than only re-checking the known-good sign.
  auto run = [](double yaw)
  {
    auto controller = makeRangerController();
    controller->setCommandedTwist({0.3, 0.0, yaw});
    const sva::PTransformd start = controller->robot().posW();
    const std::array<const char *, 4> wheelNames = {"front_left", "front_right", "rear_left", "rear_right"};
    for(int cycle = 0; cycle < 400; ++cycle)
    {
      BOOST_REQUIRE(controller->run());
      for(const auto * wheel : wheelNames)
      {
        const auto mode = controller->datastore().call<std::string, const std::string &>(
            "RollingContact::GetEstimatedMode", std::string(wheel));
        BOOST_CHECK_MESSAGE(mode != "detached",
                            "yaw=" << yaw << " wheel " << wheel << " detached at cycle " << cycle);
      }
    }
    const sva::PTransformd end = controller->robot().posW();
    return std::make_pair(chassisMotion(start, end), chassisYaw(start, end));
  };

  const auto positive = run(0.4);
  const auto negative = run(-0.4);

  BOOST_TEST_MESSAGE("[mirror] +yaw dx=" << positive.first.x() << " dy=" << positive.first.y()
                                         << " dyaw=" << positive.second);
  BOOST_TEST_MESSAGE("[mirror] -yaw dx=" << negative.first.x() << " dy=" << negative.first.y()
                                         << " dyaw=" << negative.second);

  // Neither direction may stall: both must make substantial forward and
  // rotational progress over the 2 s window (400 cycles at 5 ms).
  BOOST_CHECK_GT(positive.first.x(), 0.2);
  BOOST_CHECK_GT(negative.first.x(), 0.2);
  BOOST_CHECK_GT(positive.second, 0.3);
  BOOST_CHECK_LT(negative.second, -0.3);

  // Mirror symmetry: forward progress must match regardless of yaw sign, and
  // the lateral drift / yaw must be mirror images of one another about y=0.
  BOOST_CHECK_CLOSE(positive.first.x(), negative.first.x(), 25.0);
  BOOST_CHECK_CLOSE(positive.first.y(), -negative.first.y(), 25.0);
  BOOST_CHECK_CLOSE(positive.second, -negative.second, 25.0);
}

BOOST_AUTO_TEST_CASE(RollingContactControllerSynchronizesFloatingBase)
{
  auto controller = makeClosedLoopController("hold");

  // BodySensor orientation follows the mc_rbdyn convention (inertial to
  // sensor). syncControlRobotFromSensors() converts this measured packet into
  // the free joint MBC of the control robot; the Encoder/BodySensor observer
  // pipeline is what actually produces realRobot()'s estimate from it (see
  // makeClosedLoopController / stepClosedLoop above).
  const Eigen::Vector3d measuredPosition{0.11, -0.035, 0.16};
  const Eigen::Quaterniond measuredWorldToBody{Eigen::AngleAxisd(0.23, Eigen::Vector3d::UnitZ())};
  const Eigen::Vector3d measuredLinearVelocity{0.18, -0.07, 0.0};
  const Eigen::Vector3d measuredAngularVelocity{0.0, 0.0, 0.21};
  auto & sensor = controller->robot().data()->bodySensors[
      controller->robot().data()->bodySensorsIndex.at("FloatingBase")];
  sensor.position(measuredPosition);
  sensor.orientation(measuredWorldToBody);
  sensor.linearVelocity(measuredLinearVelocity);
  sensor.angularVelocity(measuredAngularVelocity);
  controller->resetObserverPipelines();

  BOOST_REQUIRE(stepClosedLoop(*controller));
  const auto & control = controller->robot();
  const auto & real = controller->realRobot();
  const auto & chassis = control.frame("chassis").position();
  const auto & floatingBase = control.bodySensor("FloatingBase");
  const sva::PTransformd measuredFloatingBase(floatingBase.orientation(), floatingBase.position());
  BOOST_CHECK_EQUAL(floatingBase.parentBody(), "chassis");
  BOOST_CHECK_SMALL((floatingBase.X_b_s().matrix() - sva::PTransformd::Identity().matrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((control.posW().matrix() - chassis.matrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((measuredFloatingBase.matrix() - chassis.matrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((control.posW().translation() - measuredPosition).norm(), 1e-12);
  BOOST_CHECK_SMALL((control.posW().rotation() - measuredWorldToBody.toRotationMatrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((control.velW().linear() - measuredLinearVelocity).norm(), 1e-12);
  BOOST_CHECK_SMALL((control.velW().angular() - measuredAngularVelocity).norm(), 1e-12);
  BOOST_CHECK_SMALL((real.posW().translation() - measuredPosition).norm(), 1e-12);
  BOOST_CHECK_SMALL((real.posW().rotation() - measuredWorldToBody.toRotationMatrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((real.velW().linear() - measuredLinearVelocity).norm(), 1e-12);
  BOOST_CHECK_SMALL((real.velW().angular() - measuredAngularVelocity).norm(), 1e-12);
}

BOOST_AUTO_TEST_CASE(KeyboardClosedLoopYawTargetMirrorsTheMeasuredWorldHeading)
{
  // The closed-loop keyboard path is the one place where baseYawTarget_ is not
  // the world heading: it copies measuredYaw, read off posW().rotation().col(0).
  // posW().rotation() is the inertial-to-body map E_0_b, so a chassis yawed by
  // psi in the world stores -psi there, and run() has to negate it back before
  // building the translation heading. That negation had no coverage at all, so
  // pin both halves of it here.
  ScopedPseudoTerminalStdin tty;
  BOOST_REQUIRE_MESSAGE(tty.active(), "could not allocate a pseudo-terminal for the keyboard scenario");
  auto controller = makeClosedLoopController("keyboard");

  // Sensor orientation is the inertial-to-body rotation, so this is a chassis
  // whose +X axis really points at -0.4 rad in the world.
  constexpr double sensorYaw = 0.4;
  auto & sensor = controller->robot().data()->bodySensors[
      controller->robot().data()->bodySensorsIndex.at("FloatingBase")];
  sensor.orientation(Eigen::Quaterniond{Eigen::AngleAxisd(sensorYaw, Eigen::Vector3d::UnitZ())});
  sensor.position(controller->robot().posW().translation());
  controller->resetObserverPipelines();
  BOOST_REQUIRE(stepClosedLoop(*controller));

  // The true world heading of the chassis' +X axis: E_0_b^T * e_x.
  const Eigen::Vector3d worldForward = controller->robot().posW().rotation().transpose().col(0);
  const double worldYaw = std::atan2(worldForward.y(), worldForward.x());
  BOOST_CHECK_CLOSE(worldYaw, -sensorYaw, 1e-6);

  // baseYawTarget_ holds the mirrored value, which is exactly why run() negates
  // it. If this ever equals worldYaw instead, the negation must go with it.
  const double baseYawTarget = controller->datastore().call<double>("RollingContact::GetBaseYawTarget");
  BOOST_CHECK_CLOSE(baseYawTarget, sensorYaw, 1e-6);
  BOOST_CHECK_CLOSE(-baseYawTarget, worldYaw, 1e-6);

  // Now pin the consequence rather than only the premise: a forward command
  // must walk the integrated position target along the chassis' true world
  // heading. Without the negation the target walks along +sensorYaw instead,
  // i.e. mirrored about the world X axis. Drive it through the GUI input,
  // which the keyboard poll adds to the key state every cycle; setCommandedTwist
  // would be overwritten by that same poll.
  BOOST_REQUIRE(controller->gui()->handleRequest({"Rolling Contact", "Command"}, "Forward velocity",
                                                 mc_rtc::Configuration::fromData("0.3")));
  const Eigen::Vector3d targetBefore =
      controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionTarget");
  for(int cycle = 0; cycle < 20; ++cycle) { BOOST_REQUIRE(stepClosedLoop(*controller)); }
  const Eigen::Vector3d targetAfter =
      controller->datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionTarget");

  const Eigen::Vector3d walked = targetAfter - targetBefore;
  BOOST_REQUIRE_GT(walked.head<2>().norm(), 1e-6);
  const double walkedYaw = std::atan2(walked.y(), walked.x());
  BOOST_TEST_MESSAGE("[keyboard-heading] sensorYaw=" << sensorYaw << " worldYaw=" << worldYaw
                                                     << " walkedYaw=" << walkedYaw);
  BOOST_CHECK_SMALL(std::remainder(walkedYaw - worldYaw, 2.0 * 3.14159265358979323846), 1e-6);
}

BOOST_AUTO_TEST_CASE(RollingContactObserverPipelineDrivesRealRobotEstimate)
{
  driveObserverPipelineAndCheckRealRobotTracksSensor(/* withBodySensor = */ true);
}

BOOST_AUTO_TEST_CASE(RollingContactControllerRejectsInvalidConfiguration)
{
  auto unknownScenario = controllerConfiguration("unknown");
  BOOST_CHECK_THROW(mc_control::MCRollingContactController(
                        robotModule("RollingContactDifferential"), 0.005, unknownScenario),
                    std::invalid_argument);

  auto invalidTerrain = controllerConfiguration("hold");
  invalidTerrain("RollingContact").add(
      "terrainNormal", Eigen::Vector3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0});
  BOOST_CHECK_THROW(mc_control::MCRollingContactController(
                        robotModule("RollingContactDifferential"), 0.005, invalidTerrain),
                    std::invalid_argument);

  auto invalidRecovery = controllerConfiguration("hold");
  invalidRecovery("RollingContact").add("rollingWeight", 1000.0);
  invalidRecovery("RollingContact").add("recoveryRollingWeight", 10.0);
  BOOST_CHECK_THROW(mc_control::MCRollingContactController(
                        robotModule("RollingContactDifferential"), 0.005, invalidRecovery),
                    std::invalid_argument);

  auto invalidPeriod = controllerConfiguration("hold");
  invalidPeriod("RollingContact").add("commandPeriod", -1.0);
  BOOST_CHECK_THROW(mc_control::MCRollingContactController(
                        robotModule("RollingContactDifferential"), 0.005, invalidPeriod),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(HeadingTargetCannotWindUpToTheRotationErrorSingularity)
{
  // Regression for a hard QP failure measured against real mc_mujoco: the
  // accumulated heading target integrates the commanded yaw rate with no
  // feedback, so whenever the chassis cannot deliver that rate the orientation
  // task's error grows without bound. It does not just get large.
  // mc_tasks::OrientationTask's error is sva::rotationError(), whose near-pi
  // branch (taken once |error| > pi - 1.105e-2 rad) evaluates
  // s = (2*diag(E) + (1-trace)) / (3-trace) and then s.cwiseSqrt(). For a
  // near-planar rotation the first two entries of s are zero only in exact
  // arithmetic; on the Ranger the instrumented values at the failing cycle were
  //   1+trace = 8.7057077548191586e-05  (threshold 1.220703125e-04)
  //   s       = [-1.1102471883440498e-16, -1.1102471883440498e-16, 1.0000000000000002]
  // so cwiseSqrt() returned NaN, the NaN reached the QP's linear term and the
  // solve failed - with all four contacts at 183.9 N, 119 N of friction margin
  // and the chassis level to 1e-5 rad. Against mc_mujoco this killed
  // scenario: pure_yaw at 0.35 rad/s at t = 9.72 s (PD actuation) and t = 12.87
  // s (--torque-control), and mc_mujoco then spun forever because its
  // simulate() loop discards stepSimulation()'s return value.
  //
  // This reproduces the same windup deterministically and without a simulator:
  // closedLoopFeedback pins the chassis to a FloatingBase sensor that never
  // rotates, so the measured heading stays at zero while a 0.35 rad/s yaw is
  // commanded. Unbounded, the error would pass pi - 1.105e-2 rad at
  // t = 8.95 s = cycle 1791; this runs to cycle 2400 (12 s).
  auto controller = makeClosedLoopController("hold");
  auto & sensor = controller->robot().data()->bodySensors[
      controller->robot().data()->bodySensorsIndex.at("FloatingBase")];
  sensor.position(Eigen::Vector3d{0.0, 0.0, 0.16});
  sensor.orientation(Eigen::Quaterniond::Identity());
  sensor.linearVelocity(Eigen::Vector3d::Zero());
  sensor.angularVelocity(Eigen::Vector3d::Zero());
  controller->resetObserverPipelines();

  // M_PI is not guaranteed by the C++ standard.
  constexpr double pi = 3.14159265358979323846;
  constexpr double bound = 0.5 * pi;
  double worstYawTarget = 0.0;
  double worstOrientationEval = 0.0;
  for(int cycle = 0; cycle < 2400; ++cycle)
  {
    controller->setCommandedTwist({0.0, 0.0, 0.35});
    BOOST_REQUIRE_MESSAGE(stepClosedLoop(*controller), "QP failed at cycle " << cycle);
    const double yawTarget = controller->datastore().call<double>("RollingContact::GetBaseYawTarget");
    const double orientationEval =
        controller->datastore().call<double>("RollingContact::GetOrientationEvalNorm");
    BOOST_REQUIRE_MESSAGE(std::isfinite(yawTarget), "non-finite yaw target at cycle " << cycle);
    BOOST_REQUIRE_MESSAGE(std::isfinite(orientationEval), "non-finite orientation error at cycle " << cycle);
    worstYawTarget = std::max(worstYawTarget, std::abs(yawTarget));
    worstOrientationEval = std::max(worstOrientationEval, orientationEval);
  }
  BOOST_TEST_MESSAGE("[heading-windup] worst |yaw target|=" << worstYawTarget
                                                            << " worst |orientation eval|=" << worstOrientationEval);
  // The measured heading never leaves zero, so the target itself is the error.
  BOOST_CHECK_LE(worstYawTarget, bound + 1e-9);
  // ...and so is the task's rotation error, which is the quantity that goes
  // singular. Well clear of the pi - 1.105e-2 rad branch that produces the NaN.
  BOOST_CHECK_LE(worstOrientationEval, bound + 1e-6);
}

BOOST_AUTO_TEST_CASE(MaxYawTargetErrorMustStayClearOfTheRotationErrorSingularity)
{
  constexpr double pi = 3.14159265358979323846;
  for(const double invalid : {0.0, -0.1, 0.95 * pi, pi, 2.0 * pi})
  {
    auto config = controllerConfiguration("hold");
    config("RollingContact").add("maxYawTargetError", invalid);
    BOOST_CHECK_THROW(
        mc_control::MCRollingContactController(robotModule("RollingContactRangerMiniV3"), 0.005, config),
        std::invalid_argument);
  }
  auto valid = controllerConfiguration("hold");
  valid("RollingContact").add("maxYawTargetError", 0.5);
  BOOST_CHECK_NO_THROW(
      mc_control::MCRollingContactController(robotModule("RollingContactRangerMiniV3"), 0.005, valid));
}

BOOST_AUTO_TEST_CASE(QpContactMultiplierAloneNeverDetachesAWheel)
{
  // Regression for the front_right detachment measured against real mc_mujoco
  // on four-steering crab and ackermann_left. Four coplanar wheel contacts
  // leave the normal-force distribution with a one-dimensional null space - the
  // diagonal mode (+1, -1, -1, +1), which produces no net force and no net
  // moment - and nothing in this QP's objective penalises it, so the solver may
  // return zero on one wheel while every wheel is loaded. updateModes() used to
  // read that zero as a detachment; desiredMode() makes that transition
  // immediate (no dwell), and it is self-confirming because a detached wheel's
  // multipliers are identically zero, so front_right stayed detached for 4990
  // of 4995 cycles with the chassis level to 1e-5 rad throughout.
  //
  // The stimulus below is the same adversarial command the twistWeight
  // closed-loop check uses. Instrumented at the previous behaviour it estimated
  // 1395 detached and 1459 sliding wheel-samples out of 3200 - on flat ground,
  // with no disturbance, and with the first "sliding" verdict reported at a
  // perfectly healthy 183.9 N and 147.0 N of friction margin.
  //
  // No external measurement is fed here, which is exactly the mc_mujoco default
  // and the headless-ticker case. An adapter that calls
  // RollingContact::SetMeasuredContact keeps full detection; that path is
  // checked by the CPU MuJoCo suite's disturbance cases.
  auto controller = makeRangerController(
      [](mc_rtc::Configuration & settings)
      {
        settings.add("longitudinal", "soft");
        settings.add("rollingRateWeight", 0.0);
        settings.add("steeringRateWeight", 0.0);
      });
  const std::array<std::string, 4> wheels = {"front_left", "front_right", "rear_left", "rear_right"};
  int detachedSamples = 0;
  for(int cycle = 0; cycle < 400; ++cycle)
  {
    controller->setCommandedTwist({0.3, 0.0, ((cycle / 15) % 2 == 0) ? 0.8 : -0.8});
    BOOST_REQUIRE(controller->run());
    for(const auto & wheel : wheels)
    {
      if(controller->datastore().call<std::string, const std::string &>("RollingContact::GetEstimatedMode", wheel)
         == "detached")
      {
        ++detachedSamples;
      }
    }
  }
  BOOST_TEST_MESSAGE("[qp-multiplier-detach] detached wheel-samples=" << detachedSamples);
  BOOST_CHECK_EQUAL(detachedSamples, 0);
}
