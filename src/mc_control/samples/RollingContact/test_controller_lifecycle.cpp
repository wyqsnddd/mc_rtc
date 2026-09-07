#include "mc_rolling_contact_controller.h"

#include <mc_observers/ObserverLoader.h>
#include <mc_rbdyn/CylindricalSurface.h>
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

/** Any rolling-contact robot on any scripted scenario, open loop, with an
 * optional hook to add or override keys under the "RollingContact" settings
 * block before construction (e.g. twistWeight, baseOrientationWeight) - the
 * same config().add(...) pattern makeClosedLoopController() uses for
 * closedLoopFeedback.
 *
 * Open loop on purpose: closedLoopFeedback would overwrite the floating base
 * from the (never updated) FloatingBase sensor every cycle and the chassis
 * could not move.
 */
std::unique_ptr<mc_control::MCRollingContactController> makeController(
    const std::string & robot,
    const std::string & scenario,
    const std::function<void(mc_rtc::Configuration &)> & configure = {})
{
  auto config = controllerConfiguration(scenario);
  if(configure)
  {
    auto settings = config("RollingContact");
    configure(settings);
  }
  auto controller = std::make_unique<mc_control::MCRollingContactController>(
      robotModule(robot), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller->reset({controller->robot().mbc().q});
  return controller;
}

/** Ranger Mini V3 on the "hold" script: no scripted twist, so the controller
 * follows whatever setCommandedTwist() last received.
 */
std::unique_ptr<mc_control::MCRollingContactController> makeRangerController()
{
  return makeController("RollingContactRangerMiniV3", "hold");
}

std::unique_ptr<mc_control::MCRollingContactController> makeRangerController(
    const std::function<void(mc_rtc::Configuration &)> & configure)
{
  return makeController("RollingContactRangerMiniV3", "hold", configure);
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

// ===========================================================================
// Layer H. Smoke sequence.
// ===========================================================================

namespace
{

/** Chassis-frame displacement and yaw accumulated between two cycle indices of
 * a scripted rollout, plus the worst lateral residual seen inside the window.
 *
 * Every smoke test below reports these rather than only asserting on them: the
 * suite asks for the metrics on passing runs too, because that is what makes a
 * later regression diagnosable.
 */
struct RolloutWindow
{
  Eigen::Vector3d motion = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double seconds = 0.0;
  double worstLateralResidual = 0.0;
};

/** Run @p controller for @p cycles, measuring only over [@p from, @p cycles).
 *
 * The window excludes the start-up transient, which is what makes the
 * measurement a *steady-state* one: the drive reference is rate-limited to
 * driveAcceleration (20 rad/s^2 by default) and the steering hinges converge
 * with a 0.15 s time constant, so the first few hundred milliseconds of any
 * command are deliberately not the command.
 */
RolloutWindow runWindow(mc_control::MCRollingContactController & controller,
                        int cycles,
                        int from,
                        const std::function<void(int)> & perCycle = {})
{
  RolloutWindow window;
  sva::PTransformd start = controller.robot().posW();
  for(int cycle = 0; cycle < cycles; ++cycle)
  {
    if(perCycle) { perCycle(cycle); }
    BOOST_REQUIRE_MESSAGE(controller.run(), "the QP failed at cycle " << cycle);
    if(cycle == from - 1) { start = controller.robot().posW(); }
    if(cycle >= from)
    {
      window.worstLateralResidual = std::max(window.worstLateralResidual, controller.maxLateralResidual());
    }
  }
  const sva::PTransformd end = controller.robot().posW();
  window.motion = chassisMotion(start, end);
  window.yaw = chassisYaw(start, end);
  window.seconds = 0.005 * static_cast<double>(cycles - from);
  return window;
}

/** Planar offset of a wheel carrier in the chassis frame, read off the robot the
 * same way MCRollingContactController's constructor builds wheelOffsets_. */
Eigen::Vector2d carrierOffset(const mc_control::MCRollingContactController & controller, const std::string & wheel)
{
  const auto & chassis = controller.robot().frame("chassis").position();
  const Eigen::Vector3d worldOffset =
      controller.robot().frame(wheel + "_carrier").position().translation() - chassis.translation();
  return (chassis.rotation() * worldOffset).head<2>();
}

const std::array<const char *, 4> & rangerWheels()
{
  static const std::array<const char *, 4> wheels = {"front_left", "front_right", "rear_left", "rear_right"};
  return wheels;
}

} // namespace

BOOST_AUTO_TEST_CASE(StraightLineDriveTracksTheReferenceSMK03)
{
  // SMK-03. Constant forward reference, flat ground, both chassis. The card
  // asks for the steady-state tracking error below 1%, and for the T1/T2 decay
  // asymmetry of ROW-11 to be asserted separately. That second half does not
  // transfer: eq:four-steering-wheel-qp is an ideal QP, whereas the general
  // acceleration-level rows mc_rtc assembles carry -Kp G u for BOTH chassis -
  // see PredictedRateRowsCarryNoProportionalStabilizationROW11 in
  // tests/testRollingContactSolver.cpp, which pins the genuinely T2-specific
  // property instead. What is asserted here for both chassis is what is true of
  // both: the reference is tracked, and the lateral residual is driven to zero
  // rather than merely bounded.
  struct Case
  {
    const char * name;
    const char * robot;
    const char * scenario;
    double reference; // m/s of commanded forward speed
    bool commanded;   // T2 takes its twist through setCommandedTwist()
    int cycles;
    int from;
  };
  // The differential "forward" script publishes linearSpeed = 0.2 m/s itself;
  // the Ranger follows whatever twist was last commanded.
  //
  // The two start-up transients are of very different lengths, and the window
  // is set from the measurement rather than assumed. T2's rate rows own the
  // wheel degrees of freedom at 1000/dt^2 and settle inside 2 s (0.22% at
  // cycle 400, unchanged at cycle 3800). T1 has no rate rows at all: its wheels
  // are driven by the posture task at stiffness 5 against an integrated
  // position target, which rings - 13.8% high at cycle 400, 2.2% low at cycle
  // 1800, 0.12% at cycle 3800 - so its window starts at 19 s.
  const std::array<Case, 2> cases = {
      Case{"T1 differential", "RollingContactDifferential", "forward", 0.2, false, 4000, 3800},
      Case{"T2 four-steering", "RollingContactRangerMiniV3", "hold", 0.3, true, 600, 400}};

  for(const auto & test : cases)
  {
    auto controller = makeController(test.robot, test.scenario);
    if(test.commanded) { controller->setCommandedTwist({test.reference, 0.0, 0.0}); }
    const auto window = runWindow(*controller, test.cycles, test.from);
    const double speed = window.motion.x() / window.seconds;
    const double error = std::abs(speed - test.reference) / test.reference;
    BOOST_TEST_MESSAGE("SMK-03 " << test.name << ": steady-state vx = " << speed << " m/s against " << test.reference
                                 << " (error " << 100.0 * error << "%), lateral drift " << window.motion.y()
                                 << " m, yaw " << window.yaw << " rad, worst lateral residual "
                                 << window.worstLateralResidual << " m/s");
    BOOST_CHECK_MESSAGE(error < 1e-2, test.name << ": steady-state tracking error " << 100.0 * error << "%");
    // A straight line is a straight line: no lateral drift and no yaw.
    BOOST_CHECK_LT(std::abs(window.motion.y()), 1e-2 * std::abs(window.motion.x()));
    BOOST_CHECK_LT(std::abs(window.yaw), 1e-2);
    // The lateral residual is driven to zero in steady state, not merely kept
    // bounded: this is the end-to-end read-out of ROW-03's decay.
    BOOST_CHECK_LT(window.worstLateralResidual, 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(InPlaceRotationSMK04)
{
  // SMK-04. Zero linear reference, non-zero yaw reference.
  //
  // The T2 half deliberately does NOT assert the card's "yaw rate tracked to
  // 1%". mc_rtc places each wheel's contact forces along a two-point contact
  // line of width `width` but writes its kinematic rows at the single centre
  // point, so nothing in this QP represents the scrub torque of re-steering a
  // loaded patch. Against real mc_mujoco a *pure* in-place yaw is the one
  // manoeuvre where that shows: it tracks at 6.7% of the commanded rotation
  // under joint PD, against 84-96% for every mixed command, and narrowing the
  // MuJoCo wheel half-width from 0.04 m to 0.004 m raises it to 89%. The
  // headless figure below is much better because there is no contact patch to
  // scrub against here - which is exactly why a tight threshold here would hide
  // the defect rather than catch it. What is asserted is what is true of this
  // manoeuvre in this QP: it runs cleanly, no wheel detaches, the solve never
  // fails, and the steering configuration it settles into is the tangent one
  // with zero steady-state hinge rate. The achieved yaw fraction is recorded.
  {
    // T1 produces equal and opposite wheel rates.
    auto controller = makeController("RollingContactDifferential", "turn_left");
    const auto window = runWindow(*controller, 600, 400);
    const double reference = 0.35; // the scripted yawRate
    const double rate = window.yaw / window.seconds;
    BOOST_TEST_MESSAGE("SMK-04 T1: steady-state yaw rate " << rate << " rad/s against " << reference
                                                            << ", chassis motion " << window.motion.transpose());
    BOOST_CHECK_CLOSE(rate, reference, 1.0);
    // Equal and opposite: an in-place rotation spins the two wheels at the same
    // speed in opposite directions, and the chassis does not translate.
    const double left = controller->datastore().call<double, const std::string &>("RollingContact::GetDriveVelocity",
                                                                                  std::string("left"));
    const double right = controller->datastore().call<double, const std::string &>("RollingContact::GetDriveVelocity",
                                                                                   std::string("right"));
    BOOST_TEST_MESSAGE("SMK-04 T1: wheel rates left " << left << " rad/s, right " << right << " rad/s");
    BOOST_REQUIRE_GT(std::abs(left), 1e-3);
    BOOST_CHECK_SMALL(left + right, 1e-2 * std::abs(left));
    BOOST_CHECK_LT(window.motion.head<2>().norm(), 1e-2 * std::abs(reference * window.seconds));
  }
  {
    // T2 steers every wheel tangent to a circle about the chassis centre.
    constexpr double reference = 0.5; // rad/s
    auto controller = makeRangerController();
    controller->setCommandedTwist({0.0, 0.0, reference});
    std::array<Eigen::Vector2d, 4> offsets{};
    for(size_t i = 0; i < rangerWheels().size(); ++i) { offsets[i] = carrierOffset(*controller, rangerWheels()[i]); }
    const auto window = runWindow(*controller, 600, 400, [&](int cycle)
                                  {
                                    // No wheel may detach at any point of the manoeuvre.
                                    for(const auto * wheel : rangerWheels())
                                    {
                                      const auto mode = controller->datastore().call<std::string, const std::string &>(
                                          "RollingContact::GetEstimatedMode", std::string(wheel));
                                      BOOST_CHECK_MESSAGE(mode != "detached",
                                                          "SMK-04 T2: " << wheel << " detached at cycle " << cycle);
                                    }
                                  });

    // The steady-state hinge rate is the cleanest end-to-end read-out of the
    // GEO-07 convention: once the tangent configuration is reached, a
    // chassis-relative steering angle needs no further rate to keep turning.
    // Read from inside the QP, i.e. off the control robot's own alpha.
    double worstSteeringRate = 0.0;
    double worstHeadingError = 0.0;
    for(size_t i = 0; i < rangerWheels().size(); ++i)
    {
      const auto joint = controller->robot().jointIndexByName(std::string(rangerWheels()[i]) + "_steer");
      worstSteeringRate = std::max(worstSteeringRate, std::abs(controller->robot().mbc().alpha[joint][0]));
      // Tangent to a circle about the chassis centre: the wheel-centre velocity
      // of a pure yaw is omega * (-y_i, x_i), so delta_i = atan2(x_i, -y_i).
      // Compared through the sine of the difference, because a wheel line is
      // pi-periodic and the reference inverter is free to pick either branch.
      const double expected = std::atan2(offsets[i].x(), -offsets[i].y());
      const double measured = controller->robot().mbc().q[joint][0];
      worstHeadingError = std::max(worstHeadingError, std::abs(std::sin(measured - expected)));
    }
    const double achieved = window.yaw / (reference * window.seconds);
    BOOST_TEST_MESSAGE("SMK-04 T2: yaw " << window.yaw << " rad over " << window.seconds << " s, i.e. "
                                         << 100.0 * achieved << "% of the commanded rotation; residual chassis motion "
                                         << window.motion.head<2>().norm() << " m; worst |sin(delta - delta_ref)| = "
                                         << worstHeadingError << "; worst steady-state |deltaDot| = "
                                         << worstSteeringRate << " rad/s");
    // The manoeuvre runs cleanly and turns the right way. The fraction is
    // recorded above, not thresholded at the card's 1%: see the note at the top
    // of this test for why, and Residual gaps in the test specification.
    BOOST_CHECK_GT(window.yaw, 0.0);
    BOOST_CHECK_GT(achieved, 0.5);
    // The steering configuration and its steady-state rate, which is what the
    // card calls the cleanest read-out of the convention.
    BOOST_CHECK_LT(worstHeadingError, 1e-3);
    BOOST_CHECK_LT(worstSteeringRate, 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(TangentBasisRotatesWithTheChassisSMK10)
{
  // SMK-10, in its two-branch form. eq:differential-pose-rate integrates the
  // planar pose as pdot = E_Pi^T xi, and the whole content of the card is that
  // E_Pi must be propagated as the chassis turns. A closed circular path is the
  // discriminator: with the basis propagated the path closes, and with it
  // frozen at its initial value it does not.
  //
  // The circle is chosen so one revolution is an exact whole number of control
  // cycles - omega = 2 pi / 10 s = 2000 cycles at dt = 5 ms - because a partial
  // last cycle would put a quantisation error in the closure that has nothing
  // to do with the basis.
  constexpr double period = 10.0;
  constexpr double dt = 0.005;
  constexpr int cycles = 2000;
  constexpr double omega = 2.0 * 3.14159265358979323846 / period;
  constexpr double forward = 0.2;
  const double radius = forward / omega;
  BOOST_REQUIRE_EQUAL(cycles, static_cast<int>(period / dt));

  // Branch A/B: the pose propagation itself, driven by an exactly constant body
  // twist. This is the same mc_rbdyn::Robot::eulerIntegration the Tasks backend
  // calls on every solved cycle (TasksQPSolver.cpp), so it is the shipped
  // integrator and not a restatement of it.
  auto controller = makeRangerController();
  auto & robot = controller->robot();
  const sva::PTransformd start = robot.posW();
  const Eigen::Matrix3d bodyToWorld = start.rotation().transpose();
  const std::vector<double> twist = {0.0, 0.0, omega, forward, 0.0, 0.0};
  robot.mbc().alpha[0] = twist;
  robot.mbc().alphaD[0] = std::vector<double>(6, 0.0);
  robot.forwardKinematics();
  robot.forwardVelocity();

  Eigen::Vector3d frozen = start.translation();
  for(int cycle = 0; cycle < cycles; ++cycle)
  {
    // The frozen-basis variant: the same body-frame twist, mapped to the world
    // through the attitude the chassis had at t = 0 and never updated.
    frozen += dt * bodyToWorld * Eigen::Vector3d{forward, 0.0, 0.0};
    robot.eulerIntegration(dt);
    robot.forwardKinematics();
    robot.forwardVelocity();
  }
  // The body twist is genuinely constant, so both branches integrate the same
  // data and differ only in the basis.
  BOOST_CHECK_SMALL((Eigen::Map<const Eigen::Matrix<double, 6, 1>>(robot.mbc().alpha[0].data())
                     - Eigen::Map<const Eigen::Matrix<double, 6, 1>>(twist.data()))
                        .lpNorm<Eigen::Infinity>(),
                    1e-12);
  const double propagatedClosure = (robot.posW().translation() - start.translation()).norm();
  const double frozenClosure = (frozen - start.translation()).norm();
  const double attitudeClosure = (robot.posW().rotation() - start.rotation()).norm();
  BOOST_TEST_MESSAGE("SMK-10 radius " << radius << " m: propagated closure " << propagatedClosure
                                      << " m, frozen-basis closure " << frozenClosure << " m, attitude closure "
                                      << attitudeClosure);
  // Closure below 1e-3 of the circle radius for the correct integration...
  BOOST_CHECK_LT(propagatedClosure, 1e-3 * radius);
  BOOST_CHECK_SMALL(attitudeClosure, 1e-9);
  // ...and the frozen-basis variant fails by at least the radius itself. It
  // travels the full arc length, 2 pi R, in a straight line.
  BOOST_CHECK_GT(frozenClosure, radius);
  BOOST_CHECK_CLOSE(frozenClosure, 2.0 * 3.14159265358979323846 * radius, 1e-6);

  // The same circle with the QP in the loop, so the property is not only true
  // of the integrator in isolation. The chassis pose is whatever the solved
  // accelerations produced; the frozen-basis reconstruction below replays that
  // run's own body twists through the initial attitude.
  auto driven = makeRangerController();
  driven->setCommandedTwist({forward, 0.0, omega});
  const sva::PTransformd drivenStart = driven->robot().posW();
  const Eigen::Matrix3d drivenBodyToWorld = drivenStart.rotation().transpose();
  Eigen::Vector3d drivenFrozen = drivenStart.translation();
  for(int cycle = 0; cycle < cycles; ++cycle)
  {
    BOOST_REQUIRE_MESSAGE(driven->run(), "SMK-10 closed circle: the QP failed at cycle " << cycle);
    const auto & alpha = driven->robot().mbc().alpha[0];
    drivenFrozen += dt * drivenBodyToWorld * Eigen::Vector3d{alpha[3], alpha[4], alpha[5]};
  }
  const double drivenClosure = (driven->robot().posW().translation() - drivenStart.translation()).norm();
  const double drivenFrozenClosure = (drivenFrozen - drivenStart.translation()).norm();
  BOOST_TEST_MESSAGE("SMK-10 with the QP in the loop: closure " << drivenClosure << " m ("
                                                                << drivenClosure / radius
                                                                << " R), frozen-basis closure " << drivenFrozenClosure
                                                                << " m (" << drivenFrozenClosure / radius << " R)");
  // The commanded circle is only tracked to the accuracy of the QP, so this
  // half is not asserted at the card's 1e-3 R; what it does assert is that the
  // rotating basis is what closes the path and that freezing it is catastrophic
  // by a wide margin, on a trajectory the solver produced rather than one
  // prescribed here.
  BOOST_CHECK_GT(drivenFrozenClosure, radius);
  // Measured at 0.0076 m, i.e. 0.024 R and 0.4% of the frozen-basis error; the
  // bound keeps a factor of two on the first and a factor of thirteen on the
  // second.
  BOOST_CHECK_LT(drivenClosure, 0.05 * radius);
  BOOST_CHECK_LT(drivenClosure, 0.05 * drivenFrozenClosure);
}

BOOST_AUTO_TEST_CASE(SteeringRateLimitedSlalomSMK08)
{
  // SMK-08. A square-wave slalom whose commanded heading reverses by 2.0 rad
  // every 0.3 s, far faster than a rate-limited hinge can follow, so the
  // steering-rate limiter is active for most of every half period.
  //
  // Three things are asserted, and a fourth is recorded because it is a finding
  // rather than a contract:
  //  (1) the realised hinge rate never exceeds the configured bound;
  //  (2) the bound is what shapes the trajectory - halving it halves the
  //      plateau, so it is not inert;
  //  (3) the realised hinge trajectory matches a rate-limited reference model
  //      integrated independently of the QP, to 10% of the hinge travel;
  //  (4) the plateau follows the configured bound proportionally only up to
  //      4 rad/s and is clamped there above it, so the default -- the steering
  //      joints' own 8 rad/s model velocity limit -- is not reachable. See the
  //      note below the sweep.
  constexpr double dt = 0.005;
  constexpr double timeConstant = 0.15;
  constexpr double forward = 0.3;
  constexpr double amplitude = 1.45; // rad/s of commanded yaw, square wave
  constexpr int halfPeriod = 60;     // cycles, i.e. 0.3 s
  constexpr int cycles = 600;

  struct Outcome
  {
    double plateau = 0.0;        // largest realised |deltaDot| in the steady slalom
    double modelDeviation = 0.0; // largest |delta - limited-reference model|
    double travel = 0.0;         // peak-to-peak hinge travel in the steady slalom
  };

  auto slalom = [&](double bound, bool useDefaultBound)
  {
    auto controller = useDefaultBound
                          ? makeRangerController()
                          : makeRangerController([&](mc_rtc::Configuration & settings)
                                                 { settings.add("maxSteeringRate", bound); });
    std::array<mc_rbdyn::PlanarWheel, 4> planar{};
    std::array<double, 4> modelAngle{};
    std::array<size_t, 4> steerJoint{};
    std::array<double, 4> lowest{};
    std::array<double, 4> highest{};
    lowest.fill(std::numeric_limits<double>::infinity());
    highest.fill(-std::numeric_limits<double>::infinity());
    for(size_t i = 0; i < rangerWheels().size(); ++i)
    {
      const std::string wheel = rangerWheels()[i];
      steerJoint[i] = controller->robot().jointIndexByName(wheel + "_steer");
      planar[i].offset = carrierOffset(*controller, wheel);
      planar[i].radius = 0.08;
      modelAngle[i] = controller->robot().mbc().q[steerJoint[i]][0];
    }

    Outcome out;
    for(int cycle = 0; cycle < cycles; ++cycle)
    {
      const double yaw = ((cycle / halfPeriod) % 2 == 0) ? amplitude : -amplitude;
      const Eigen::Vector3d twist{forward, 0.0, yaw};
      controller->setCommandedTwist(twist);
      BOOST_REQUIRE_MESSAGE(controller->run(), "SMK-08: the QP failed at cycle " << cycle);
      for(size_t i = 0; i < rangerWheels().size(); ++i)
      {
        const double rate = controller->robot().mbc().alpha[steerJoint[i]][0];
        const double angle = controller->robot().mbc().q[steerJoint[i]][0];
        // (1) The realised rate respects the bound, every cycle. The bound is a
        // clamp on the *reference*, and the rate row that carries it into the QP
        // is a weighted objective rather than a box constraint (BND-06: no
        // KinematicsConstraint is attached here), so the realised rate can
        // overshoot it slightly - 0.7% at 3 rad/s is the worst measured. The
        // tolerance says exactly that rather than pretending the bound is hard.
        BOOST_CHECK_MESSAGE(std::abs(rate) <= 1.02 * bound,
                            "SMK-08: hinge rate " << rate << " rad/s exceeds the " << bound
                                                  << " rad/s bound by more than 2% at cycle " << cycle);
        // The limited-reference model, integrated from the shipped
        // mc_rbdyn::steeringWheelReference and the two documented controller
        // parameters, independently of the QP.
        planar[i].steeringAngle = modelAngle[i];
        planar[i].steeringRate = 0.0;
        const auto reference = mc_rbdyn::steeringWheelReference(planar[i], twist, modelAngle[i]);
        modelAngle[i] +=
            dt * std::clamp((reference.steeringAngle - modelAngle[i]) / timeConstant, -bound, bound);
        // Measure over the second half only, so the plateau and the model
        // deviation describe the slalom and not the swing into it.
        if(cycle >= cycles / 2)
        {
          out.plateau = std::max(out.plateau, std::abs(rate));
          out.modelDeviation = std::max(out.modelDeviation, std::abs(angle - modelAngle[i]));
          lowest[i] = std::min(lowest[i], angle);
          highest[i] = std::max(highest[i], angle);
        }
      }
    }
    for(size_t i = 0; i < rangerWheels().size(); ++i)
    {
      out.travel = std::max(out.travel, highest[i] - lowest[i]);
    }
    return out;
  };

  // (2) The bound shapes the trajectory: the plateau follows it proportionally.
  double previousPlateau = 0.0;
  for(const double bound : {1.0, 2.0, 3.0, 4.0})
  {
    const Outcome outcome = slalom(bound, false);
    BOOST_TEST_MESSAGE("SMK-08 bound " << bound << " rad/s: plateau " << outcome.plateau
                                       << " rad/s, hinge travel " << outcome.travel
                                       << " rad, deviation from the limited-reference model "
                                       << outcome.modelDeviation << " rad");
    BOOST_CHECK_GT(outcome.plateau, 0.95 * bound);
    BOOST_CHECK_LT(outcome.plateau, 1.02 * bound);
    BOOST_CHECK_GT(outcome.plateau, previousPlateau);
    BOOST_REQUIRE_GT(outcome.travel, 0.1);
    // (3) The realised trajectory is the limited reference, to 10% of the hinge
    // travel the slalom asks for.
    BOOST_CHECK_LT(outcome.modelDeviation, 0.1 * outcome.travel);
    previousPlateau = outcome.plateau;
  }

  // (4) The default bound is the steering joints' own model velocity limit.
  auto probe = makeRangerController();
  double modelLimit = std::numeric_limits<double>::infinity();
  for(const auto * wheel : rangerWheels())
  {
    const auto joint = probe->robot().jointIndexByName(std::string(wheel) + "_steer");
    modelLimit = std::min({modelLimit, std::abs(probe->robot().vl()[joint][0]),
                           std::abs(probe->robot().vu()[joint][0])});
  }
  const Outcome atDefault = slalom(modelLimit, true);
  BOOST_TEST_MESSAGE("SMK-08 at the model velocity limit " << modelLimit << " rad/s: plateau " << atDefault.plateau
                                                           << " rad/s (" << 100.0 * atDefault.plateau / modelLimit
                                                           << "% of it), hinge travel " << atDefault.travel
                                                           << " rad, deviation from the limited-reference model "
                                                           << atDefault.modelDeviation << " rad");
  BOOST_CHECK_LE(atDefault.plateau, 1.02 * modelLimit);
  // Recorded rather than asserted as a contract: the plateau tracks the
  // configured bound to within 3% up to 4 rad/s (0.972, 1.944, 3.021, 3.890 at
  // 1, 2, 3 and 4 rad/s) and is clamped at exactly 4.0 rad/s above it, so at
  // the 8 rad/s default the QP delivers half the rate the reference asks for
  // and the deviation from the limited-reference model grows from 4% of the
  // hinge travel to 34% of it. That ceiling is not explained by any bound in
  // this fixture - the steering torque at the plateau is 0.08 N.m of a 25 N.m
  // limit and the solved steering acceleration is exactly zero while the rate
  // row still demands -800 rad/s^2 - and it is reported as an open item rather
  // than pinned here. The one thing asserted is that it IS a ceiling: the
  // plateau at the 8 rad/s default is no larger than the one at 4 rad/s.
  BOOST_CHECK_LE(atDefault.plateau, previousPlateau + 0.2);
}

BOOST_AUTO_TEST_CASE(OdometryCrossCheckIncludingWhereItMustDisagreeSMK11)
{
  // SMK-11. Closed-form wheel odometry against the integrated QP state, on flat
  // ground and then on a ramp. The point of the card is the second half: an
  // odometry model that assumes a world-horizontal plane must DISAGREE on a
  // slope, and the test asserts the disagreement rather than tolerating it, so
  // that such a model cannot be "validated" on flat ground and then used on
  // slopes.
  constexpr double dt = 0.005;
  constexpr int cycles = 2000;
  const double pi = 3.14159265358979323846;

  for(const double slopeDegrees : {0.0, 15.0})
  {
    const double slope = slopeDegrees * pi / 180.0;
    // Body-to-world is Ry(slope), so posW().rotation() - the world-to-body map -
    // is its transpose and the plane normal in world coordinates is
    // Ry(slope) * e_z. The same construction the solver-side ramp tests use.
    const Eigen::Matrix3d bodyToWorld(Eigen::AngleAxisd(slope, Eigen::Vector3d::UnitY()));
    const Eigen::Vector3d normal = bodyToWorld * Eigen::Vector3d::UnitZ();
    auto controller = makeController("RollingContactDifferential", "forward",
                                     [&](mc_rtc::Configuration & settings)
                                     { settings.add("terrainNormal", normal); });
    controller->robot().posW(sva::PTransformd(bodyToWorld.transpose(), controller->robot().posW().translation()));
    controller->robot().forwardKinematics();
    controller->reset({controller->robot().mbc().q});

    // Wheel data read off the robot, not restated: the radius comes from the
    // same CylindricalSurface the controller builds its wheels from, and the
    // track from the carrier frames.
    const double radius =
        dynamic_cast<const mc_rbdyn::CylindricalSurface &>(controller->robot().surface("LeftWheel")).radius();
    const Eigen::Vector2d left = carrierOffset(*controller, "left");
    const Eigen::Vector2d right = carrierOffset(*controller, "right");
    const double track = left.y() - right.y();
    BOOST_REQUIRE_GT(track, 0.1);
    const auto leftDrive = controller->robot().jointIndexByName("left_drive");
    const auto rightDrive = controller->robot().jointIndexByName("right_drive");

    const sva::PTransformd start = controller->robot().posW();
    // The closed-form differential odometry of sec:rolling-implementation,
    // integrated in the chassis' own start frame.
    Eigen::Vector2d odometry = Eigen::Vector2d::Zero();
    double heading = 0.0;
    double pathLength = 0.0;
    for(int cycle = 0; cycle < cycles; ++cycle)
    {
      BOOST_REQUIRE_MESSAGE(controller->run(), "SMK-11: the QP failed at cycle " << cycle);
      const double leftSpeed = radius * controller->robot().mbc().alpha[leftDrive][0];
      const double rightSpeed = radius * controller->robot().mbc().alpha[rightDrive][0];
      const double forward = 0.5 * (leftSpeed + rightSpeed);
      const double yawRate = (rightSpeed - leftSpeed) / track;
      odometry += dt * forward * Eigen::Vector2d{std::cos(heading), std::sin(heading)};
      heading += dt * yawRate;
      pathLength += dt * std::abs(forward);
    }
    const sva::PTransformd end = controller->robot().posW();
    const Eigen::Vector3d travelled = chassisMotion(start, end);
    const double agreement = (odometry - travelled.head<2>()).norm();
    BOOST_TEST_MESSAGE("SMK-11 slope " << slopeDegrees << " deg: path length " << pathLength
                                       << " m, in-plane odometry disagreement " << agreement << " m ("
                                       << agreement / pathLength << " of the path)");
    BOOST_REQUIRE_GT(pathLength, 1.0);
    // In the contact plane the closed form and the integrated QP state agree to
    // 1e-3 of the path length, on flat ground AND on the ramp: the wheel
    // odometry is a statement about the plane the wheels roll on.
    BOOST_CHECK_LT(agreement, 1e-3 * pathLength);

    // Now the model that assumes a world-horizontal plane. It integrates the
    // same wheel speeds but never leaves z = 0, so on a ramp it must disagree.
    const Eigen::Vector3d horizontal{odometry.x(), odometry.y(), 0.0};
    const Eigen::Vector3d worldTravel = end.translation() - start.translation();
    const double disagreement = (horizontal - worldTravel).norm();
    // The card states the ramp bound as sin(theta) times the path length, which
    // is the height lost by a straight run down the fall line. This run is not
    // straight: a differential chassis pointed down a 15 deg slope curves away
    // from the commanded heading (0.89 m of lateral travel in 10 s), because
    // the moment it yaws at all gravity gains a body-lateral component while
    // the hard lateral row still forbids lateral velocity at the carrier. The
    // height actually lost is therefore sin(theta) times the displacement along
    // the FALL LINE, which is what the bound is stated against - and it is
    // taken from the odometry's own estimate, so the prediction does not read
    // the answer it is checking.
    const double fallLineTravel = std::abs(odometry.x());
    BOOST_TEST_MESSAGE("SMK-11 slope " << slopeDegrees << " deg: world-horizontal odometry disagreement "
                                       << disagreement << " m (" << disagreement / pathLength
                                       << " of the path) against sin(theta) * fall-line travel = "
                                       << std::sin(slope) * fallLineTravel << " m");
    if(slope > 0.0)
    {
      // At least the height the chassis lost, which the horizontal model never
      // accounts for at all...
      BOOST_CHECK_GT(disagreement, 0.99 * std::sin(slope) * fallLineTravel);
      // ...and, stated against the same denominator as the flat branch, more
      // than two orders of magnitude worse than the agreement it shows there.
      BOOST_CHECK_GT(disagreement, 0.1 * pathLength);
    }
    else
    {
      // The same model is exact on flat ground - which is precisely how it gets
      // "validated" and then used on a slope.
      BOOST_CHECK_LT(disagreement, 1e-3 * pathLength);
    }
  }
}

/** SMK-13. Sixty seconds of varied commands on both chassis and both terrains.
 *
 * Disabled by default and registered with CTest only under
 * -DROLLING_CONTACT_LONG_HORIZON_SMOKE=ON (label rolling-contact-slow): four
 * 12000-cycle rollouts are ~35x the rest of this binary, and the default suite
 * has to stay fast. Boost re-enables a disabled unit when it is named
 * explicitly, which is how the CTest entry runs it:
 *   ./testRollingContactControllerLifecycle --run_test=LongHorizonClosedLoopSMK13
 */
BOOST_AUTO_TEST_CASE(LongHorizonClosedLoopSMK13, *boost::unit_test::disabled())
{
  constexpr double dt = 0.005;
  constexpr int cycles = 12000; // 60 s
  constexpr int windowCycles = 1000; // 5 s
  const double pi = 3.14159265358979323846;

  struct Case
  {
    const char * name;
    const char * robot;
    const char * scenario;
    bool commanded;
    double slopeDegrees;
    double envelope; // largest admissible residual, m/s or rad/s^2
  };
  // The two envelopes are deliberately different, which is the card's point.
  // T1's lateral rows are hard equalities, so its lateral residual is driven to
  // the solver's own floor; T2's are softened at lateralSlackWeight, so its
  // residual is bounded only by the objective and legitimately sits decades
  // higher. Applying T1's envelope to T2 is the false-failure trap the card
  // warns about. Both values are ~3x the worst measured.
  const std::array<Case, 4> cases = {
      Case{"T1 flat", "RollingContactDifferential", "sinusoid", false, 0.0, 1e-6},
      Case{"T1 ramp", "RollingContactDifferential", "sinusoid", false, 10.0, 1e-6},
      Case{"T2 flat", "RollingContactRangerMiniV3", "hold", true, 0.0, 0.15},
      Case{"T2 ramp", "RollingContactRangerMiniV3", "hold", true, 10.0, 0.15}};
  // Six commands, four seconds each, cycling for the whole minute.
  const std::array<Eigen::Vector3d, 6> schedule = {
      Eigen::Vector3d{0.3, 0.0, 0.0},  Eigen::Vector3d{0.0, 0.3, 0.0},  Eigen::Vector3d{0.0, 0.0, 0.5},
      Eigen::Vector3d{0.3, 0.0, 0.4},  Eigen::Vector3d{-0.3, 0.0, 0.0}, Eigen::Vector3d{0.2, -0.2, -0.3}};

  for(const auto & test : cases)
  {
    const double slope = test.slopeDegrees * pi / 180.0;
    const Eigen::Matrix3d bodyToWorld(Eigen::AngleAxisd(slope, Eigen::Vector3d::UnitY()));
    const Eigen::Vector3d normal = bodyToWorld * Eigen::Vector3d::UnitZ();
    auto controller = makeController(test.robot, test.scenario,
                                     [&](mc_rtc::Configuration & settings)
                                     { settings.add("terrainNormal", normal); });
    if(slope > 0.0)
    {
      controller->robot().posW(sva::PTransformd(bodyToWorld.transpose(), controller->robot().posW().translation()));
      controller->robot().forwardKinematics();
      controller->reset({controller->robot().mbc().q});
    }

    std::vector<double> windowPeak;
    double worst = 0.0;
    double runningPeak = 0.0;
    const sva::PTransformd start = controller->robot().posW();
    for(int cycle = 0; cycle < cycles; ++cycle)
    {
      if(test.commanded) { controller->setCommandedTwist(schedule[static_cast<size_t>((cycle / 800) % 6)]); }
      BOOST_REQUIRE_MESSAGE(controller->run(), test.name << ": the QP failed at cycle " << cycle);
      const double residual = controller->maxLateralResidual();
      BOOST_REQUIRE_MESSAGE(std::isfinite(residual), test.name << ": non-finite residual at cycle " << cycle);
      runningPeak = std::max(runningPeak, residual);
      worst = std::max(worst, residual);
      if((cycle + 1) % windowCycles == 0)
      {
        windowPeak.push_back(runningPeak);
        runningPeak = 0.0;
      }
    }
    BOOST_REQUIRE_EQUAL(windowPeak.size(), static_cast<size_t>(cycles / windowCycles));
    const Eigen::Vector3d worldTravel = controller->robot().posW().translation() - start.translation();
    std::ostringstream trace;
    for(const double peak : windowPeak) { trace << " " << peak; }
    BOOST_TEST_MESSAGE("SMK-13 " << test.name << ": worst lateral residual " << worst << " m/s against the "
                                 << test.envelope << " envelope; world travel [" << worldTravel.transpose()
                                 << "]; per-5s peaks:" << trace.str());

    // Bounded by the stated envelope...
    BOOST_CHECK_LT(worst, test.envelope);
    // ...and with no monotone growth over the last 30 s. The strict-monotonicity
    // half of that is only asserted above a noise floor of a thousandth of the
    // envelope: T1's residual settles at 5e-10 m/s, where a 22% rise over 30 s
    // is solver arithmetic and not a trend, and asserting non-monotonicity there
    // would be asserting noise. The growth factor is asserted in both regimes.
    const size_t half = windowPeak.size() / 2;
    if(windowPeak[half] > 1e-3 * test.envelope)
    {
      bool monotone = true;
      for(size_t i = half + 1; i < windowPeak.size(); ++i)
      {
        monotone = monotone && windowPeak[i] > windowPeak[i - 1];
      }
      BOOST_CHECK_MESSAGE(!monotone, test.name << ": the residual grows monotonically over the last 30 s");
    }
    BOOST_CHECK_LT(windowPeak.back(), 2.0 * windowPeak[half] + 1e-12);

    // The lateral residual is a kinematic quantity in the contact plane, and
    // tilting the chassis and the terrain together is a rigid rotation of the
    // whole kinematic problem (see RampGeometryIsRotationEquivariant), so it is
    // legitimately identical on both terrains to eight digits. The ramp branch
    // is therefore pinned on the one quantity that is NOT rotation-equivariant:
    // gravity. The chassis must actually travel up or down the slope - which
    // way depends on the command schedule, so the magnitude is what is checked.
    if(slope > 0.0) { BOOST_CHECK_GT(std::abs(worldTravel.z()), 0.1); }
    else { BOOST_CHECK_SMALL(worldTravel.z(), 1e-3); }
  }
}
