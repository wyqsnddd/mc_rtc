#include "mc_rolling_contact_controller.h"

#include <mc_rbdyn/RobotLoader.h>

#include <boost/test/unit_test.hpp>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <array>
#include <cmath>
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
  auto config = controllerConfiguration("hold");
  config("RollingContact").add("closedLoopFeedback", true);
  mc_control::MCRollingContactController controller(
      robotModule("RollingContactRangerMiniV3"), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller.reset({controller.robot().mbc().q});

  // BodySensor orientation follows the mc_rbdyn convention (inertial to
  // sensor). The controller must convert this measured packet into the free
  // joint MBC and mirror it into realRobot() before the QP runs.
  const Eigen::Vector3d measuredPosition{0.11, -0.035, 0.16};
  const Eigen::Quaterniond measuredWorldToBody{Eigen::AngleAxisd(0.23, Eigen::Vector3d::UnitZ())};
  const Eigen::Vector3d measuredLinearVelocity{0.18, -0.07, 0.0};
  const Eigen::Vector3d measuredAngularVelocity{0.0, 0.0, 0.21};
  auto & sensor = controller.robot().data()->bodySensors[
      controller.robot().data()->bodySensorsIndex.at("FloatingBase")];
  sensor.position(measuredPosition);
  sensor.orientation(measuredWorldToBody);
  sensor.linearVelocity(measuredLinearVelocity);
  sensor.angularVelocity(measuredAngularVelocity);

  BOOST_REQUIRE(controller.run());
  const auto & control = controller.robot();
  const auto & real = controller.realRobot();
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
  auto config = controllerConfiguration("keyboard");
  config("RollingContact").add("closedLoopFeedback", true);
  mc_control::MCRollingContactController controller(
      robotModule("RollingContactRangerMiniV3"), 0.005, config, mc_control::MCController::Backend::Tasks);
  controller.reset({controller.robot().mbc().q});

  // Sensor orientation is the inertial-to-body rotation, so this is a chassis
  // whose +X axis really points at -0.4 rad in the world.
  constexpr double sensorYaw = 0.4;
  auto & sensor = controller.robot().data()->bodySensors[
      controller.robot().data()->bodySensorsIndex.at("FloatingBase")];
  sensor.orientation(Eigen::Quaterniond{Eigen::AngleAxisd(sensorYaw, Eigen::Vector3d::UnitZ())});
  sensor.position(controller.robot().posW().translation());
  BOOST_REQUIRE(controller.run());

  // The true world heading of the chassis' +X axis: E_0_b^T * e_x.
  const Eigen::Vector3d worldForward = controller.robot().posW().rotation().transpose().col(0);
  const double worldYaw = std::atan2(worldForward.y(), worldForward.x());
  BOOST_CHECK_CLOSE(worldYaw, -sensorYaw, 1e-6);

  // baseYawTarget_ holds the mirrored value, which is exactly why run() negates
  // it. If this ever equals worldYaw instead, the negation must go with it.
  const double baseYawTarget = controller.datastore().call<double>("RollingContact::GetBaseYawTarget");
  BOOST_CHECK_CLOSE(baseYawTarget, sensorYaw, 1e-6);
  BOOST_CHECK_CLOSE(-baseYawTarget, worldYaw, 1e-6);

  // Now pin the consequence rather than only the premise: a forward command
  // must walk the integrated position target along the chassis' true world
  // heading. Without the negation the target walks along +sensorYaw instead,
  // i.e. mirrored about the world X axis. Drive it through the GUI input,
  // which the keyboard poll adds to the key state every cycle; setCommandedTwist
  // would be overwritten by that same poll.
  BOOST_REQUIRE(controller.gui()->handleRequest({"Rolling Contact", "Command"}, "Forward velocity",
                                                mc_rtc::Configuration::fromData("0.3")));
  const Eigen::Vector3d targetBefore =
      controller.datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionTarget");
  for(int cycle = 0; cycle < 20; ++cycle) { BOOST_REQUIRE(controller.run()); }
  const Eigen::Vector3d targetAfter =
      controller.datastore().call<Eigen::Vector3d>("RollingContact::GetBasePositionTarget");

  const Eigen::Vector3d walked = targetAfter - targetBefore;
  BOOST_REQUIRE_GT(walked.head<2>().norm(), 1e-6);
  const double walkedYaw = std::atan2(walked.y(), walked.x());
  BOOST_TEST_MESSAGE("[keyboard-heading] sensorYaw=" << sensorYaw << " worldYaw=" << worldYaw
                                                     << " walkedYaw=" << walkedYaw);
  BOOST_CHECK_SMALL(std::remainder(walkedYaw - worldYaw, 2.0 * 3.14159265358979323846), 1e-6);
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
