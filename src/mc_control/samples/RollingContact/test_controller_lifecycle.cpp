#include "mc_rolling_contact_controller.h"

#include <mc_rbdyn/RobotLoader.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <limits>
#include <string>

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
