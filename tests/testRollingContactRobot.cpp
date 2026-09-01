#include <mc_rbdyn/CylindricalSurface.h>
#include <mc_rbdyn/RollingContact.h>
#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/Robots.h>

#include <RBDyn/MultiBodyConfig.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <cmath>
#include <string>

#include "utils.h"

namespace
{

mc_rbdyn::RobotModulePtr loadModule(const std::string & variant)
{
  configureRobotLoader();
  return mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), variant);
}

void checkBodyInertias(const mc_rbdyn::Robot & robot)
{
  for(const auto & body : robot.mb().bodies())
  {
    BOOST_CHECK_GT(body.inertia().mass(), 0.0);
    BOOST_CHECK(body.inertia().inertia().allFinite());
    BOOST_CHECK_GT(body.inertia().inertia().diagonal().minCoeff(), 0.0);
  }
}

mc_rbdyn::RollingContactDescription frontLeftDescription()
{
  mc_rbdyn::RollingContactDescription description;
  description.name = "front_left";
  description.carrierFrame = "front_left_carrier";
  description.wheelBody = "front_left_wheel";
  description.driveJoint = "front_left_drive";
  description.steeringJoint = "front_left_steer";
  description.radius = 0.2;
  description.width = 0.08;
  return description;
}

void setVelocity(mc_rbdyn::Robot & robot, const Eigen::VectorXd & velocity)
{
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.forwardVelocity();
}

} // namespace

BOOST_AUTO_TEST_CASE(LoadDifferentialRollingRobot)
{
  auto module = loadModule("rolling_diff");
  BOOST_REQUIRE(module);
  BOOST_CHECK_EQUAL(module->name, "rolling_diff");
  BOOST_CHECK_EQUAL(module->ref_joint_order().size(), 2);
  BOOST_CHECK_CLOSE(module->_default_attitude[6], 0.2, 1e-12);

  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  BOOST_CHECK_EQUAL(robot.mb().nrDof(), 8);
  BOOST_CHECK(robot.hasJoint("left_drive"));
  BOOST_CHECK(robot.hasJoint("right_drive"));
  BOOST_CHECK(robot.hasBody("left_wheel"));
  BOOST_CHECK(robot.hasBody("right_wheel"));
  BOOST_CHECK(robot.hasFrame("left_carrier"));
  BOOST_CHECK(robot.hasFrame("right_carrier"));
  BOOST_CHECK(robot.hasSurface("LeftWheel"));
  BOOST_CHECK(robot.hasSurface("RightWheel"));
  const auto * left = dynamic_cast<const mc_rbdyn::CylindricalSurface *>(&robot.surface("LeftWheel"));
  BOOST_REQUIRE(left);
  BOOST_CHECK_CLOSE(left->radius(), 0.2, 1e-12);
  BOOST_CHECK_CLOSE(left->width(), 0.08, 1e-12);
  const auto leftJoint = robot.jointIndexByName("left_drive");
  const auto rightJoint = robot.jointIndexByName("right_drive");
  BOOST_CHECK(robot.mb().joint(leftJoint).type() == rbd::Joint::Rev);
  BOOST_CHECK_SMALL((robot.mb().joint(leftJoint).motionSubspace().topRows<3>().col(0)
                     - Eigen::Vector3d::UnitY())
                        .norm(),
                    1e-12);
  BOOST_CHECK_CLOSE(robot.tl()[leftJoint][0], -35.0, 1e-12);
  BOOST_CHECK_CLOSE(robot.tu()[leftJoint][0], 35.0, 1e-12);
  BOOST_CHECK_CLOSE(robot.vu()[rightJoint][0], 30.0, 1e-12);

  const Eigen::Vector3d centerBefore = robot.frame("left_carrier").position().translation();
  const Eigen::Vector3d contactBefore = centerBefore - 0.2 * Eigen::Vector3d::UnitZ();
  BOOST_CHECK_SMALL(contactBefore.z(), 1e-12);
  robot.mbc().q[leftJoint][0] = 1.234;
  robot.mbc().q[rightJoint][0] = -0.731;
  robot.forwardKinematics();
  const Eigen::Vector3d centerAfter = robot.frame("left_carrier").position().translation();
  const Eigen::Vector3d contactAfter = centerAfter - 0.2 * Eigen::Vector3d::UnitZ();
  BOOST_CHECK_SMALL((centerAfter - centerBefore).norm(), 1e-12);
  BOOST_CHECK_SMALL((contactAfter - contactBefore).norm(), 1e-12);
  checkBodyInertias(robot);
}

BOOST_AUTO_TEST_CASE(LoadFourSteeringRollingRobot)
{
  auto module = loadModule("rolling_4s");
  BOOST_REQUIRE(module);
  BOOST_CHECK_EQUAL(module->ref_joint_order().size(), 8);
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  BOOST_CHECK_EQUAL(robot.mb().nrDof(), 14);
  for(const std::string corner : {"front_left", "front_right", "rear_left", "rear_right"})
  {
    BOOST_CHECK(robot.hasJoint(corner + "_steer"));
    BOOST_CHECK(robot.hasJoint(corner + "_drive"));
    BOOST_CHECK(robot.hasBody(corner + "_knuckle"));
    BOOST_CHECK(robot.hasBody(corner + "_wheel"));
    BOOST_CHECK(robot.hasFrame(corner + "_carrier"));
    const auto steer = robot.jointIndexByName(corner + "_steer");
    const auto drive = robot.jointIndexByName(corner + "_drive");
    BOOST_CHECK_SMALL((robot.mb().joint(steer).motionSubspace().topRows<3>().col(0)
                       - Eigen::Vector3d::UnitZ())
                          .norm(),
                      1e-12);
    BOOST_CHECK_SMALL((robot.mb().joint(drive).motionSubspace().topRows<3>().col(0)
                       - Eigen::Vector3d::UnitY())
                          .norm(),
                      1e-12);
    BOOST_CHECK_CLOSE(robot.tu()[steer][0], 25.0, 1e-12);
    BOOST_CHECK_CLOSE(robot.tu()[drive][0], 35.0, 1e-12);
  }
  BOOST_CHECK(robot.hasSurface("FrontLeftWheel"));
  BOOST_CHECK(robot.hasSurface("FrontRightWheel"));
  BOOST_CHECK(robot.hasSurface("RearLeftWheel"));
  BOOST_CHECK(robot.hasSurface("RearRightWheel"));
  checkBodyInertias(robot);
}

BOOST_AUTO_TEST_CASE(RejectUnknownRollingRobotVariant)
{
  BOOST_CHECK_THROW(loadModule("rolling_unknown"), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ResolvedRollingGeometryMatchesRBDynFiniteDifferences)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  const auto steeringJoint = robot.jointIndexByName("front_left_steer");
  const auto driveJoint = robot.jointIndexByName("front_left_drive");
  robot.mbc().q[steeringJoint][0] = 0.37;
  robot.mbc().q[driveJoint][0] = -0.81;
  robot.forwardKinematics();

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, frontLeftDescription());
  const auto originalQ = robot.mbc().q;
  constexpr double step = 1e-7;
  double worstPositionError = 0.0;
  double worstDirectionError = 0.0;
  for(int column = 0; column < robot.mb().nrDof(); ++column)
  {
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
    velocity(column) = 1.0;
    robot.mbc().q = originalQ;
    robot.forwardKinematics();
    setVelocity(robot, velocity);
    const auto nominal = geometry.update(robot, Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d expectedPositionColumn = geometry.kinematics().carrierJacobian.col(column);
    const Eigen::Vector3d expectedRollingRate = nominal.rollingDirectionRate;
    const Eigen::Vector3d expectedLateralRate = nominal.lateralDirectionRate;

    robot.mbc().q = originalQ;
    setVelocity(robot, velocity);
    robot.eulerIntegration(step);
    robot.forwardKinematics();
    robot.forwardVelocity();
    const auto plus = geometry.update(robot, Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d plusCenter = plus.carrierCenter;
    const Eigen::Vector3d plusRolling = plus.rollingDirection;
    const Eigen::Vector3d plusLateral = plus.lateralDirection;

    robot.mbc().q = originalQ;
    setVelocity(robot, velocity);
    robot.eulerIntegration(-step);
    robot.forwardKinematics();
    robot.forwardVelocity();
    const auto minus = geometry.update(robot, Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d positionDifference = (plusCenter - minus.carrierCenter) / (2.0 * step);
    const Eigen::Vector3d rollingDifference = (plusRolling - minus.rollingDirection) / (2.0 * step);
    const Eigen::Vector3d lateralDifference = (plusLateral - minus.lateralDirection) / (2.0 * step);
    worstPositionError = std::max(worstPositionError, (positionDifference - expectedPositionColumn).norm());
    worstDirectionError =
        std::max(worstDirectionError, (rollingDifference - expectedRollingRate).norm());
    worstDirectionError =
        std::max(worstDirectionError, (lateralDifference - expectedLateralRate).norm());
  }
  BOOST_TEST_MESSAGE("Worst carrier Jacobian finite-difference error: " << worstPositionError);
  BOOST_TEST_MESSAGE("Worst rolling-frame finite-difference error: " << worstDirectionError);
  BOOST_CHECK_SMALL(worstPositionError, 1e-6);
  BOOST_CHECK_SMALL(worstDirectionError, 1e-6);
}

BOOST_AUTO_TEST_CASE(ResolvedRollingBiasMatchesMatrixDirectionalDerivative)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  const auto steeringJoint = robot.jointIndexByName("front_left_steer");
  robot.mbc().q[steeringJoint][0] = -0.29;
  robot.forwardKinematics();

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, frontLeftDescription());
  Eigen::VectorXd velocity(robot.mb().nrDof());
  velocity << 0.13, -0.21, 0.37, 0.71, -0.26, 0.19, 0.83, -1.17, 0.31, -0.27, 0.22, 0.16, -0.34,
      0.29;
  setVelocity(robot, velocity);
  const Eigen::Vector3d normal =
      Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitZ();
  const auto originalQ = robot.mbc().q;
  const Eigen::Vector3d expectedBias = geometry.update(robot, normal).accelerationBias;
  constexpr double step = 2e-7;

  robot.mbc().q = originalQ;
  setVelocity(robot, velocity);
  robot.eulerIntegration(step);
  robot.forwardKinematics();
  robot.forwardVelocity();
  const Eigen::MatrixXd plusMatrix = geometry.update(robot, normal).rollingMatrix;

  robot.mbc().q = originalQ;
  setVelocity(robot, velocity);
  robot.eulerIntegration(-step);
  robot.forwardKinematics();
  robot.forwardVelocity();
  const Eigen::MatrixXd minusMatrix = geometry.update(robot, normal).rollingMatrix;
  const Eigen::Vector3d finiteDifference = (plusMatrix * velocity - minusMatrix * velocity) / (2.0 * step);
  BOOST_TEST_MESSAGE("A-dot alpha error: " << (finiteDifference - expectedBias).norm());
  BOOST_CHECK_SMALL((finiteDifference - expectedBias).norm(), 1e-5);
}

BOOST_AUTO_TEST_CASE(FourSteeringResolvedGeometryAcceptsAckermannTwist)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  constexpr double linearSpeed = 0.15;
  constexpr double yawRate = 0.05;
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3},
                                                  Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3},
                                                  Eigen::Vector2d{-0.45, -0.3}};
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
  velocity(2) = yawRate;
  velocity(3) = linearSpeed;
  for(size_t i = 0; i < names.size(); ++i)
  {
    const double x = linearSpeed - yawRate * offsets[i].y();
    const double y = yawRate * offsets[i].x();
    const auto steer = robot.jointIndexByName(names[i] + "_steer");
    const auto drive = robot.jointIndexByName(names[i] + "_drive");
    robot.mbc().q[steer][0] = std::atan2(y, x);
    velocity(robot.mb().jointPosInDof(static_cast<int>(drive))) = std::hypot(x, y) / 0.2;
  }
  robot.forwardKinematics();
  setVelocity(robot, velocity);

  double residual = 0.0;
  for(const auto & name : names)
  {
    auto description = frontLeftDescription();
    description.name = name;
    description.carrierFrame = name + "_carrier";
    description.wheelBody = name + "_wheel";
    description.driveJoint = name + "_drive";
    description.steeringJoint = name + "_steer";
    mc_rbdyn::RollingContactRobotGeometry geometry(robot, description);
    const auto & result = geometry.update(robot, Eigen::Vector3d::UnitZ());
    residual = std::max(residual, result.velocityResidual.lpNorm<Eigen::Infinity>());
  }
  BOOST_TEST_MESSAGE("Four-steering Ackermann velocity residual: " << residual);
  BOOST_CHECK_SMALL(residual, 1e-10);
}
