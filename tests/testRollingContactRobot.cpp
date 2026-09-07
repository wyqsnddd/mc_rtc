#include <mc_rbdyn/CylindricalSurface.h>
#include <mc_rbdyn/RollingContact.h>
#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/Robots.h>

#include <mc_rtc/constants.h>

#include <RBDyn/MultiBodyConfig.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "utils.h"

namespace
{

mc_rbdyn::RobotModulePtr loadModule(const std::string & variant)
{
  configureRobotLoader();
  if(variant == "rolling_diff") { return mc_rbdyn::RobotLoader::get_robot_module("RollingContactDifferential"); }
  if(variant == "rolling_4s") { return mc_rbdyn::RobotLoader::get_robot_module("RollingContactFourSteering"); }
  if(variant == "ranger_mini_v3")
  {
    return mc_rbdyn::RobotLoader::get_robot_module("RollingContactRangerMiniV3");
  }
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

/** Description of the "left" wheel of the differential-drive variant: no steering joint. */
mc_rbdyn::RollingContactDescription leftDescription()
{
  mc_rbdyn::RollingContactDescription description;
  description.name = "left";
  description.carrierFrame = "left_carrier";
  description.wheelBody = "left_wheel";
  description.driveJoint = "left_drive";
  description.radius = 0.2;
  description.width = 0.08;
  return description;
}

void setVelocity(mc_rbdyn::Robot & robot, const Eigen::VectorXd & velocity)
{
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.forwardVelocity();
}

const std::array<std::string, 4> & rangerCorners()
{
  static const std::array<std::string, 4> corners = {"front_left", "front_right", "rear_left", "rear_right"};
  return corners;
}

/** Headings used by the chassis-heading-invariance pins. The first one is the
 * reference every other heading is compared against.
 */
const std::array<double, 4> & headingSweep()
{
  static const std::array<double, 4> yaws = {0.0, mc_rtc::constants::PI / 4.0, mc_rtc::constants::PI / 2.0,
                                             -2.0 * mc_rtc::constants::PI / 3.0};
  return yaws;
}

/** Place the floating base at a pure yaw about the world vertical. */
void setChassisYaw(mc_rbdyn::Robot & robot, double yaw)
{
  robot.posW(sva::PTransformd(sva::RotZ(yaw), robot.posW().translation()));
  robot.forwardKinematics();
  robot.forwardVelocity();
}

/** Carrier offset from the chassis origin, in the inertial frame. */
Eigen::Vector3d worldWheelOffset(const mc_rbdyn::Robot & robot, const std::string & corner)
{
  return robot.frame(corner + "_carrier").position().translation()
         - robot.frame("chassis").position().translation();
}

} // namespace

BOOST_AUTO_TEST_CASE(LoadDifferentialRollingRobot)
{
  auto module = loadModule("rolling_diff");
  BOOST_REQUIRE(module);
  BOOST_CHECK_EQUAL(module->name, "rolling_diff");
  BOOST_REQUIRE_EQUAL(module->parameters().size(), 1);
  BOOST_CHECK_EQUAL(module->parameters()[0], "RollingContactDifferential");
  BOOST_REQUIRE_EQUAL(module->canonicalParameters().size(), 3);
  BOOST_CHECK_EQUAL(module->canonicalParameters()[2], "rolling_diff");
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

BOOST_AUTO_TEST_CASE(LoadRangerMiniV3RollingRobot)
{
  auto module = loadModule("ranger_mini_v3");
  BOOST_REQUIRE(module);
  BOOST_CHECK_EQUAL(module->name, "ranger_mini_v3");
  BOOST_REQUIRE_EQUAL(module->parameters().size(), 1);
  BOOST_CHECK_EQUAL(module->parameters()[0], "RollingContactRangerMiniV3");
  BOOST_REQUIRE_EQUAL(module->canonicalParameters().size(), 3);
  BOOST_CHECK_EQUAL(module->canonicalParameters()[2], "ranger_mini_v3");
  BOOST_CHECK_CLOSE(module->_default_attitude[6], 0.16, 1e-12);
  BOOST_CHECK_EQUAL(module->ref_joint_order().size(), 8);

  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  BOOST_CHECK_EQUAL(robot.mb().nrDof(), 14);
  // The free-joint/body-sensor origin is the chassis link origin. Keep this
  // explicit: a non-zero sensor transform makes the visual floating-base
  // marker and the controller's chassis task disagree even when FK succeeds.
  BOOST_REQUIRE(robot.hasBodySensor("FloatingBase"));
  const auto & floatingBase = robot.bodySensor("FloatingBase");
  BOOST_CHECK_EQUAL(floatingBase.parentBody(), "chassis");
  BOOST_CHECK_SMALL((floatingBase.X_b_s().matrix() - sva::PTransformd::Identity().matrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((robot.posW().matrix() - robot.frame("chassis").position().matrix()).norm(), 1e-12);
  BOOST_REQUIRE_EQUAL(module->_visual.at("chassis").size(), 1);
  BOOST_CHECK_SMALL(module->_visual.at("chassis").front().origin.translation().norm(), 1e-12);
  BOOST_CHECK_SMALL(
      (module->_visual.at("chassis").front().origin.rotation() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  BOOST_REQUIRE_EQUAL(module->_collision.at("chassis").size(), 1);
  BOOST_CHECK_SMALL(module->_collision.at("chassis").front().origin.translation().norm(), 1e-12);
  BOOST_CHECK_SMALL(
      (module->_collision.at("chassis").front().origin.rotation() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  const std::array<std::string, 4> corners = {"front_left", "front_right", "rear_left", "rear_right"};
  constexpr double halfPi = 1.5707963267948966;
  const std::array<Eigen::Vector2d, 4> expectedOffsets = {Eigen::Vector2d{0.247, 0.182},
                                                         Eigen::Vector2d{0.247, -0.182},
                                                         Eigen::Vector2d{-0.247, 0.182},
                                                         Eigen::Vector2d{-0.247, -0.182}};
  for(size_t i = 0; i < corners.size(); ++i)
  {
    const auto & corner = corners[i];
    BOOST_CHECK(robot.hasJoint(corner + "_steer"));
    BOOST_CHECK(robot.hasJoint(corner + "_drive"));
    BOOST_CHECK(robot.hasBody(corner + "_knuckle"));
    BOOST_CHECK(robot.hasBody(corner + "_wheel"));
    BOOST_CHECK(robot.hasFrame(corner + "_carrier"));
    const Eigen::Vector3d offset = robot.frame(corner + "_carrier").position().translation()
                                   - robot.frame("chassis").position().translation();
    BOOST_CHECK_SMALL((offset.head<2>() - expectedOffsets[i]).norm(), 1e-12);
    BOOST_CHECK_CLOSE(offset.z(), -0.035, 1e-12);
    const auto steer = robot.jointIndexByName(corner + "_steer");
    BOOST_CHECK_CLOSE(robot.ql()[steer][0], -halfPi, 1e-5);
    BOOST_CHECK_CLOSE(robot.qu()[steer][0], halfPi, 1e-5);
  }
  const auto * wheel = dynamic_cast<const mc_rbdyn::CylindricalSurface *>(&robot.surface("FrontLeftWheel"));
  BOOST_REQUIRE(wheel);
  BOOST_CHECK_CLOSE(wheel->radius(), 0.125, 1e-12);
  BOOST_CHECK_CLOSE(wheel->width(), 0.08, 1e-12);
  checkBodyInertias(robot);
}

BOOST_AUTO_TEST_CASE(RangerWheelOffsetsMatchTheUrdfLayout)
{
  // The controller derives wheelOffsets_ from the carrier frames expressed in
  // the chassis frame rather than from a hand-written table. Pin the layout
  // those frames must produce so a model or ordering change cannot silently
  // desynchronize the QP's steering geometry from the URDF.
  auto module = loadModule("ranger_mini_v3");
  BOOST_REQUIRE(module);
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();

  const std::array<std::pair<std::string, Eigen::Vector2d>, 4> layout = {
      std::make_pair(std::string("front_left"), Eigen::Vector2d(0.247, 0.182)),
      std::make_pair(std::string("front_right"), Eigen::Vector2d(0.247, -0.182)),
      std::make_pair(std::string("rear_left"), Eigen::Vector2d(-0.247, 0.182)),
      std::make_pair(std::string("rear_right"), Eigen::Vector2d(-0.247, -0.182))};
  const auto & chassis = robot.frame("chassis").position();
  for(const auto & [corner, expected] : layout)
  {
    BOOST_REQUIRE(robot.hasFrame(corner + "_carrier"));
    const Eigen::Vector3d worldOffset =
        robot.frame(corner + "_carrier").position().translation() - chassis.translation();
    const Eigen::Vector2d offset = (chassis.rotation() * worldOffset).head<2>();
    BOOST_CHECK_SMALL((offset - expected).norm(), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(PlanarWheelOffsetsAreChassisHeadingInvariant)
{
  // GEO-01. Assumption A1 (as:planar-basis) asks for the planar quantities to
  // be resolved in the chassis-aligned basis, so that rho_i is constant. The
  // controller already builds wheelOffsets_ that way
  // (mc_rolling_contact_controller.cpp:279-281); this pins the property the
  // assumption actually needs, rather than the one line that produces it.
  auto module = loadModule("ranger_mini_v3");
  BOOST_REQUIRE(module);
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();

  std::array<std::array<Eigen::Vector2d, 4>, 4> chassisOffsets;
  std::array<std::array<Eigen::Vector2d, 4>, 4> worldOffsets;
  for(size_t k = 0; k < headingSweep().size(); ++k)
  {
    setChassisYaw(robot, headingSweep()[k]);
    const auto & chassis = robot.frame("chassis").position();
    for(size_t i = 0; i < rangerCorners().size(); ++i)
    {
      const Eigen::Vector3d offset = worldWheelOffset(robot, rangerCorners()[i]);
      chassisOffsets[k][i] = (chassis.rotation() * offset).head<2>();
      worldOffsets[k][i] = offset.head<2>();
    }
  }

  double worstChassisDrift = 0.0;
  double smallestWorldDrift = std::numeric_limits<double>::infinity();
  for(size_t k = 1; k < headingSweep().size(); ++k)
  {
    for(size_t i = 0; i < rangerCorners().size(); ++i)
    {
      worstChassisDrift = std::max(worstChassisDrift, (chassisOffsets[k][i] - chassisOffsets[0][i]).norm());
      smallestWorldDrift = std::min(smallestWorldDrift, (worldOffsets[k][i] - worldOffsets[0][i]).norm());
    }
  }
  BOOST_TEST_MESSAGE("Chassis-frame offset drift across headings: " << worstChassisDrift
                                                                    << ", world-frame offset drift: "
                                                                    << smallestWorldDrift);
  BOOST_CHECK_SMALL(worstChassisDrift, 1e-12);
  // Non-vacuity: the same quantity resolved in the world-fixed basis moves by
  // at least 0.2 m over this heading sweep, so the check above is a real
  // observation about the chassis-aligned basis and not an identity.
  BOOST_CHECK_GT(smallestWorldDrift, 0.2);
}

BOOST_AUTO_TEST_CASE(ChassisAlignedPlanarBasisIsOrthonormalRightHandedAndBodyConstant)
{
  // GEO-02. The chassis-aligned basis of A1 is the chassis frame's own forward
  // axis projected onto the contact plane; assert it is a proper right-handed
  // orthonormal frame and that the planar offsets it resolves are unchanged by
  // a pure chassis rotation.
  auto module = loadModule("ranger_mini_v3");
  BOOST_REQUIRE(module);
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  const Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  const Eigen::Matrix3d worldBasis = mc_rbdyn::planarContactBasis(normal, Eigen::Vector3d::UnitX());

  std::array<std::array<Eigen::Vector2d, 4>, 4> chassisOffsets;
  std::array<std::array<Eigen::Vector2d, 4>, 4> worldFixedOffsets;
  for(size_t k = 0; k < headingSweep().size(); ++k)
  {
    setChassisYaw(robot, headingSweep()[k]);
    // position().rotation() maps world into the chassis, so its transpose holds
    // the chassis axes expressed in the world.
    const Eigen::Matrix3d chassisAxes = robot.frame("chassis").position().rotation().transpose();
    const Eigen::Matrix3d basis = mc_rbdyn::planarContactBasis(normal, chassisAxes.col(0));
    const Eigen::Vector3d ex = basis.col(0);
    const Eigen::Vector3d ey = basis.col(1);
    BOOST_CHECK_CLOSE(ex.norm(), 1.0, 1e-10);
    BOOST_CHECK_CLOSE(ey.norm(), 1.0, 1e-10);
    BOOST_CHECK_SMALL(ex.dot(ey), 1e-12);
    BOOST_CHECK_SMALL((ex.cross(ey) - normal).norm(), 1e-12);
    BOOST_CHECK_SMALL((basis.col(2) - normal).norm(), 1e-12);
    // The basis rotates with the chassis, so it is not the world-fixed one.
    if(k > 0) { BOOST_CHECK_GT((basis - worldBasis).norm(), 0.5); }

    for(size_t i = 0; i < rangerCorners().size(); ++i)
    {
      const Eigen::Vector3d offset = worldWheelOffset(robot, rangerCorners()[i]);
      chassisOffsets[k][i] = Eigen::Vector2d{ex.dot(offset), ey.dot(offset)};
      worldFixedOffsets[k][i] =
          Eigen::Vector2d{worldBasis.col(0).dot(offset), worldBasis.col(1).dot(offset)};
    }
  }

  double worstChassisDrift = 0.0;
  double smallestWorldDrift = std::numeric_limits<double>::infinity();
  for(size_t k = 1; k < headingSweep().size(); ++k)
  {
    for(size_t i = 0; i < rangerCorners().size(); ++i)
    {
      worstChassisDrift = std::max(worstChassisDrift, (chassisOffsets[k][i] - chassisOffsets[0][i]).norm());
      smallestWorldDrift = std::min(smallestWorldDrift, (worldFixedOffsets[k][i] - worldFixedOffsets[0][i]).norm());
    }
  }
  BOOST_TEST_MESSAGE("Chassis-aligned basis offset drift: " << worstChassisDrift
                                                            << ", world-fixed basis offset drift: "
                                                            << smallestWorldDrift);
  BOOST_CHECK_SMALL(worstChassisDrift, 1e-12);
  // Non-vacuity again: the world-fixed basis of the same construction, which is
  // what the controller uses as its yaw reference, does move.
  BOOST_CHECK_GT(smallestWorldDrift, 0.2);
}

BOOST_AUTO_TEST_CASE(DegenerateForwardAxisAbortsThePlanarBasis)
{
  // GEO-03. A forward axis parallel to the normal leaves nothing to normalize;
  // the surviving vector would be pure round-off, so the construction has to
  // abort rather than hand back a plausible-looking frame.
  const Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(normal, normal), std::invalid_argument);
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(normal, -normal), std::invalid_argument);
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(normal, Eigen::Vector3d{1e-12, 0.0, 1.0}), std::invalid_argument);
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(normal, Eigen::Vector3d::Zero()), std::invalid_argument);
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX()),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      mc_rbdyn::planarContactBasis(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
                                   Eigen::Vector3d::UnitX()),
      std::invalid_argument);
  BOOST_CHECK_THROW(
      mc_rbdyn::planarContactBasis(normal, Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity())),
      std::invalid_argument);

  // A forward axis that only just clears the threshold still yields a proper
  // frame; the guard rejects the unrecoverable case, not merely a steep one.
  const Eigen::Matrix3d basis = mc_rbdyn::planarContactBasis(normal, Eigen::Vector3d{1e-6, 0.0, 1.0});
  BOOST_CHECK_SMALL((basis.col(0) - Eigen::Vector3d::UnitX()).norm(), 1e-12);
  BOOST_CHECK_SMALL((basis.col(1) - Eigen::Vector3d::UnitY()).norm(), 1e-12);
  BOOST_CHECK_SMALL((basis.transpose() * basis - Eigen::Matrix3d::Identity()).norm(), 1e-12);

  // A non-vertical normal keeps the same contract.
  const Eigen::Vector3d ramp = Eigen::Vector3d{0.2, -0.1, 1.0}.normalized();
  const Eigen::Matrix3d rampBasis = mc_rbdyn::planarContactBasis(ramp, Eigen::Vector3d::UnitX());
  BOOST_CHECK_SMALL((rampBasis.transpose() * rampBasis - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  BOOST_CHECK_SMALL((rampBasis.col(0).cross(rampBasis.col(1)) - ramp).norm(), 1e-12);
  BOOST_CHECK_THROW(mc_rbdyn::planarContactBasis(ramp, ramp), std::invalid_argument);
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

BOOST_AUTO_TEST_CASE(SteeringSelectorAddressesTheSteeringDof)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, frontLeftDescription());
  // Exercise update() once to prove it never clobbers the selector set up at construction.
  geometry.update(robot, Eigen::Vector3d::UnitZ());

  const auto & selector = geometry.kinematics().steeringSelector;
  const auto steeringDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
  const auto driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_drive")));
  BOOST_REQUIRE_EQUAL(selector.size(), robot.mb().nrDof());
  BOOST_CHECK_CLOSE(selector(steeringDof), 1.0, 1e-12);
  BOOST_CHECK_SMALL(selector(driveDof), 1e-12);
  BOOST_CHECK_CLOSE(selector.sum(), 1.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(MeasuredRotatingRatesFollowJointVelocities)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();
  const auto steeringJoint = robot.jointIndexByName("front_left_steer");
  const auto driveJoint = robot.jointIndexByName("front_left_drive");

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, frontLeftDescription());

  robot.mbc().alpha[driveJoint][0] = 2.5;
  robot.mbc().alpha[steeringJoint][0] = -0.75;
  robot.forwardVelocity();

  const auto & result = geometry.update(robot, Eigen::Vector3d::UnitZ());
  BOOST_CHECK_CLOSE(result.measuredRollingRate, 2.5, 1e-9);
  BOOST_CHECK_CLOSE(result.measuredSteeringRate, -0.75, 1e-9);
}

BOOST_AUTO_TEST_CASE(DifferentialWheelHasEmptySteeringSelector)
{
  auto module = loadModule("rolling_diff");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, leftDescription());
  const auto & result = geometry.update(robot, Eigen::Vector3d::UnitZ());

  BOOST_CHECK_EQUAL(geometry.kinematics().steeringSelector.size(), 0);
  BOOST_CHECK_EQUAL(result.measuredSteeringRate, 0.0);
}

BOOST_AUTO_TEST_CASE(RejectsSteeringJointEqualToDriveJoint)
{
  auto module = loadModule("rolling_4s");
  auto robots = mc_rbdyn::loadRobot(*module);
  auto & robot = robots->robot();

  auto description = frontLeftDescription();
  description.steeringJoint = description.driveJoint;
  BOOST_CHECK_THROW(mc_rbdyn::RollingContactRobotGeometry(robot, description), std::invalid_argument);
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
