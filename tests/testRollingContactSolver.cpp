#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/Robots.h>

#include <mc_solver/EqualityConstraint.h>
#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/RollingContactConstraint.h>
#include <mc_solver/RollingContactDynamicsConstraint.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_solver/TVMQPSolver.h>

#include <mc_tvm/RollingContactFunction.h>
#include <mc_tvm/Robot.h>

#include <mc_tasks/PostureTask.h>
#include <mc_tasks/OrientationTask.h>
#include <mc_tasks/PositionTask.h>

#include <boost/test/unit_test.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <RBDyn/MultiBodyConfig.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "utils.h"

namespace
{

class TargetAccelerationTask : public tasks::qp::Task
{
public:
  TargetAccelerationTask(const rbd::MultiBody & mb, int robotIndex)
  : tasks::qp::Task(1.0), robotIndex_(robotIndex), Q_(Eigen::MatrixXd::Identity(mb.nrDof(), mb.nrDof())),
    C_(Eigen::VectorXd::Zero(mb.nrDof()))
  {
  }

  void target(const Eigen::VectorXd & target) { C_ = -target; }
  std::pair<int, int> begin() const override { return {alphaDBegin_, alphaDBegin_}; }
  void updateNrVars(const std::vector<rbd::MultiBody> &, const tasks::qp::SolverData & data) override
  {
    alphaDBegin_ = data.alphaDBegin(robotIndex_);
  }
  void update(const std::vector<rbd::MultiBody> &,
              const std::vector<rbd::MultiBodyConfig> &,
              const tasks::qp::SolverData &) override
  {
  }
  const Eigen::MatrixXd & Q() const override { return Q_; }
  const Eigen::VectorXd & C() const override { return C_; }

private:
  int robotIndex_;
  int alphaDBegin_ = 0;
  Eigen::MatrixXd Q_;
  Eigen::VectorXd C_;
};

class FixedAccelerationRow : public mc_solver::EqualityConstraintRobot
{
public:
  FixedAccelerationRow(unsigned int robotIndex, Eigen::RowVectorXd row)
  : mc_solver::EqualityConstraintRobot(robotIndex), A_(std::move(row)), b_(Eigen::VectorXd::Zero(1))
  {
  }

  const Eigen::MatrixXd & A() const override { return A_; }
  void compute() override {}
  int maxEq() const override { return 1; }
  std::string nameEq() const override { return "FixedAccelerationRow"; }
  const Eigen::VectorXd & bEq() const override { return b_; }

private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

mc_rbdyn::RobotsPtr loadRollingRobots()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobots({four, differential});
}

mc_rbdyn::RobotsPtr loadDifferentialRobot()
{
  configureRobotLoader();
  auto differential = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_diff"));
  return mc_rbdyn::loadRobot(*differential);
}

mc_rbdyn::RobotsPtr loadFourSteeringRobot()
{
  configureRobotLoader();
  auto four = mc_rbdyn::RobotLoader::get_robot_module(
      "RollingContact", std::string(ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH), std::string("rolling_4s"));
  return mc_rbdyn::loadRobot(*four);
}

std::vector<mc_rbdyn::RollingContactDescription> differentialWheels()
{
  std::vector<mc_rbdyn::RollingContactDescription> wheels(2);
  wheels[0].name = "left";
  wheels[0].carrierFrame = "left_carrier";
  wheels[0].wheelBody = "left_wheel";
  wheels[0].driveJoint = "left_drive";
  wheels[0].radius = 0.2;
  wheels[0].width = 0.08;
  wheels[1] = wheels[0];
  wheels[1].name = "right";
  wheels[1].carrierFrame = "right_carrier";
  wheels[1].wheelBody = "right_wheel";
  wheels[1].driveJoint = "right_drive";
  return wheels;
}

std::vector<mc_rbdyn::RollingContactDescription> fourSteeringWheels()
{
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  for(const std::string corner : {"front_left", "front_right", "rear_left", "rear_right"})
  {
    mc_rbdyn::RollingContactDescription wheel;
    wheel.name = corner;
    wheel.carrierFrame = corner + "_carrier";
    wheel.wheelBody = corner + "_wheel";
    wheel.driveJoint = corner + "_drive";
    wheel.steeringJoint = corner + "_steer";
    wheel.radius = 0.2;
    wheel.width = 0.08;
    wheels.push_back(std::move(wheel));
  }
  return wheels;
}

mc_rtc::Configuration rollingConfiguration(const std::string & type,
                                            const std::vector<mc_rbdyn::RollingContactDescription> & wheels)
{
  mc_rtc::Configuration config;
  config.add("type", type);
  config.add("robot", "rolling_diff");
  auto configuredWheels = config.array("wheels");
  for(const auto & wheel : wheels)
  {
    mc_rtc::Configuration configuredWheel;
    configuredWheel.add("name", wheel.name);
    configuredWheel.add("carrierFrame", wheel.carrierFrame);
    configuredWheel.add("forceBody", wheel.wheelBody);
    configuredWheel.add("driveJoint", wheel.driveJoint);
    configuredWheel.add("steeringJoint", wheel.steeringJoint);
    configuredWheel.add("radius", wheel.radius);
    configuredWheel.add("width", wheel.width);
    configuredWheel.add("friction", wheel.friction);
    configuredWheel.add("spinSign", wheel.spinSign);
    configuredWheel.add("mode", mc_rbdyn::to_string(wheel.mode));
    configuredWheel.add("activation", wheel.activation);
    configuredWheels.push(configuredWheel);
  }
  return config;
}

Eigen::VectorXd compatibleTarget(const mc_rbdyn::Robot & robot,
                                 const Eigen::MatrixXd & A,
                                 double leftAcceleration,
                                 double rightAcceleration)
{
  Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
  const int leftDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("left_drive")));
  const int rightDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("right_drive")));
  target(leftDof) = leftAcceleration;
  target(rightDof) = rightAcceleration;
  target.head(6) = A.leftCols(6).completeOrthogonalDecomposition().solve(-A.rightCols(2) * target.tail(2));
  return target;
}

Eigen::VectorXd ackermannTarget(mc_rbdyn::Robot & robot, double linear, double yaw)
{
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3},
                                                  Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3},
                                                  Eigen::Vector2d{-0.45, -0.3}};
  Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
  target(2) = yaw;
  target(3) = linear;
  for(size_t i = 0; i < names.size(); ++i)
  {
    const double x = linear - yaw * offsets[i].y();
    const double y = yaw * offsets[i].x();
    const auto steer = robot.jointIndexByName(names[i] + "_steer");
    const auto drive = robot.jointIndexByName(names[i] + "_drive");
    robot.mbc().q[steer][0] = std::atan2(y, x);
    target(robot.mb().jointPosInDof(static_cast<int>(drive))) = std::hypot(x, y) / 0.2;
  }
  robot.forwardKinematics();
  robot.forwardVelocity();
  return target;
}

bool solve(mc_solver::TasksQPSolver & solver, mc_solver::RollingContactConstraint & rolling)
{
  rolling.update(solver);
  return solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs());
}

/** Row index of a soft row, by label. Fails the test if the row is absent. */
Eigen::Index softRowIndex(const mc_solver::RollingContactConstraint & rolling, const std::string & label)
{
  const auto & labels = rolling.softRowLabels();
  const auto found = std::find(labels.begin(), labels.end(), label);
  BOOST_REQUIRE_MESSAGE(found != labels.end(), "No soft rolling row labelled " << label);
  return std::distance(labels.begin(), found);
}

/** The TVM function must mirror the soft block exactly, at the current row count. */
void checkTVMSoftFunctionMatchesRows(const mc_solver::RollingContactConstraint & rolling)
{
  const auto * soft = rolling.tvmSoftFunction();
  BOOST_REQUIRE(soft != nullptr);
  BOOST_CHECK_EQUAL(static_cast<Eigen::Index>(soft->size()), rolling.softMatrix().rows());
  BOOST_CHECK_SMALL((soft->matrix() - rolling.softMatrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((soft->rhs() - rolling.softRhs()).norm(), 1e-12);
}

Eigen::VectorXd quadraticOracle(const Eigen::VectorXd & target,
                                const Eigen::MatrixXd & softA,
                                double weight,
                                const Eigen::MatrixXd & equalityA,
                                const Eigen::VectorXd & equalityB)
{
  const Eigen::Index n = target.size();
  const Eigen::Index m = equalityA.rows();
  Eigen::MatrixXd kkt = Eigen::MatrixXd::Zero(n + m, n + m);
  kkt.topLeftCorner(n, n) = Eigen::MatrixXd::Identity(n, n) + weight * softA.transpose() * softA;
  kkt.topRightCorner(n, m) = equalityA.transpose();
  kkt.bottomLeftCorner(m, n) = equalityA;
  Eigen::VectorXd rhs(n + m);
  rhs.head(n) = target;
  rhs.tail(m) = equalityB;
  return kkt.completeOrthogonalDecomposition().solve(rhs).head(n);
}

struct BackendParitySnapshot
{
  Eigen::MatrixXd hardMatrix;
  Eigen::VectorXd hardRhs;
  Eigen::VectorXd acceleration;
  Eigen::VectorXd torque;
  Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
  double kinematicsResidual = 0.0;
  double dynamicsResidual = 0.0;
};

template<typename SolverT>
BackendParitySnapshot backendParitySnapshot()
{
  auto robots = loadDifferentialRobot();
  SolverT solver(robots, 0.005);
  const auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  mc_tasks::PositionTask position("chassis", solver.robots(), 0, 20.0, 2000.0);
  position.position(Eigen::Vector3d{0.0, 0.0, 0.2});
  mc_tasks::OrientationTask orientation("chassis", solver.robots(), 0, 20.0, 500.0);

  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addTask(&posture);
  solver.addTask(&position);
  solver.addTask(&orientation);
  if(!solver.run()) { throw std::runtime_error("Rolling-contact backend parity solve failed"); }

  BackendParitySnapshot snapshot;
  snapshot.hardMatrix = rolling.hardMatrix();
  snapshot.hardRhs = rolling.hardRhs();
  if constexpr(std::is_same_v<SolverT, mc_solver::TasksQPSolver>)
  {
    snapshot.acceleration = solver.solver().alphaDVec(0);
    const Eigen::VectorXd & lambda = solver.solver().lambdaVec();
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      const auto forces = dynamics.endpointForces(wheels[i].name, lambda.segment<8>(static_cast<Eigen::Index>(8 * i)));
      const auto & geometry = dynamics.geometryResult(wheels[i].name);
      snapshot.wrench.head<3>() += geometry.lineStart.cross(forces[0]) + geometry.lineEnd.cross(forces[1]);
      snapshot.wrench.tail<3>() += forces[0] + forces[1];
    }
    const Eigen::VectorXd dynamicsValue = dynamics.motionConstr().matrix() * solver.result();
    constexpr Eigen::Index floatingBaseDof = 6;
    snapshot.dynamicsResidual =
        (dynamicsValue.head(floatingBaseDof) - dynamics.motionConstr().LowerGenInEq().head(floatingBaseDof))
            .template lpNorm<Eigen::Infinity>();
  }
  else
  {
    snapshot.acceleration = solver.robot(0).tvmRobot().alphaD()->value();
    for(const auto & wheel : wheels)
    {
      const auto forces = dynamics.endpointForces(wheel.name);
      const auto & geometry = dynamics.geometryResult(wheel.name);
      snapshot.wrench.head<3>() += geometry.lineStart.cross(forces[0]) + geometry.lineEnd.cross(forces[1]);
      snapshot.wrench.tail<3>() += forces[0] + forces[1];
    }
    dynamics.dynamicFunction().updateValue();
    snapshot.dynamicsResidual = dynamics.dynamicFunction().value().lpNorm<Eigen::Infinity>();
  }
  snapshot.torque.resize(solver.robot(0).mb().nrDof());
  rbd::paramToVector(solver.robot(0).mbc().jointTorque, snapshot.torque);
  snapshot.kinematicsResidual =
      (snapshot.hardMatrix * snapshot.acceleration - snapshot.hardRhs).lpNorm<Eigen::Infinity>();

  solver.removeTask(&orientation);
  solver.removeTask(&position);
  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  return snapshot;
}

/** Carrier offsets of the rolling_4s corners, in the order fourSteeringWheels() uses. */
const std::array<Eigen::Vector2d, 4> & fourSteeringOffsets()
{
  static const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3}, Eigen::Vector2d{0.45, -0.3},
                                                         Eigen::Vector2d{-0.45, 0.3}, Eigen::Vector2d{-0.45, -0.3}};
  return offsets;
}

/** Steering angles that put all four wheels on the common ICR of a planar twist. */
std::array<double, 4> icrSteeringAngles(double linear, double lateral, double yaw)
{
  std::array<double, 4> angles{};
  for(size_t i = 0; i < angles.size(); ++i)
  {
    angles[i] = std::atan2(lateral + yaw * fourSteeringOffsets()[i].x(), linear - yaw * fourSteeringOffsets()[i].y());
  }
  return angles;
}

/** The 4x3 planar lateral block a_i^T = [-sin d_i, cos d_i, x_i cos d_i + y_i sin d_i].
 *
 * Built through mc_rbdyn::steeringRollingMatrix() rather than restated here, so
 * the rank findings below are about the shipped planar rows.
 */
Eigen::MatrixXd planarLateralBlock(const std::array<double, 4> & steeringAngles)
{
  std::vector<mc_rbdyn::PlanarWheel> planar(4);
  for(size_t i = 0; i < planar.size(); ++i)
  {
    planar[i].offset = fourSteeringOffsets()[i];
    planar[i].steeringAngle = steeringAngles[i];
    planar[i].radius = 0.2;
  }
  const auto rows = mc_rbdyn::steeringRollingMatrix(planar, Eigen::Vector3d::Zero());
  Eigen::MatrixXd block(4, 3);
  for(Eigen::Index i = 0; i < 4; ++i) { block.row(i) = rows.matrix.block<1, 3>(2 * i + 1, 0); }
  return block;
}

/** Steering angles sharing no common ICR, reused by every lateral-row test. */
const std::array<double, 4> & uncoordinatedSteeringAngles()
{
  static const std::array<double, 4> angles = {0.30, -0.10, 0.20, -0.30};
  return angles;
}

/** Put the four steering hinges at @p angles and refresh the kinematic state. */
void setSteeringAngles(mc_rbdyn::Robot & robot, const std::array<double, 4> & angles)
{
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
  for(size_t i = 0; i < names.size(); ++i)
  {
    robot.mbc().q[robot.jointIndexByName(names[i] + "_steer")][0] = angles[i];
  }
  robot.forwardKinematics();
  robot.forwardVelocity();
}

/** A four-wheel state whose stabilised lateral right-hand side is NOT in range(A).
 *
 * Uncoordinated angles alone are not enough: the stabilisation term -Kp * A * alpha
 * lies in range(A) by construction, and so does the Jdot part of the bias while the
 * hinges are still. The out-of-range component comes from the ldot . v term, i.e.
 * from the four independent *measured steering rates* - exactly the quantity the
 * report says no controller can coordinate. A moving chassis is required too,
 * since ldot . v vanishes when the carriers are at rest.
 */
void setIncompatibleFourWheelState(mc_rbdyn::Robot & robot)
{
  setSteeringAngles(robot, uncoordinatedSteeringAngles());
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
  const std::array<double, 4> steeringRates = {0.5, -0.3, 0.7, -0.9};
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
  velocity(2) = 0.2;
  velocity(3) = 0.35;
  velocity(4) = 0.12;
  for(size_t i = 0; i < names.size(); ++i)
  {
    velocity(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(names[i] + "_steer")))) =
        steeringRates[i];
  }
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.forwardVelocity();
}

/** The unscaled lateral rows and their right-hand side, straight from the geometry. */
std::pair<Eigen::MatrixXd, Eigen::VectorXd> unscaledLateralRows(const mc_solver::RollingContactConstraint & rolling,
                                                                Eigen::Index nrDof)
{
  const auto & results = rolling.geometryResults();
  Eigen::MatrixXd A(static_cast<Eigen::Index>(results.size()), nrDof);
  Eigen::VectorXd c(static_cast<Eigen::Index>(results.size()));
  for(size_t i = 0; i < results.size(); ++i)
  {
    A.row(static_cast<Eigen::Index>(i)) = results[i].rollingMatrix.row(1);
    c(static_cast<Eigen::Index>(i)) = results[i].rhs(1);
  }
  return {A, c};
}

/** Rows of @p matrix whose label contains @p needle, stacked in label order. */
Eigen::MatrixXd labelledRows(const Eigen::MatrixXd & matrix,
                             const std::vector<std::string> & labels,
                             const std::string & needle)
{
  std::vector<Eigen::Index> selected;
  for(size_t i = 0; i < labels.size(); ++i)
  {
    if(labels[i].find(needle) != std::string::npos) { selected.push_back(static_cast<Eigen::Index>(i)); }
  }
  Eigen::MatrixXd out(static_cast<Eigen::Index>(selected.size()), matrix.cols());
  for(size_t i = 0; i < selected.size(); ++i) { out.row(static_cast<Eigen::Index>(i)) = matrix.row(selected[i]); }
  return out;
}

Eigen::VectorXd singularValues(const Eigen::MatrixXd & matrix)
{
  return Eigen::JacobiSVD<Eigen::MatrixXd>(matrix).singularValues();
}

/** Rank at an explicit absolute threshold, reported by both available oracles. */
Eigen::Index rankAt(const Eigen::MatrixXd & matrix, double threshold)
{
  Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
  lu.setThreshold(threshold);
  const Eigen::VectorXd sv = singularValues(matrix);
  const Eigen::Index svdRank = (sv.array() > threshold).count();
  BOOST_CHECK_EQUAL(lu.rank(), svdRank);
  return lu.rank();
}

std::string formatSingularValues(const Eigen::MatrixXd & matrix)
{
  std::ostringstream out;
  const Eigen::VectorXd sv = singularValues(matrix);
  for(Eigen::Index i = 0; i < sv.size(); ++i) { out << (i == 0 ? "" : ", ") << sv(i); }
  return out.str();
}

/** True when the four wheel axes admit a common instantaneous centre of rotation.
 *
 * Computed from the geometry alone, independently of the lateral block, so that
 * GEO-09's equivalence is an observation and not a restatement. The ICR must lie
 * on the line through rho_i orthogonal to the wheel heading, i.e.
 * c . (cos d_i, sin d_i) = rho_i . (cos d_i, sin d_i); a common ICR at infinity
 * is the all-parallel case, which admits a pure translation instead.
 */
bool hasCommonICR(const std::array<double, 4> & steeringAngles)
{
  Eigen::Matrix<double, 4, 2> lines;
  Eigen::Vector4d offsets;
  for(size_t i = 0; i < steeringAngles.size(); ++i)
  {
    const Eigen::Vector2d heading{std::cos(steeringAngles[i]), std::sin(steeringAngles[i])};
    lines.row(static_cast<Eigen::Index>(i)) = heading.transpose();
    offsets(static_cast<Eigen::Index>(i)) = fourSteeringOffsets()[i].dot(heading);
  }
  const Eigen::Vector2d centre = lines.colPivHouseholderQr().solve(offsets);
  if((lines * centre - offsets).lpNorm<Eigen::Infinity>() < 1e-9) { return true; }
  // All headings parallel: the centre is at infinity and pure translation is admissible.
  const Eigen::Vector2d reference{std::cos(steeringAngles[0]), std::sin(steeringAngles[0])};
  for(size_t i = 1; i < steeringAngles.size(); ++i)
  {
    const Eigen::Vector2d heading{std::cos(steeringAngles[i]), std::sin(steeringAngles[i])};
    if(std::abs(reference.x() * heading.y() - reference.y() * heading.x()) > 1e-9) { return false; }
  }
  return true;
}

template<typename DerivedA, typename DerivedB>
void checkScaledParity(const Eigen::MatrixBase<DerivedA> & tasksValue,
                       const Eigen::MatrixBase<DerivedB> & tvmValue,
                       double tolerance,
                       const std::string & quantity)
{
  BOOST_REQUIRE_EQUAL(tasksValue.rows(), tvmValue.rows());
  BOOST_REQUIRE_EQUAL(tasksValue.cols(), tvmValue.cols());
  const double scale = 1.0 + std::max(tasksValue.norm(), tvmValue.norm());
  const double error = (tasksValue - tvmValue).norm();
  BOOST_TEST_MESSAGE(quantity << " Tasks/TVM error=" << error << ", scaled tolerance=" << tolerance * scale);
  BOOST_CHECK_LE(error, tolerance * scale);
}

} // namespace

BOOST_AUTO_TEST_CASE(RollingTasksMatrixPlacementAndHardCases)
{
  auto robots = loadRollingRobots();
  constexpr unsigned int robotIndex = 1;
  mc_solver::TasksQPSolver solver(robots, 0.005);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), robotIndex, differentialWheels(), options);
  TargetAccelerationTask unusedRobotTask(solver.robots().robot(0).mb(), 0);
  TargetAccelerationTask targetTask(solver.robots().robot(robotIndex).mb(), static_cast<int>(robotIndex));
  solver.addTask(&unusedRobotTask);
  solver.addTask(&targetTask);
  solver.addConstraintSet(rolling);

  BOOST_REQUIRE_EQUAL(rolling.hardMatrix().rows(), 5);
  BOOST_CHECK_EQUAL(rolling.hardRowLabels()[0], "left/longitudinal");
  BOOST_CHECK_EQUAL(rolling.hardRowLabels()[1], "right/longitudinal");
  BOOST_CHECK_EQUAL(rolling.hardRowLabels()[2], "left/lateral");
  const auto & data = solver.data();
  BOOST_TEST_MESSAGE("Tasks nrVars=" << data.nrVars() << ", alphaD begins=[" << data.alphaDBegin(0) << ","
                                     << data.alphaDBegin(1) << "], lambdaBegin=" << data.lambdaBegin());
  BOOST_CHECK_EQUAL(data.nrVars(), 22);
  BOOST_CHECK_EQUAL(data.alphaDBegin(0), 0);
  BOOST_CHECK_EQUAL(data.alphaDBegin(1), 14);
  BOOST_CHECK_EQUAL(data.lambdaBegin(), 22);

  const auto & robot = solver.robot(robotIndex);
  for(const auto & accelerations : std::vector<std::pair<double, double>>{{2.0, 2.0}, {-1.0, 1.0}, {1.0, 2.0}})
  {
    const Eigen::VectorXd target =
        compatibleTarget(robot, rolling.hardMatrix(), accelerations.first, accelerations.second);
    BOOST_REQUIRE_SMALL((rolling.hardMatrix() * target - rolling.hardRhs()).norm(), 1e-10);
    targetTask.target(target);
    BOOST_REQUIRE(solve(solver, rolling));
    const Eigen::VectorXd solution = solver.solver().alphaDVec(static_cast<int>(robotIndex));
    BOOST_CHECK_SMALL((rolling.hardMatrix() * solution - rolling.hardRhs()).lpNorm<Eigen::Infinity>(), 1e-8);
    BOOST_CHECK_SMALL((solution - target).norm(), 1e-7);
  }

  const Eigen::MatrixXd & fullA = rolling.tasksFullHardMatrix();
  BOOST_REQUIRE_EQUAL(fullA.cols(), solver.data().nrVars());
  BOOST_CHECK_SMALL((fullA.middleCols(rolling.tasksAlphaDBegin(), robot.mb().nrDof()) - rolling.hardMatrix()).norm(),
                    1e-12);
  Eigen::MatrixXd structuralZeros = fullA;
  structuralZeros.middleCols(rolling.tasksAlphaDBegin(), robot.mb().nrDof()).setZero();
  BOOST_CHECK_SMALL(structuralZeros.norm(), 1e-12);
}

BOOST_AUTO_TEST_CASE(RollingTasksSoftConflictWeightAndLifecycle)
{
  auto robots = loadRollingRobots();
  constexpr unsigned int robotIndex = 1;
  mc_solver::TasksQPSolver solver(robots, 0.005);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.longitudinal = mc_solver::RollingContactLongitudinal::Soft;
  options.differentialPlanar = true;
  options.constrainNormal = false;
  options.rollingWeight = 1.0;
  mc_solver::RollingContactConstraint rolling(solver.robots(), robotIndex, differentialWheels(), options);
  TargetAccelerationTask unusedRobotTask(solver.robots().robot(0).mb(), 0);
  TargetAccelerationTask targetTask(solver.robots().robot(robotIndex).mb(), static_cast<int>(robotIndex));
  Eigen::VectorXd target = Eigen::VectorXd::Zero(solver.robots().robot(robotIndex).mb().nrDof());
  target.tail(2).setOnes();
  targetTask.target(target);
  solver.addTask(&unusedRobotTask);
  solver.addTask(&targetTask);
  solver.updateNrVars();

  Eigen::RowVectorXd fixedCarrierRow = rolling.softMatrix().row(0);
  const auto & robot = solver.robot(robotIndex);
  const int leftDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("left_drive")));
  fixedCarrierRow(leftDof) = 0.0;
  FixedAccelerationRow fixed(robotIndex, fixedCarrierRow);
  solver.addConstraint(&fixed);

  const int baselineTasks = solver.solver().nrTasks();
  const int baselineEqualities = solver.solver().nrEqualityConstraints();
  solver.addConstraintSet(rolling);
  solver.addConstraintSet(rolling);
  BOOST_CHECK_EQUAL(solver.solver().nrTasks(), baselineTasks + 1);
  BOOST_CHECK_EQUAL(solver.solver().nrEqualityConstraints(), baselineEqualities + 1);

  double previousResidual = std::numeric_limits<double>::infinity();
  for(const double weight : {1.0, 10.0, 100.0, 1000.0})
  {
    rolling.rollingWeight(weight);
    BOOST_REQUIRE(solve(solver, rolling));
    const Eigen::VectorXd solution = solver.solver().alphaDVec(static_cast<int>(robotIndex));
    const double residual = (rolling.softMatrix() * solution - rolling.softRhs()).norm();
    BOOST_CHECK_LE(residual, previousResidual + 1e-10);
    previousResidual = residual;
    BOOST_CHECK_SMALL((rolling.hardMatrix() * solution - rolling.hardRhs()).norm(), 1e-8);
    BOOST_CHECK_SMALL(fixedCarrierRow.dot(solution), 1e-8);

    Eigen::MatrixXd equalityA(rolling.hardMatrix().rows() + 1, solution.size());
    equalityA.topRows(rolling.hardMatrix().rows()) = rolling.hardMatrix();
    equalityA.bottomRows(1) = fixedCarrierRow;
    Eigen::VectorXd equalityB(rolling.hardRhs().size() + 1);
    equalityB.head(rolling.hardRhs().size()) = rolling.hardRhs();
    equalityB.tail(1).setZero();
    const Eigen::VectorXd oracle = quadraticOracle(target, rolling.softMatrix(), weight, equalityA, equalityB);
    BOOST_CHECK_SMALL((solution - oracle).norm(), 2e-7);
  }

  solver.removeConstraintSet(rolling);
  BOOST_CHECK_EQUAL(solver.solver().nrTasks(), baselineTasks);
  BOOST_CHECK_EQUAL(solver.solver().nrEqualityConstraints(), baselineEqualities);
  solver.removeConstraintSet(rolling);
  solver.addConstraintSet(rolling);
  BOOST_REQUIRE(solve(solver, rolling));
  solver.removeConstraintSet(rolling);
}

BOOST_AUTO_TEST_CASE(RollingTasksFourSteeringAckermannTarget)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  const Eigen::VectorXd target = ackermannTarget(robot, 0.15, 0.05);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.softLateralRows = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
  TargetAccelerationTask targetTask(robot.mb(), 0);
  targetTask.target(target);
  solver.addTask(&targetTask);
  solver.addConstraintSet(rolling);
  rolling.update(solver);
  BOOST_REQUIRE_SMALL((rolling.hardMatrix() * target - rolling.hardRhs()).norm(), 1e-10);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd solution = solver.solver().alphaDVec(0);
  BOOST_TEST_MESSAGE("Four-steering target error without dynamics: " << (solution - target).norm());
  BOOST_CHECK_SMALL((solution - target).norm(), 1e-7);
  solver.removeConstraintSet(rolling);
  solver.removeTask(&targetTask);
}

BOOST_AUTO_TEST_CASE(RollingTasksFourSteeringAckermannTargetWithDynamics)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  const Eigen::VectorXd target = ackermannTarget(robot, 0.15, 0.05);
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.softLateralRows = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  TargetAccelerationTask targetTask(robot.mb(), 0);
  targetTask.target(target);
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  rolling.update(solver);
  BOOST_REQUIRE_SMALL((rolling.hardMatrix() * target - rolling.hardRhs()).norm(), 1e-10);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd solution = solver.solver().alphaDVec(0);
  BOOST_TEST_MESSAGE("Four-steering target error with dynamics: " << (solution - target).norm());
  BOOST_CHECK_SMALL((rolling.hardMatrix() * solution - rolling.hardRhs()).lpNorm<Eigen::Infinity>(), 1e-8);
  // dynamics is built with the default generatorRegularization, which is
  // 0.0 (see RollingContactDynamicsConstraint.h): the Tasks backend floors
  // the Hessian diagonal at 1e-4 unconditionally, so no Tikhonov term is
  // needed here for conditioning, and this reproduces the
  // pre-generatorRegularization baseline of ~1.7e-4 rather than the ~6.7e-4
  // measured with the term enabled at 2e-4 (see
  // ...GeneratorRegularizationFRI01BiasAtEnabledAndTenX). 5e-4 keeps a
  // margin over the measured value while still catching a regression that
  // meaningfully grows it further.
  BOOST_CHECK_SMALL((solution - target).norm(), 5e-4);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.removeTask(&targetTask);
}

BOOST_AUTO_TEST_CASE(RollingTasksFourSteeringAckermannTrajectoryTasks)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  constexpr double linear = 0.15;
  constexpr double yaw = 0.05;
  constexpr double duration = 0.5;
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 5.0;
  options.softLateralRows = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 500.0);
  std::map<std::string, std::vector<double>> postureTarget;
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3},
                                                  Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3},
                                                  Eigen::Vector2d{-0.45, -0.3}};
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const double x = linear - yaw * offsets[i].y();
    const double y = yaw * offsets[i].x();
    postureTarget[wheels[i].driveJoint] = {duration * std::hypot(x, y) / wheels[i].radius};
    postureTarget[wheels[i].steeringJoint] = {std::atan2(y, x)};
  }
  posture.target(postureTarget);
  mc_tasks::PositionTask position("chassis", solver.robots(), 0, 20.0, 2000.0);
  const double targetYaw = duration * yaw;
  position.position(Eigen::Vector3d{linear * std::sin(targetYaw) / yaw,
                                    linear * (1.0 - std::cos(targetYaw)) / yaw, 0.2});
  mc_tasks::OrientationTask orientation("chassis", solver.robots(), 0, 20.0, 500.0);
  Eigen::Matrix3d targetRotation;
  targetRotation << std::cos(targetYaw), std::sin(targetYaw), 0.0, -std::sin(targetYaw), std::cos(targetYaw), 0.0,
      0.0, 0.0, 1.0;
  orientation.orientation(targetRotation);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addTask(&posture);
  solver.addTask(&orientation);
  solver.addTask(&position);
  (void)ackermannTarget(robot, linear, yaw);
  BOOST_REQUIRE(solver.run());
  double minimumDriveAcceleration = std::numeric_limits<double>::infinity();
  for(const auto & wheel : wheels)
  {
    const auto joint = robot.jointIndexByName(wheel.driveJoint);
    minimumDriveAcceleration = std::min(minimumDriveAcceleration, robot.mbc().alphaD[joint][0]);
  }
  BOOST_TEST_MESSAGE("Ackermann trajectory minimum drive acceleration: " << minimumDriveAcceleration);
  BOOST_CHECK_GT(minimumDriveAcceleration, 0.1);
  solver.removeTask(&position);
  solver.removeTask(&orientation);
  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(RollingConstraintRejectsInvalidConfiguration)
{
  auto robots = loadRollingRobots();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  wheels[1].name = wheels[0].name;
  BOOST_CHECK_THROW(mc_solver::RollingContactConstraint(solver.robots(), 1, wheels), std::invalid_argument);
  mc_solver::RollingContactConstraintOptions options;
  options.differentialPlanar = true;
  wheels.resize(1);
  BOOST_CHECK_THROW(mc_solver::RollingContactConstraint(solver.robots(), 1, wheels, options), std::invalid_argument);

  for(const std::string field : {"carrier", "body", "drive"})
  {
    auto invalidWheels = differentialWheels();
    if(field == "carrier") { invalidWheels[0].carrierFrame = "missing_carrier"; }
    else if(field == "body") { invalidWheels[0].wheelBody = "missing_body"; }
    else { invalidWheels[0].driveJoint = "missing_drive"; }
    BOOST_CHECK_THROW(mc_solver::RollingContactConstraint(solver.robots(), 1, invalidWheels), std::exception);
    BOOST_CHECK_THROW(mc_solver::RollingContactDynamicsConstraint(solver.robots(), 1, solver.dt(), invalidWheels),
                      std::exception);
  }
}

BOOST_AUTO_TEST_CASE(RollingRateReferencesRoundTripAndValidate)
{
  auto robots = loadFourSteeringRobot();
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.trackRotatingRates = true;
  BOOST_CHECK_EQUAL(options.rollingRateWeight, 200.0);
  BOOST_CHECK_EQUAL(options.steeringRateWeight, 200.0);

  mc_solver::RollingContactConstraint rolling(*robots, 0, wheels, options);
  rolling.rotatingRateReference("front_left", 3.0, -0.5);
  BOOST_CHECK_EQUAL(rolling.rollingRateReference("front_left"), 3.0);
  BOOST_CHECK_EQUAL(rolling.steeringRateReference("front_left"), -0.5);

  BOOST_CHECK_THROW(rolling.rotatingRateReference("nope", 0.0, 0.0), std::out_of_range);
  BOOST_CHECK_THROW(rolling.rotatingRateReference("front_left", std::numeric_limits<double>::quiet_NaN(), 0.0),
                    std::invalid_argument);

  mc_solver::RollingContactConstraintOptions negativeRolling = options;
  negativeRolling.rollingRateWeight = -1.0;
  BOOST_CHECK_THROW(negativeRolling.validate(4), std::invalid_argument);

  mc_solver::RollingContactConstraintOptions negativeSteering = options;
  negativeSteering.steeringRateWeight = -1.0;
  BOOST_CHECK_THROW(negativeSteering.validate(4), std::invalid_argument);

  mc_solver::RollingContactConstraintOptions nanRolling = options;
  nanRolling.rollingRateWeight = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK_THROW(nanRolling.validate(4), std::invalid_argument);

  mc_solver::RollingContactConstraintOptions nanSteering = options;
  nanSteering.steeringRateWeight = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK_THROW(nanSteering.validate(4), std::invalid_argument);

  mc_solver::RollingContactConstraintOptions infRolling = options;
  infRolling.rollingRateWeight = std::numeric_limits<double>::infinity();
  BOOST_CHECK_THROW(infRolling.validate(4), std::invalid_argument);

  mc_solver::RollingContactConstraintOptions infSteering = options;
  infSteering.steeringRateWeight = std::numeric_limits<double>::infinity();
  BOOST_CHECK_THROW(infSteering.validate(4), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsMatchTheAffinePrediction)
{
  constexpr double dt = 0.005;
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, dt);
  auto & robot = solver.robot(0);
  const auto drive = robot.jointIndexByName("front_left_drive");
  const auto steer = robot.jointIndexByName("front_left_steer");
  robot.mbc().alpha[drive][0] = 1.5;
  robot.mbc().alpha[steer][0] = 0.25;
  robot.forwardKinematics();
  robot.forwardVelocity();
  const Eigen::Index driveDof = robot.mb().jointPosInDof(static_cast<int>(drive));
  const Eigen::Index steerDof = robot.mb().jointPosInDof(static_cast<int>(steer));

  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.trackRotatingRates = true;
  options.rollingWeight = 1000.0;
  options.rollingRateWeight = 250.0;
  options.steeringRateWeight = 40.0;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
  solver.addConstraintSet(rolling);

  // A soft row scaled by s contributes rollingWeight * s^2 * residual^2, so the
  // per-row weight is realised by s = sqrt(rowWeight / rollingWeight).
  const double rollingScale = std::sqrt(250.0 / 1000.0);
  const double steeringScale = std::sqrt(40.0 / 1000.0);

  // The constructor already filled the rows, but with dt still zero. Pin that
  // state: empty coefficients, and a right-hand side that already carries the
  // residual against the default zero reference.
  const Eigen::Index initialRow = softRowIndex(rolling, "front_left/rolling-rate");
  BOOST_CHECK_SMALL(rolling.softMatrix().row(initialRow).lpNorm<Eigen::Infinity>(), 1e-14);
  BOOST_CHECK_CLOSE(rolling.softRhs()(initialRow), rollingScale * (0.0 - 1.5), 1e-9);

  rolling.rotatingRateReference("front_left", 4.5, -0.75);
  rolling.update(solver);

  const Eigen::Index rollingRow = softRowIndex(rolling, "front_left/rolling-rate");
  const Eigen::Index steeringRow = softRowIndex(rolling, "front_left/steering-rate");
  BOOST_REQUIRE_EQUAL(rolling.softMatrix().rows(), static_cast<Eigen::Index>(rolling.softRowLabels().size()));
  BOOST_CHECK_CLOSE(rolling.softMatrix()(rollingRow, driveDof), rollingScale * dt, 1e-9);
  BOOST_CHECK_SMALL(rolling.softMatrix()(rollingRow, steerDof), 1e-14);
  BOOST_CHECK_CLOSE(rolling.softRhs()(rollingRow), rollingScale * (4.5 - 1.5), 1e-9);
  BOOST_CHECK_CLOSE(rolling.softMatrix()(steeringRow, steerDof), steeringScale * dt, 1e-9);
  BOOST_CHECK_SMALL(rolling.softMatrix()(steeringRow, driveDof), 1e-14);
  BOOST_CHECK_CLOSE(rolling.softRhs()(steeringRow), steeringScale * (-0.75 - 0.25), 1e-9);
  solver.removeConstraintSet(rolling);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsFollowTheBlockWeight)
{
  constexpr double dt = 0.005;
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, dt);
  const auto & robot = solver.robot(0);
  const Eigen::Index steerDof =
      robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.trackRotatingRates = true;
  options.rollingWeight = 1000.0;
  options.steeringRateWeight = 40.0;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
  solver.addConstraintSet(rolling);
  rolling.update(solver);

  // Rate rows are the only rows whose scale depends on the block weight, so a
  // rollingWeight() change must re-derive them or the realised weight drifts.
  auto realisedSteeringRateWeight = [&]()
  {
    const Eigen::Index row = softRowIndex(rolling, "front_left/steering-rate");
    const double scale = rolling.softMatrix()(row, steerDof) / dt;
    return rolling.rollingWeight() * scale * scale;
  };

  const double before = realisedSteeringRateWeight();
  const size_t layoutRevision = rolling.layoutRevision();
  rolling.rollingWeight(4000.0);
  rolling.update(solver);
  const double after = realisedSteeringRateWeight();
  BOOST_TEST_MESSAGE("Realised steering-rate weight before rollingWeight(4000): " << before << ", after: " << after);
  BOOST_CHECK_CLOSE(before, 40.0, 1e-9);
  BOOST_CHECK_CLOSE(after, 40.0, 1e-9);
  // The row set is unchanged, so re-deriving the scales must not churn the layout.
  BOOST_CHECK_EQUAL(rolling.layoutRevision(), layoutRevision);
  BOOST_REQUIRE(solve(solver, rolling));
  solver.removeConstraintSet(rolling);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsAreAbsentWhenDisabledDetachedOrInactive)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  const auto wheels = fourSteeringWheels();
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;

  {
    mc_solver::RollingContactConstraint disabled(solver.robots(), 0, wheels, options);
    // Partial activations demote every row to the soft block, so the absence of
    // rate labels below is a real observation and not an empty-set vacuity.
    for(const auto & wheel : wheels) { disabled.activation(wheel.name, 0.5); }
    solver.addConstraintSet(disabled);
    disabled.update(solver);
    BOOST_REQUIRE_GT(disabled.softRowLabels().size(), 0u);
    for(const auto & label : disabled.softRowLabels())
    {
      BOOST_CHECK(label.find("-rate") == std::string::npos);
    }
    solver.removeConstraintSet(disabled);
  }

  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  rolling.mode("front_left", mc_rbdyn::RollingContactMode::Detached, 0.0);
  solver.addConstraintSet(rolling);
  rolling.update(solver);
  auto hasRow = [&rolling](const std::string & label)
  {
    // Re-reads softRowLabels() on every call: the vector is rebuilt by mode and
    // activation changes, so a bound reference would read as a stale-view bug.
    const auto & labels = rolling.softRowLabels();
    return std::find(labels.begin(), labels.end(), label) != labels.end();
  };
  BOOST_CHECK(!hasRow("front_left/rolling-rate"));
  BOOST_CHECK(!hasRow("front_left/steering-rate"));
  BOOST_CHECK(hasRow("rear_right/rolling-rate"));
  BOOST_CHECK(hasRow("rear_right/steering-rate"));

  // A fully de-activated wheel drops its rate rows even while it stays Rolling:
  // the zero activation would otherwise scale them to nothing anyway.
  rolling.activation("rear_left", 0.0);
  rolling.update(solver);
  BOOST_CHECK(rolling.mode("rear_left") == mc_rbdyn::RollingContactMode::Rolling);
  BOOST_CHECK(!hasRow("rear_left/rolling-rate"));
  BOOST_CHECK(!hasRow("rear_left/steering-rate"));
  BOOST_CHECK(hasRow("rear_right/steering-rate"));
  solver.removeConstraintSet(rolling);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsAreOmittedAtZeroWeight)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.trackRotatingRates = true;

  // A zero weight is the documented way to switch one axis off. It has to drop the
  // row: a row scaled to zero would occupy a slot in softA and a label in
  // softRowLabels() forever, with no runtime setter able to revive it.
  auto rateLabels = [&](double rollingRateWeight, double steeringRateWeight)
  {
    options.rollingRateWeight = rollingRateWeight;
    options.steeringRateWeight = steeringRateWeight;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
    solver.addConstraintSet(rolling);
    rolling.update(solver);
    std::vector<std::string> found;
    for(const auto & label : rolling.softRowLabels())
    {
      if(label.find("-rate") != std::string::npos) { found.push_back(label); }
    }
    // Every emitted soft row must be able to act on the solution.
    for(Eigen::Index row = 0; row < rolling.softMatrix().rows(); ++row)
    {
      BOOST_CHECK_GT(rolling.softMatrix().row(row).lpNorm<Eigen::Infinity>(), 0.0);
    }
    solver.removeConstraintSet(rolling);
    return found;
  };

  const auto withoutRolling = rateLabels(0.0, 200.0);
  BOOST_CHECK_EQUAL(withoutRolling.size(), 4u);
  for(const auto & label : withoutRolling) { BOOST_CHECK(label.find("/steering-rate") != std::string::npos); }

  const auto withoutSteering = rateLabels(200.0, 0.0);
  BOOST_CHECK_EQUAL(withoutSteering.size(), 4u);
  for(const auto & label : withoutSteering) { BOOST_CHECK(label.find("/rolling-rate") != std::string::npos); }

  BOOST_CHECK_EQUAL(rateLabels(0.0, 0.0).size(), 0u);
  BOOST_CHECK_EQUAL(rateLabels(200.0, 200.0).size(), 8u);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsSkipSteeringOnDifferentialWheels)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  const auto wheels = differentialWheels();
  mc_solver::RollingContactConstraintOptions options;
  options.differentialPlanar = true;
  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  solver.addConstraintSet(rolling);
  rolling.update(solver);
  const auto & labels = rolling.softRowLabels();
  for(const auto & wheel : wheels)
  {
    BOOST_CHECK(std::find(labels.begin(), labels.end(), wheel.name + "/rolling-rate") != labels.end());
    BOOST_CHECK(std::find(labels.begin(), labels.end(), wheel.name + "/steering-rate") == labels.end());
  }
  solver.removeConstraintSet(rolling);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsSteerTowardsTheReference)
{
  // Without the rate rows the steering column is identically zero in every
  // rolling row, so the QP has no equation touching the steering acceleration.
  auto steeringAcceleration = [](bool trackRotatingRates)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    auto & robot = solver.robot(0);
    const auto wheels = fourSteeringWheels();
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 0.0;
    options.softLateralRows = true;
    options.trackRotatingRates = trackRotatingRates;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    targetTask.target(Eigen::VectorXd::Zero(robot.mb().nrDof()));
    solver.addTask(&targetTask);
    solver.addConstraintSet(rolling);
    for(const auto & wheel : wheels) { rolling.rotatingRateReference(wheel.name, 0.0, 0.6); }
    BOOST_REQUIRE(solve(solver, rolling));
    const Eigen::VectorXd solution = solver.solver().alphaDVec(0);
    solver.removeConstraintSet(rolling);
    solver.removeTask(&targetTask);
    return solution(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer"))));
  };

  const double tracked = steeringAcceleration(true);
  const double untracked = steeringAcceleration(false);
  BOOST_TEST_MESSAGE("Steering acceleration with rate tracking: " << tracked << ", without: " << untracked);
  // Nothing else in this QP touches the steering acceleration, so with tracking off
  // the identity regularizer drives it to a hard zero -- hence the 1e-12 here
  // against the looser 1e-6 of the TVM case, where a least-squares solve and a
  // dynamics constraint sit in between.
  BOOST_CHECK_SMALL(untracked, 1e-12);
  // The identity regularizer opposes the rate row, so the observed value is about
  // 0.597 rad/s^2 (= 0.6 / 1.005): the threshold keeps a comfortable margin below it.
  BOOST_CHECK_GT(tracked, 0.4);
}

BOOST_AUTO_TEST_CASE(RollingConstraintConfigurationLoaders)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);

  auto rollingConfig = rollingConfiguration("rollingContact", differentialWheels());
  rollingConfig.add("terrainNormal", Eigen::Vector3d{0.1, -0.2, 1.0});
  rollingConfig.add("longitudinal", "soft");
  rollingConfig.add("velocityGain", 17.0);
  rollingConfig.add("rollingWeight", 321.0);
  rollingConfig.add("constrainNormal", false);
  rollingConfig.add("differentialPlanar", true);
  rollingConfig.add("trackRotatingRates", true);
  rollingConfig.add("rollingRateWeight", 42.0);
  rollingConfig.add("steeringRateWeight", 43.0);
  const auto loadedRolling = mc_solver::ConstraintSetLoader::load(solver, rollingConfig);
  const auto rolling = std::dynamic_pointer_cast<mc_solver::RollingContactConstraint>(loadedRolling);
  BOOST_REQUIRE(rolling);
  BOOST_CHECK_EQUAL(rolling->robotIndex(), 0);
  BOOST_REQUIRE_EQUAL(rolling->wheels().size(), 2);
  BOOST_CHECK_EQUAL(rolling->wheels()[0].wheelBody, "left_wheel");
  BOOST_CHECK(rolling->options().longitudinal == mc_solver::RollingContactLongitudinal::Soft);
  BOOST_CHECK_CLOSE(rolling->options().velocityGain, 17.0, 1e-12);
  BOOST_CHECK_CLOSE(rolling->rollingWeight(), 321.0, 1e-12);
  BOOST_CHECK(!rolling->options().constrainNormal);
  BOOST_CHECK(rolling->options().differentialPlanar);
  BOOST_CHECK(rolling->options().trackRotatingRates);
  BOOST_CHECK_CLOSE(rolling->options().rollingRateWeight, 42.0, 1e-12);
  BOOST_CHECK_CLOSE(rolling->options().steeringRateWeight, 43.0, 1e-12);

  auto softLateralConfig = rollingConfiguration("rollingContact", differentialWheels());
  softLateralConfig.add("softLateralRows", true);
  softLateralConfig.add("lateralSlackWeight", 12345.0);
  const auto loadedSoft = std::dynamic_pointer_cast<mc_solver::RollingContactConstraint>(
      mc_solver::ConstraintSetLoader::load(solver, softLateralConfig));
  BOOST_REQUIRE(loadedSoft);
  BOOST_CHECK(loadedSoft->options().softLateralRows);
  BOOST_CHECK_CLOSE(loadedSoft->options().lateralSlackWeight, 12345.0, 1e-12);

  // steeringPlanar/steeringPlanarWheels were retired with the hard lateral rows.
  // Ignoring them would silently restore four hard rows on a four-wheel chassis,
  // so a stale configuration is rejected instead.
  for(const std::string removed : {"steeringPlanar", "steeringPlanarWheels"})
  {
    auto staleConfig = rollingConfiguration("rollingContact", differentialWheels());
    if(removed == "steeringPlanar") { staleConfig.add(removed, true); }
    else { staleConfig.add(removed, std::vector<std::string>{"left", "right"}); }
    BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load(solver, staleConfig), std::invalid_argument);
  }

  auto defaultRateConfig = rollingConfiguration("rollingContact", differentialWheels());
  const auto loadedDefaultRates = mc_solver::ConstraintSetLoader::load(solver, defaultRateConfig);
  const auto defaultRates = std::dynamic_pointer_cast<mc_solver::RollingContactConstraint>(loadedDefaultRates);
  BOOST_REQUIRE(defaultRates);
  BOOST_CHECK(!defaultRates->options().trackRotatingRates);
  BOOST_CHECK(!defaultRates->options().softLateralRows);
  BOOST_CHECK_CLOSE(defaultRates->options().lateralSlackWeight, 1e5, 1e-12);
  BOOST_CHECK_CLOSE(defaultRates->options().rollingRateWeight, 200.0, 1e-12);
  BOOST_CHECK_CLOSE(defaultRates->options().steeringRateWeight, 200.0, 1e-12);

  auto dynamicsConfig = rollingConfiguration("rollingContactDynamics", differentialWheels());
  const Eigen::Vector3d rampNormal = Eigen::Vector3d{0.1, -0.2, 1.0}.normalized();
  dynamicsConfig.add("terrainNormal", rampNormal);
  dynamicsConfig.add("infTorque", true);
  const auto loadedDynamics = mc_solver::ConstraintSetLoader::load(solver, dynamicsConfig);
  const auto dynamics = std::dynamic_pointer_cast<mc_solver::RollingContactDynamicsConstraint>(loadedDynamics);
  BOOST_REQUIRE(dynamics);
  BOOST_REQUIRE_EQUAL(dynamics->wheels().size(), 2);
  BOOST_CHECK_SMALL((dynamics->terrainNormal() - rampNormal).norm(), 1e-12);

  mc_rtc::Configuration missingWheels;
  missingWheels.add("type", "rollingContact");
  missingWheels.add("robot", "rolling_diff");
  BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load(solver, missingWheels), std::invalid_argument);

  auto invalidMode = rollingConfiguration("rollingContact", differentialWheels());
  invalidMode("wheels")[0].add("mode", "hovering");
  BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load(solver, invalidMode), std::invalid_argument);

  auto missingRadius = rollingConfiguration("rollingContactDynamics", differentialWheels());
  missingRadius("wheels")[0].remove("radius");
  BOOST_CHECK_THROW(mc_solver::ConstraintSetLoader::load(solver, missingRadius), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(RollingDynamicForcesShareTheWholeBodyQP)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions rollingOptions;
  rollingOptions.velocityGain = 0.0;
  rollingOptions.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, rollingOptions);
  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  targetTask.target(Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof()));
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  solver.addConstraintSet(rolling);

  BOOST_CHECK_EQUAL(solver.data().nrVars(), 24);
  BOOST_CHECK_EQUAL(solver.data().totalLambda(), 16);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("left"), 8);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("right"), 16);
  BOOST_CHECK_EQUAL(dynamics.lambdaCount("left"), 8);
  rolling.update(solver);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

  const Eigen::VectorXd solvedLambda = solver.solver().lambdaVec();
  BOOST_REQUIRE_EQUAL(solvedLambda.size(), 16);
  BOOST_CHECK_GE(solvedLambda.minCoeff(), -1e-10);
  const Eigen::VectorXd motionValue = dynamics.motionConstr().matrix() * solver.result();
  BOOST_CHECK_SMALL((motionValue.head(6) - dynamics.motionConstr().LowerGenInEq().head(6)).norm(), 1e-7);
  BOOST_CHECK_GE(dynamics.normalForce("left", solvedLambda.head(8)), -1e-10);
  BOOST_CHECK_GE(dynamics.frictionMargin("left", solvedLambda.head(8)), -1e-10);

  Eigen::VectorXd lambda(8);
  lambda << 0.3, 1.1, 0.7, 0.2, 1.4, 0.4, 0.6, 0.9;
  const Eigen::VectorXd generalized = dynamics.generalizedForceMatrix("left") * lambda;
  const auto forces = dynamics.endpointForces("left", lambda);
  const auto & geometry = dynamics.geometryResult("left");
  auto & robot = solver.robot(0);
  const auto wheelBody = robot.bodyIndexByName("left_wheel");
  const auto & X_0_b = robot.mbc().bodyPosW[wheelBody];
  const Eigen::Vector3d startInBody = X_0_b.rotation() * (geometry.lineStart - X_0_b.translation());
  const Eigen::Vector3d endInBody = X_0_b.rotation() * (geometry.lineEnd - X_0_b.translation());
  const auto originalQ = robot.mbc().q;
  const auto originalAlpha = robot.mbc().alpha;
  const auto originalAlphaD = robot.mbc().alphaD;
  Eigen::VectorXd velocity(robot.mb().nrDof());
  velocity << 0.17, -0.23, 0.31, 0.57, -0.19, 0.11, 0.83, -0.47;
  constexpr double step = 1e-7;
  auto worldPoint = [](const sva::PTransformd & X_0_body, const Eigen::Vector3d & local)
  { return X_0_body.rotation().transpose() * local + X_0_body.translation(); };

  for(auto & alphaD : robot.mbc().alphaD) { std::fill(alphaD.begin(), alphaD.end(), 0.0); }
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.mbc().q = originalQ;
  robot.eulerIntegration(step);
  robot.forwardKinematics();
  const auto & plusX = robot.mbc().bodyPosW[wheelBody];
  const Eigen::Vector3d plusStart = worldPoint(plusX, startInBody);
  const Eigen::Vector3d plusEnd = worldPoint(plusX, endInBody);
  robot.mbc().q = originalQ;
  rbd::vectorToParam(-velocity, robot.mbc().alpha);
  robot.eulerIntegration(step);
  robot.forwardKinematics();
  const auto & minusX = robot.mbc().bodyPosW[wheelBody];
  const Eigen::Vector3d minusStart = worldPoint(minusX, startInBody);
  const Eigen::Vector3d minusEnd = worldPoint(minusX, endInBody);
  const Eigen::Vector3d startVelocity = (plusStart - minusStart) / (2.0 * step);
  const Eigen::Vector3d endVelocity = (plusEnd - minusEnd) / (2.0 * step);
  const double pointPower = startVelocity.dot(forces[0]) + endVelocity.dot(forces[1]);
  BOOST_CHECK_SMALL(velocity.dot(generalized) - pointPower, 1e-7);

  const auto wrench = dynamics.resultantWrenchAtCarrier("left", lambda);
  const int driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("left_drive")));
  BOOST_CHECK_SMALL(generalized(driveDof) - wrench.head<3>().dot(geometry.lateralDirection), 1e-9);
  BOOST_CHECK_SMALL(wrench.head<3>().dot(geometry.lateralDirection)
                        + wheels[0].radius * (forces[0] + forces[1]).dot(geometry.rollingDirection),
                    1e-9);

  robot.mbc().q = originalQ;
  robot.mbc().alpha = originalAlpha;
  robot.mbc().alphaD = originalAlphaD;
  robot.forwardKinematics();
  robot.forwardVelocity();
  dynamics.motionConstr().computeTorque(solver.solver().alphaDVec(), solver.solver().lambdaVec());
  const Eigen::VectorXd torque = dynamics.motionConstr().torque();
  BOOST_CHECK_SMALL(torque.head(6).lpNorm<Eigen::Infinity>(), 1e-8);
  BOOST_CHECK_LE(torque(6), robot.tu()[robot.jointIndexByName("left_drive")][0] + 1e-9);
  BOOST_CHECK_GE(torque(6), robot.tl()[robot.jointIndexByName("left_drive")][0] - 1e-9);

  const int leftBegin = dynamics.lambdaBegin("left");
  solver.removeConstraintSet(dynamics);
  BOOST_CHECK_EQUAL(solver.data().totalLambda(), 0);
  solver.addConstraintSet(dynamics);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("left"), leftBegin);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(RollingDynamicConeFrameAndStableLayout)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  targetTask.target(Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof()));
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

  const int initialVariables = solver.data().nrVars();
  const int initialBegin = dynamics.lambdaBegin("front_left");
  const Eigen::MatrixXd initialMap = dynamics.generalizedForceMatrix("front_left");
  auto & robot = solver.robot(0);
  const auto drive = robot.jointIndexByName("front_left_drive");
  robot.mbc().q[drive][0] = 1.1;
  robot.forwardKinematics();
  robot.forwardVelocity();
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  BOOST_CHECK_EQUAL(solver.data().nrVars(), initialVariables);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("front_left"), initialBegin);
  BOOST_CHECK_SMALL((dynamics.generalizedForceMatrix("front_left") - initialMap).norm(), 1e-10);

  const auto steer = robot.jointIndexByName("front_left_steer");
  robot.mbc().q[steer][0] = 0.4;
  robot.forwardKinematics();
  robot.forwardVelocity();
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  BOOST_CHECK_EQUAL(solver.data().nrVars(), initialVariables);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("front_left"), initialBegin);
  BOOST_CHECK_GT((dynamics.generalizedForceMatrix("front_left") - initialMap).norm(), 1e-3);

  const Eigen::Vector3d rampNormal = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
  dynamics.terrainNormal(rampNormal);
  BOOST_CHECK_SMALL((dynamics.terrainNormal() - rampNormal).norm(), 1e-12);
  BOOST_CHECK_THROW(dynamics.terrainNormal(Eigen::Vector3d::Zero()), std::invalid_argument);
  BOOST_CHECK_THROW(dynamics.terrainNormal(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
                    std::invalid_argument);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  BOOST_CHECK_SMALL((dynamics.geometryResult("front_left").normalDirection - rampNormal).norm(), 1e-12);
  for(unsigned int endpoint = 0; endpoint < 2; ++endpoint)
  {
    const auto & generators = dynamics.forceGenerators("front_left", endpoint);
    BOOST_REQUIRE_EQUAL(generators.cols(), 4);
    for(Eigen::Index generator = 0; generator < generators.cols(); ++generator)
    {
      const Eigen::Vector3d direction = generators.col(generator);
      const double normalComponent = rampNormal.dot(direction);
      const Eigen::Vector3d tangent = direction - normalComponent * rampNormal;
      BOOST_CHECK_CLOSE(direction.norm(), 1.0, 1e-10);
      BOOST_CHECK_GT(normalComponent, 0.0);
      BOOST_CHECK_CLOSE(tangent.norm() / normalComponent, wheels[0].friction, 1e-8);
    }
  }
  BOOST_CHECK_THROW(dynamics.forceGenerators("front_left", 2), std::out_of_range);

  Eigen::VectorXd lambda(8);
  lambda << 0.8, 0.2, 0.4, 0.6, 1.2, 0.7, 0.3, 0.9;
  BOOST_CHECK_GT(dynamics.normalForce("front_left", lambda), 0.0);
  BOOST_CHECK_GE(dynamics.frictionMargin("front_left", lambda), -1e-12);
  BOOST_CHECK_LE(dynamics.tangentialForce("front_left", lambda),
                 wheels[0].friction * dynamics.normalForce("front_left", lambda) + 1e-12);
}

BOOST_AUTO_TEST_CASE(RollingTasksModesKeepVariablesAndEnforceForcePolicy)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  targetTask.target(Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof()));
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  BOOST_REQUIRE(solve(solver, rolling));

  const int variableCount = solver.data().nrVars();
  const int leftBegin = dynamics.lambdaBegin("left");
  const int rightBegin = dynamics.lambdaBegin("right");
  const int boundCount = solver.solver().nrBoundConstraints();
  const size_t initialRevision = rolling.layoutRevision();

  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof());
  velocity(3) = 0.4;
  rbd::vectorToParam(velocity, solver.robot(0).mbc().alpha);
  solver.robot(0).forwardVelocity();
  rolling.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  BOOST_REQUIRE(solve(solver, rolling));
  BOOST_CHECK(rolling.mode("left") == mc_rbdyn::RollingContactMode::Sliding);
  BOOST_CHECK(dynamics.mode("left") == mc_rbdyn::RollingContactMode::Sliding);
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 4);
  BOOST_CHECK_EQUAL(rolling.softMatrix().rows(), 0);
  BOOST_CHECK_GT(rolling.layoutRevision(), initialRevision);
  BOOST_CHECK_EQUAL(solver.data().nrVars(), variableCount);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("left"), leftBegin);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("right"), rightBegin);
  BOOST_CHECK_EQUAL(solver.solver().nrBoundConstraints(), boundCount);

  const int selected = dynamics.slidingGenerator("left");
  BOOST_CHECK_GE(selected, 0);
  BOOST_CHECK_LT(selected, 4);
  const Eigen::VectorXd slidingLambda = solver.solver().lambdaVec().head<8>();
  for(int point = 0; point < 2; ++point)
  {
    for(int generator = 0; generator < 4; ++generator)
    {
      if(generator != selected) { BOOST_CHECK_SMALL(slidingLambda(point * 4 + generator), 1e-10); }
    }
  }
  const auto slidingForces = dynamics.endpointForces("left", slidingLambda);
  const auto & slidingGeometry = dynamics.geometryResult("left");
  const Eigen::Vector3d totalForce = slidingForces[0] + slidingForces[1];
  const Eigen::Vector3d tangentForce =
      totalForce - slidingGeometry.normalDirection.dot(totalForce) * slidingGeometry.normalDirection;
  BOOST_CHECK_LE(tangentForce.dot(slidingGeometry.slipVelocity), 1e-10);
  BOOST_CHECK_LE(dynamics.slidingDirection("left").dot(slidingGeometry.slipVelocity), 1e-10);

  velocity.setZero();
  rbd::vectorToParam(velocity, solver.robot(0).mbc().alpha);
  solver.robot(0).forwardVelocity();
  BOOST_REQUIRE(solve(solver, rolling));
  BOOST_CHECK_EQUAL(dynamics.slidingGenerator("left"), selected);

  rolling.mode("left", mc_rbdyn::RollingContactMode::Detached);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Detached);
  BOOST_REQUIRE(solve(solver, rolling));
  BOOST_CHECK_EQUAL(rolling.activation("left"), 0.0);
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 3);
  BOOST_CHECK_SMALL(solver.solver().lambdaVec().head<8>().norm(), 1e-12);
  BOOST_CHECK_EQUAL(solver.data().nrVars(), variableCount);
  BOOST_CHECK_EQUAL(dynamics.lambdaBegin("left"), leftBegin);

  rolling.mode("left", mc_rbdyn::RollingContactMode::Fixed, 0.25);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Fixed);
  BOOST_REQUIRE(solve(solver, rolling));
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 3);
  BOOST_CHECK_EQUAL(rolling.softMatrix().rows(), 3);
  BOOST_CHECK_CLOSE(rolling.activation("left"), 0.25, 1e-12);
  for(const auto & label : rolling.softRowLabels()) { BOOST_CHECK(label.find("left/") == 0); }

  rolling.activation("left", 1.0);
  BOOST_REQUIRE(solve(solver, rolling));
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 6);
  BOOST_CHECK_EQUAL(rolling.softMatrix().rows(), 0);
  BOOST_CHECK(std::find(rolling.hardRowLabels().begin(), rolling.hardRowLabels().end(), "left/drive-lock")
              != rolling.hardRowLabels().end());
  BOOST_CHECK_THROW(rolling.mode("missing"), std::out_of_range);
  BOOST_CHECK_THROW(rolling.mode("left", mc_rbdyn::RollingContactMode::Rolling, -0.1), std::invalid_argument);
  BOOST_CHECK_THROW(dynamics.mode("missing"), std::out_of_range);

  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  BOOST_CHECK_EQUAL(solver.data().totalLambda(), 0);
}

BOOST_AUTO_TEST_CASE(RollingTasksTVMStaticSolutionParity)
{
  const auto tasks = backendParitySnapshot<mc_solver::TasksQPSolver>();
  const auto tvm = backendParitySnapshot<mc_solver::TVMQPSolver>();

  checkScaledParity(tasks.hardMatrix, tvm.hardMatrix, 1e-10, "rolling coefficient matrix");
  checkScaledParity(tasks.hardRhs, tvm.hardRhs, 1e-10, "rolling acceleration bias");
  // The default TVM least-squares solver terminates this redundant static-hold problem at about 4e-7.
  checkScaledParity(tasks.acceleration, tvm.acceleration, 1e-6, "generalized acceleration");
  checkScaledParity(tasks.torque.tail(2), tvm.torque.tail(2), 1e-7, "actuated wheel torque");
  checkScaledParity(tasks.wrench, tvm.wrench, 1e-7, "resultant contact wrench about world origin");

  BOOST_CHECK_SMALL(tasks.torque.head(6).lpNorm<Eigen::Infinity>(), 1e-10);
  BOOST_CHECK_SMALL(tvm.torque.head(6).lpNorm<Eigen::Infinity>(), 1e-10);
  BOOST_CHECK_SMALL(tasks.kinematicsResidual, 1e-8);
  BOOST_CHECK_SMALL(tvm.kinematicsResidual, 1e-6);
  BOOST_CHECK_SMALL(tasks.dynamicsResidual, 1e-7);
  BOOST_CHECK_SMALL(tvm.dynamicsResidual, 1e-7);
}

BOOST_AUTO_TEST_CASE(RollingTVMModesRebuildRowsButKeepForceVariables)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TVMQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addTask(&posture);
  BOOST_REQUIRE(solver.run());

  rolling.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 4);
  BOOST_CHECK_EQUAL(rolling.tvmHardFunction().size(), 4);
  BOOST_CHECK(rolling.tvmSoftFunction() == nullptr);
  const auto slip = dynamics.geometryResult("left").slipVelocity;
  if(slip.norm() > 1e-6) { BOOST_CHECK_LE(dynamics.slidingDirection("left").dot(slip), 1e-10); }

  rolling.mode("left", mc_rbdyn::RollingContactMode::Detached);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Detached);
  BOOST_REQUIRE(solver.run());
  const auto detachedForces = dynamics.endpointForces("left");
  BOOST_CHECK_SMALL(detachedForces[0].norm(), 1e-9);
  BOOST_CHECK_SMALL(detachedForces[1].norm(), 1e-9);
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 3);

  rolling.mode("left", mc_rbdyn::RollingContactMode::Rolling, 0.5);
  dynamics.mode("left", mc_rbdyn::RollingContactMode::Rolling);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 3);
  BOOST_CHECK_EQUAL(rolling.softMatrix().rows(), 2);
  BOOST_REQUIRE(rolling.tvmSoftFunction());
  BOOST_CHECK_EQUAL(rolling.tvmSoftFunction()->size(), 2);
  const auto * softFunction = rolling.tvmSoftFunction();
  const size_t transitionLayoutRevision = rolling.layoutRevision();
  rolling.activation("left", 0.75);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_EQUAL(rolling.layoutRevision(), transitionLayoutRevision);
  BOOST_CHECK_EQUAL(rolling.tvmSoftFunction(), softFunction);

  rolling.activation("left", 1.0);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_EQUAL(rolling.hardMatrix().rows(), 5);
  BOOST_CHECK_EQUAL(rolling.softMatrix().rows(), 0);
  BOOST_CHECK(rolling.tvmSoftFunction() == nullptr);
  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(RollingTVMSoftRowsInfluenceSolution)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TVMQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.rollingWeight = 1e6;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  rolling.activation("left", 0.5);
  rolling.activation("right", 0.5);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  mc_tasks::PositionTask position("chassis", solver.robots(), 0, 20.0, 2000.0);
  position.position(Eigen::Vector3d{0.0, 0.0, 0.2});
  mc_tasks::OrientationTask orientation("chassis", solver.robots(), 0, 20.0, 500.0);
  solver.addTask(&posture);
  solver.addTask(&position);
  solver.addTask(&orientation);

  BOOST_REQUIRE(solver.run());
  BOOST_REQUIRE_EQUAL(rolling.hardMatrix().rows(), 0);
  BOOST_REQUIRE_EQUAL(rolling.softMatrix().rows(), 5);
  const auto & alphaD = solver.robot(0).tvmRobot().alphaD()->value();
  const double residual = (rolling.softMatrix() * alphaD - rolling.softRhs()).lpNorm<Eigen::Infinity>();
  BOOST_TEST_MESSAGE("TVM soft rolling residual: " << residual);
  BOOST_CHECK_SMALL(residual, 1e-4);
  BOOST_CHECK_GT(dynamics.normalForce("left") + dynamics.normalForce("right"), 100.0);
  BOOST_CHECK_SMALL(solver.robot(0).tvmRobot().tau()->value().head<6>().norm(), 1e-10);

  for(const auto & wheel : wheels)
  {
    rolling.mode(wheel.name, mc_rbdyn::RollingContactMode::Detached);
    dynamics.mode(wheel.name, mc_rbdyn::RollingContactMode::Detached);
  }
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_SMALL(dynamics.normalForce("left") + dynamics.normalForce("right"), 1e-9);

  for(const auto & wheel : wheels)
  {
    rolling.mode(wheel.name, mc_rbdyn::RollingContactMode::Rolling, 0.5);
    dynamics.mode(wheel.name, mc_rbdyn::RollingContactMode::Rolling);
  }
  BOOST_REQUIRE(solver.run());
  const auto & recoveredAlphaD = solver.robot(0).tvmRobot().alphaD()->value();
  const double recoveredResidual =
      (rolling.softMatrix() * recoveredAlphaD - rolling.softRhs()).lpNorm<Eigen::Infinity>();
  BOOST_CHECK_SMALL(recoveredResidual, 1e-3);
  BOOST_CHECK_GT(dynamics.normalForce("left") + dynamics.normalForce("right"), 100.0);
  BOOST_CHECK_SMALL(solver.robot(0).tvmRobot().tau()->value().head<6>().norm(), 1e-10);

  solver.removeTask(&orientation);
  solver.removeTask(&position);
  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(RollingTVMKinematicsDynamicsAndLifecycle)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TVMQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);

  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addTask(&posture);
  double driveTarget = 0.0;
  for(int iteration = 0; iteration < 300; ++iteration)
  {
    driveTarget += solver.dt();
    posture.target({{"left_drive", {driveTarget}}, {"right_drive", {driveTarget}}});
    BOOST_REQUIRE(solver.run());
  }

  const auto & alphaD = solver.robot(0).tvmRobot().alphaD()->value();
  const auto & hardFunction = rolling.tvmHardFunction();
  BOOST_CHECK_SMALL((hardFunction.matrix() - rolling.hardMatrix()).norm(), 1e-12);
  BOOST_CHECK_SMALL((hardFunction.rhs() - rolling.hardRhs()).norm(), 1e-12);
  BOOST_CHECK_SMALL((rolling.hardMatrix() * alphaD - rolling.hardRhs()).lpNorm<Eigen::Infinity>(), 1e-8);
  dynamics.dynamicFunction().updateValue();
  BOOST_CHECK_SMALL(dynamics.dynamicFunction().value().lpNorm<Eigen::Infinity>(), 1e-7);
  BOOST_CHECK_THROW(dynamics.lambdaBegin("left"), std::logic_error);
  BOOST_CHECK_THROW(dynamics.lambdaCount("left"), std::logic_error);
  BOOST_CHECK_THROW(rolling.tasksFullHardMatrix(), std::logic_error);
  BOOST_CHECK_THROW(rolling.tasksAlphaDBegin(), std::logic_error);

  for(const auto & wheel : wheels)
  {
    const auto forces = dynamics.endpointForces(wheel.name);
    BOOST_CHECK(forces[0].allFinite());
    BOOST_CHECK(forces[1].allFinite());
    BOOST_CHECK_GE(dynamics.normalForce(wheel.name), -1e-9);
    BOOST_CHECK_GE(dynamics.frictionMargin(wheel.name), -1e-8);
    BOOST_CHECK_LE(dynamics.tangentialForce(wheel.name),
                   wheel.friction * dynamics.normalForce(wheel.name) + 1e-8);
  }

  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_SMALL((rolling.hardMatrix() * solver.robot(0).tvmRobot().alphaD()->value() - rolling.hardRhs())
                        .lpNorm<Eigen::Infinity>(),
                    1e-8);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.removeTask(&posture);
}

BOOST_AUTO_TEST_CASE(RollingTVMFourSteeringRepeatedLifecycle)
{
  for(int replay = 0; replay < 4; ++replay)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TVMQPSolver solver(robots, 0.005);
    const auto wheels = fourSteeringWheels();
    mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 5.0;
    options.softLateralRows = true;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
    mc_tasks::PostureTask posture(solver, 0, 5.0, 500.0);
    std::vector<tasks::qp::JointStiffness> steeringGains;
    std::map<std::string, double> steeringWeights;
    for(const auto & wheel : wheels)
    {
      steeringGains.emplace_back(wheel.steeringJoint, 1000.0);
      steeringWeights.emplace(wheel.steeringJoint, 10.0);
    }
    posture.jointStiffness(solver, steeringGains);
    posture.jointWeights(steeringWeights);
    const Eigen::VectorXd dimWeight = posture.dimWeight();
    BOOST_REQUIRE_EQUAL(dimWeight.size(), 8);
    BOOST_CHECK_CLOSE(dimWeight(0), 10.0, 1e-12);
    BOOST_CHECK_CLOSE(dimWeight(2), 10.0, 1e-12);
    BOOST_CHECK_CLOSE(dimWeight(4), 10.0, 1e-12);
    BOOST_CHECK_CLOSE(dimWeight(6), 10.0, 1e-12);
    mc_tasks::OrientationTask orientation("chassis", solver.robots(), 0, 10.0, 500.0);
    mc_tasks::PositionTask position("chassis", solver.robots(), 0, 5.0, 2000.0);
    solver.addConstraintSet(dynamics);
    solver.addConstraintSet(rolling);
    solver.addTask(&posture);
    solver.addTask(&orientation);
    solver.addTask(&position);
    BOOST_REQUIRE(solver.run());
    solver.removeTask(&position);
    solver.removeTask(&orientation);
    solver.removeTask(&posture);
    solver.removeConstraintSet(rolling);
    solver.removeConstraintSet(dynamics);
  }
}

BOOST_AUTO_TEST_CASE(RollingTVMRotatingRateRowsReachTheSolver)
{
  auto steeringAcceleration = [](bool trackRotatingRates)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TVMQPSolver solver(robots, 0.005);
    auto & robot = solver.robot(0);
    const auto wheels = fourSteeringWheels();
    mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 5.0;
    options.softLateralRows = true;
    options.trackRotatingRates = trackRotatingRates;
    options.steeringRateWeight = 50000.0;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
    mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
    solver.addConstraintSet(dynamics);
    solver.addConstraintSet(rolling);
    solver.addTask(&posture);
    for(const auto & wheel : wheels) { rolling.rotatingRateReference(wheel.name, 0.0, 0.6); }
    BOOST_REQUIRE(solver.run());
    if(trackRotatingRates)
    {
      checkTVMSoftFunctionMatchesRows(rolling);
      (void)softRowIndex(rolling, "front_left/steering-rate");
    }
    const auto steer = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
    const double alphaD = solver.robot(0).tvmRobot().alphaD()->value()(steer);
    solver.removeTask(&posture);
    solver.removeConstraintSet(rolling);
    solver.removeConstraintSet(dynamics);
    return alphaD;
  };

  const double tracked = steeringAcceleration(true);
  const double untracked = steeringAcceleration(false);
  BOOST_TEST_MESSAGE("TVM steering acceleration with rate tracking: " << tracked << ", without: " << untracked);
  BOOST_CHECK_SMALL(untracked, 1e-6);
  // The posture task regularizes the same acceleration, so the observed value is
  // about 1.48 rad/s^2: the threshold keeps a comfortable margin below it.
  BOOST_CHECK_GT(tracked, 1.0);
}

BOOST_AUTO_TEST_CASE(RollingTVMRotatingRateRowsRebuildOnLayoutChange)
{
  auto robots = loadFourSteeringRobot();
  mc_solver::TVMQPSolver solver(robots, 0.005);
  const auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 5.0;
  options.softLateralRows = true;
  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addTask(&posture);

  // A mode or activation change adds or drops rate rows, so the TVM function
  // must be rebuilt at the new size or the solver reads a stale block.
  auto checkSoftFunctionMatchesRows = [&](size_t expectedRateRows)
  {
    const auto & labels = rolling.softRowLabels();
    const size_t rateRows = static_cast<size_t>(
        std::count_if(labels.begin(), labels.end(),
                      [](const std::string & label) { return label.find("-rate") != std::string::npos; }));
    BOOST_CHECK_EQUAL(rateRows, expectedRateRows);
    checkTVMSoftFunctionMatchesRows(rolling);
  };

  BOOST_REQUIRE(solver.run());
  checkSoftFunctionMatchesRows(8);

  rolling.mode("front_left", mc_rbdyn::RollingContactMode::Detached);
  dynamics.mode("front_left", mc_rbdyn::RollingContactMode::Detached);
  BOOST_REQUIRE(solver.run());
  checkSoftFunctionMatchesRows(6);

  rolling.mode("front_left", mc_rbdyn::RollingContactMode::Rolling, 0.5);
  dynamics.mode("front_left", mc_rbdyn::RollingContactMode::Rolling);
  BOOST_REQUIRE(solver.run());
  checkSoftFunctionMatchesRows(8);

  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(LateralRowsAreStructurallyOverDeterminedOnThreePlanarDof)
{
  // ROW-08. Four lateral rows act on three planar chassis DOF. The rank of that
  // block is the whole argument for softening them: rank 3 means the only
  // admissible planar twist is zero, so hard rows freeze the chassis.
  const auto coordinated = icrSteeringAngles(0.15, 0.0, 0.05);
  const std::array<double, 4> uncoordinated = {0.30, -0.10, 0.20, -0.30};
  BOOST_REQUIRE(hasCommonICR(coordinated));
  BOOST_REQUIRE(!hasCommonICR(uncoordinated));

  // The planar 4x3 block of the theory, straight out of steeringRollingMatrix().
  const Eigen::MatrixXd planarCoordinated = planarLateralBlock(coordinated);
  const Eigen::MatrixXd planarUncoordinated = planarLateralBlock(uncoordinated);
  BOOST_TEST_MESSAGE("Planar lateral block, coordinated: sv=[" << formatSingularValues(planarCoordinated) << "]");
  BOOST_TEST_MESSAGE("Planar lateral block, uncoordinated: sv=[" << formatSingularValues(planarUncoordinated) << "]");
  BOOST_CHECK_EQUAL(rankAt(planarCoordinated, 1e-9), 2);
  BOOST_CHECK_EQUAL(rankAt(planarUncoordinated, 1e-9), 3);
  // Assert the singular values, not only the rank integer: a nominally full but
  // near-singular block is a different finding from a robustly full one.
  const Eigen::VectorXd coordinatedSv = singularValues(planarCoordinated);
  BOOST_CHECK_CLOSE(coordinatedSv(0), 1.97723, 1e-2);
  BOOST_CHECK_CLOSE(coordinatedSv(1), 0.951658, 1e-2);
  BOOST_CHECK_SMALL(coordinatedSv(2), 1e-12);
  const Eigen::VectorXd uncoordinatedSv = singularValues(planarUncoordinated);
  BOOST_CHECK_CLOSE(uncoordinatedSv(0), 1.95018, 1e-2);
  BOOST_CHECK_CLOSE(uncoordinatedSv(1), 0.87305, 1e-2);
  // Not marginal: the third direction is a fifth of the second, far above any
  // sensible rank threshold, so the over-determination is structural.
  BOOST_CHECK_CLOSE(uncoordinatedSv(2), 0.440016, 1e-2);

  // The whole-body rows the QP actually assembles carry the same block. The
  // steering and drive columns are exactly zero (assumption A3: the steering
  // axis passes through the carrier centre), so the lateral rows reach the
  // chassis only, and their planar sub-block reproduces the theory's a_i^T.
  for(const bool coordinatedCase : {true, false})
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    setSteeringAngles(solver.robot(0), coordinatedCase ? coordinated : uncoordinated);
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 0.0;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
    const Eigen::MatrixXd lateral = labelledRows(rolling.hardMatrix(), rolling.hardRowLabels(), "/lateral");
    BOOST_REQUIRE_EQUAL(lateral.rows(), 4);
    BOOST_TEST_MESSAGE("Whole-body lateral block, " << (coordinatedCase ? "coordinated" : "uncoordinated")
                                                    << ": sv=[" << formatSingularValues(lateral) << "]");
    BOOST_CHECK_SMALL(lateral.rightCols(lateral.cols() - 6).lpNorm<Eigen::Infinity>(), 1e-12);
    BOOST_CHECK_EQUAL(rankAt(lateral, 1e-9), coordinatedCase ? 2 : 3);
    const Eigen::VectorXd expected = singularValues(coordinatedCase ? planarCoordinated : planarUncoordinated);
    const Eigen::VectorXd observed = singularValues(lateral);
    BOOST_CHECK_SMALL((observed.head(3) - expected).norm(), 1e-9);
    BOOST_CHECK_SMALL(observed(3), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(IcrConcurrencyIsExactlyTheLateralRankCondition)
{
  // GEO-09. hasCommonICR() is computed from the wheel-axis lines alone, so this
  // is an equivalence between two independently-derived predicates, not a
  // restatement of one in terms of the other.
  std::vector<std::array<double, 4>> configurations;
  // Coordinated by construction: every planar twist has a common ICR.
  for(const double yaw : {0.05, 0.4, -0.3})
  {
    for(const double linear : {0.15, -0.2})
    {
      for(const double lateral : {0.0, 0.1}) { configurations.push_back(icrSteeringAngles(linear, lateral, yaw)); }
    }
  }
  // Pure translation: all headings parallel, ICR at infinity.
  for(const double heading : {0.0, 0.3, -0.9, 1.2})
  {
    configurations.push_back({heading, heading, heading, heading});
  }
  // Deliberately not coordinated.
  configurations.push_back({0.30, -0.10, 0.20, -0.30});
  configurations.push_back({0.0, 0.0, 0.0, 0.2});
  configurations.push_back({0.1, 0.2, 0.3, 0.4});
  configurations.push_back({-0.5, 0.5, 0.5, -0.5});
  configurations.push_back({0.7, -0.7, 0.2, 0.9});

  size_t coordinatedCount = 0;
  for(const auto & angles : configurations)
  {
    const Eigen::MatrixXd block = planarLateralBlock(angles);
    const bool concurrent = hasCommonICR(angles);
    const Eigen::Index rank = rankAt(block, 1e-9);
    BOOST_TEST_MESSAGE("ICR=" << concurrent << " rank=" << rank << " sv=[" << formatSingularValues(block) << "]");
    BOOST_CHECK_EQUAL(concurrent, rank <= 2);
    coordinatedCount += concurrent ? 1 : 0;
  }
  // Both branches are exercised, so the equivalence above is not vacuous.
  BOOST_CHECK_EQUAL(coordinatedCount, 16u);
  BOOST_CHECK_EQUAL(configurations.size() - coordinatedCount, 5u);
}

BOOST_AUTO_TEST_CASE(LateralSlacksAreTheRangeSpaceDefect)
{
  // ROW-09. The softening is realised by moving the lateral rows into the
  // objective, which is the explicit-slack formulation written out: minimising
  // w||A x - c||^2 IS minimising w||sigma||^2 subject to A x - c = sigma. Both
  // halves are asserted: that lateralSlack() reports A x* - c, and that the
  // realised slack is the residual of the least-squares projection of c onto
  // range(A), i.e. orthogonal to that range.
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  setIncompatibleFourWheelState(robot);

  // The identity regulariser below competes with the lateral rows, so the
  // realised slack only reaches the pure least-squares residual as the lateral
  // weight grows. That approach is itself the assertion: a slack that is not the
  // range-space defect would not converge to it at rate 1/w.
  double previousError = 0.0;
  double previousOrthogonality = 0.0;
  double defectNorm = 0.0;
  for(const double weight : {1e6, 1e8, 1e10, 1e12})
  {
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 20.0;
    options.softLateralRows = true;
    options.rollingWeight = 1.0;
    options.lateralSlackWeight = weight;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
    target(3) = 1.0;
    targetTask.target(target);
    solver.addTask(&targetTask);
    solver.addConstraintSet(rolling);
    BOOST_REQUIRE(solve(solver, rolling));
    const Eigen::VectorXd solution = solver.solver().alphaDVec(0);

    // No lateral row reached the hard block.
    BOOST_CHECK_EQUAL(labelledRows(rolling.hardMatrix(), rolling.hardRowLabels(), "/lateral").rows(), 0);
    const std::vector<std::string> expectedLabels = {"front_left", "front_right", "rear_left", "rear_right"};
    BOOST_CHECK(rolling.lateralSlackLabels() == expectedLabels);

    const auto [A, c] = unscaledLateralRows(rolling, robot.mb().nrDof());
    const Eigen::VectorXd slack = rolling.lateralSlack(solution);
    BOOST_REQUIRE_EQUAL(slack.size(), 4);
    // First half of ROW-09, and exact: the accessor reports A x* - c.
    BOOST_CHECK_SMALL((slack - (A * solution - c)).lpNorm<Eigen::Infinity>(), 1e-12);

    // c is genuinely outside range(A), so there is a defect to find at all.
    const Eigen::VectorXd defect = A * A.completeOrthogonalDecomposition().solve(c) - c;
    const double error = (slack - defect).lpNorm<Eigen::Infinity>();
    const double orthogonality = (A.transpose() * slack).lpNorm<Eigen::Infinity>() / slack.norm();
    BOOST_TEST_MESSAGE("w=" << weight << ": ||c||=" << c.norm() << ", ||(I-P)c||=" << defect.norm()
                            << ", ||sigma||=" << slack.norm() << ", ||sigma - (I-P)c||=" << error
                            << ", ||A^T sigma||/||sigma||=" << orthogonality);
    BOOST_REQUIRE_GT(defect.norm(), 1e-3);
    // The least-squares residual is minimal, so no realised slack can be shorter.
    BOOST_CHECK_GE(slack.norm(), defect.norm() - 1e-12);
    if(previousError != 0.0)
    {
      // One hundredfold weight, one hundredfold closer: first order in 1/w.
      BOOST_CHECK_GT(previousError / error, 50.0);
      BOOST_CHECK_LT(previousError / error, 200.0);
      BOOST_CHECK_GT(previousOrthogonality / orthogonality, 50.0);
    }
    previousError = error;
    previousOrthogonality = orthogonality;
    defectNorm = defect.norm();
    solver.removeConstraintSet(rolling);
    solver.removeTask(&targetTask);
  }
  // Second half of ROW-09 at the end of the sweep: the realised slack is the
  // residual of the least-squares projection, and orthogonal to range(A).
  BOOST_TEST_MESSAGE("Range-space defect norm: " << defectNorm);
  BOOST_CHECK_SMALL(previousError, 1e-9);
  BOOST_CHECK_SMALL(previousOrthogonality, 1e-8);
}

BOOST_AUTO_TEST_CASE(IncompatibleFourWheelDataSurfacesAsLateralSlack)
{
  // SMK-07. Incompatible four-wheel data must surface as slack rather than as
  // infeasibility: the solve succeeds and the slack norm is non-zero. The hard
  // half of the comparison is the decisive test below.
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  setIncompatibleFourWheelState(solver.robot(0));

  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 20.0;
  options.softLateralRows = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addTask(&posture);
  BOOST_REQUIRE(solver.run());

  const Eigen::VectorXd slack = rolling.lateralSlack(solver.solver().alphaDVec(0));
  BOOST_TEST_MESSAGE("Incompatible four-wheel lateral slack: [" << slack.transpose() << "], norm " << slack.norm());
  BOOST_CHECK_GT(slack.norm(), 1e-6);
  BOOST_CHECK(slack.allFinite());

  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(HardLateralRowsFreezeTheChassisAndSoftOnesDoNot)
{
  // The decisive regression test. Both halves are asserted, so it demonstrates
  // the difference rather than only the fixed behaviour.
  BOOST_REQUIRE(!hasCommonICR(uncoordinatedSteeringAngles()));

  struct Outcome
  {
    bool solved = false;
    double planarNorm = 0.0;
    double forward = 0.0;
    double hardLateralResidual = 0.0;
    double slackNorm = 0.0;
  };

  auto run = [&](bool soft, double slackWeight, double demand, bool incompatible)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    auto & robot = solver.robot(0);
    if(incompatible) { setIncompatibleFourWheelState(robot); }
    else { setSteeringAngles(robot, uncoordinatedSteeringAngles()); }
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = incompatible ? 20.0 : 0.0;
    options.softLateralRows = soft;
    options.lateralSlackWeight = slackWeight;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
    target(3) = demand;
    targetTask.target(target);
    solver.addTask(&targetTask);
    solver.addConstraintSet(rolling);
    Outcome out;
    out.solved = solve(solver, rolling);
    if(out.solved)
    {
      const Eigen::VectorXd solution = solver.solver().alphaDVec(0);
      // Planar chassis acceleration: [vx, vy, omega] at dof 3, 4 and 2.
      out.planarNorm = Eigen::Vector3d{solution(3), solution(4), solution(2)}.norm();
      out.forward = solution(3);
      const Eigen::MatrixXd hardLateral = labelledRows(rolling.hardMatrix(), rolling.hardRowLabels(), "/lateral");
      if(hardLateral.rows() != 0)
      {
        const auto [A, c] = unscaledLateralRows(rolling, robot.mb().nrDof());
        out.hardLateralResidual = (A * solution - c).lpNorm<Eigen::Infinity>();
      }
      out.slackNorm = rolling.lateralSlack(solution).norm();
    }
    solver.removeConstraintSet(rolling);
    solver.removeTask(&targetTask);
    return out;
  };

  // Half one: a chassis at rest, so the stabilised right-hand side is exactly
  // zero and the hard system A xi_dot = 0 is consistent. Rank 3 on three planar
  // DOF then leaves xi_dot = 0 as the only solution: the QP freezes the chassis
  // no matter how hard the task pulls, and the freeze is a property of the row
  // structure, so raising the demand tenfold changes nothing.
  const Outcome frozen = run(false, 1e5, 1.0, false);
  const Outcome frozenHarder = run(false, 1e5, 10.0, false);
  BOOST_REQUIRE(frozen.solved);
  BOOST_REQUIRE(frozenHarder.solved);
  BOOST_TEST_MESSAGE("Hard lateral rows at rest: ||planar alphaD|| = " << frozen.planarNorm << " at demand 1, "
                                                                      << frozenHarder.planarNorm << " at demand 10");
  BOOST_CHECK_SMALL(frozen.planarNorm, 1e-12);
  BOOST_CHECK_SMALL(frozenHarder.planarNorm, 1e-12);
  BOOST_CHECK_SMALL(frozen.hardLateralResidual, 1e-9);

  // Half two: the same problem with the four lateral rows softened moves, and
  // the twist is a trade-off against the lateral weight rather than a frozen
  // zero. It grows monotonically as the rows are relaxed, and it answers the
  // demand: at a negligible weight it reaches the twist the QP would produce
  // with no lateral rows at all, and it scales linearly with the demand.
  const Outcome unconstrained = run(true, 1e-9, 1.0, false);
  BOOST_REQUIRE(unconstrained.solved);
  double previousPlanar = -1.0;
  for(const double weight : {1e5, 1e3, 1e1, 1.0, 1e-2})
  {
    const Outcome relaxed = run(true, weight, 1.0, false);
    BOOST_REQUIRE(relaxed.solved);
    BOOST_TEST_MESSAGE("Soft lateral rows at w=" << weight << ": ||planar alphaD|| = " << relaxed.planarNorm
                                                 << ", vx = " << relaxed.forward
                                                 << ", ||sigma|| = " << relaxed.slackNorm);
    BOOST_CHECK_GT(relaxed.planarNorm, previousPlanar);
    BOOST_CHECK_GT(relaxed.forward, 0.0);
    previousPlanar = relaxed.planarNorm;
  }
  BOOST_TEST_MESSAGE("Twist with the lateral rows effectively absent: " << unconstrained.planarNorm);
  // Relaxing the lateral rows recovers the twist the QP would produce with no
  // lateral rows at all: the softened rows trade against the command, they do
  // not remove the chassis degrees of freedom the way the rank deficiency does.
  BOOST_CHECK_GT(previousPlanar, 0.9 * unconstrained.planarNorm);
  // At the default weight the chassis still moves, by four orders of magnitude
  // more than the hard case, which is exactly zero.
  const Outcome soft = run(true, 1e5, 1.0, false);
  BOOST_CHECK_GT(soft.planarNorm, 1e-6);
  BOOST_CHECK_GT(soft.forward, 0.0);
  // And it answers the demand linearly, which the frozen case does not.
  const Outcome softHarder = run(true, 1e5, 10.0, false);
  BOOST_CHECK_CLOSE(softHarder.planarNorm, 10.0 * soft.planarNorm, 1e-6);

  // Half three: incompatible measured steering rates put c outside range(A).
  // That is where hard rows stop merely freezing the chassis and become
  // unsatisfiable, while the softened rows report the defect and carry on.
  const Outcome hardMoving = run(false, 1e5, 1.0, true);
  const Outcome softMoving = run(true, 1e5, 1.0, true);
  BOOST_TEST_MESSAGE("Incompatible data, hard: solved=" << hardMoving.solved << ", lateral row residual "
                                                        << hardMoving.hardLateralResidual << "; soft: solved="
                                                        << softMoving.solved << ", ||sigma|| = "
                                                        << softMoving.slackNorm);
  BOOST_CHECK(!hardMoving.solved || hardMoving.hardLateralResidual > 1e-6);
  BOOST_REQUIRE(softMoving.solved);
  BOOST_CHECK_GT(softMoving.slackNorm, 1e-6);
}

BOOST_AUTO_TEST_CASE(LateralSlacksCarryNoBoxConstraint)
{
  // BND-06. The softening moves rows into the objective; it adds no decision
  // variable, so there is nothing to bound and nothing that could be bounded.
  auto snapshot = [](bool soft, bool incompatible)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    if(incompatible) { setIncompatibleFourWheelState(solver.robot(0)); }
    else { setSteeringAngles(solver.robot(0), uncoordinatedSteeringAngles()); }
    auto wheels = fourSteeringWheels();
    mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 20.0;
    options.softLateralRows = soft;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
    mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
    solver.addConstraintSet(dynamics);
    solver.addConstraintSet(rolling);
    solver.addTask(&posture);
    BOOST_REQUIRE(solver.run());
    const std::array<int, 4> counts = {solver.data().nrVars(), solver.solver().nrBoundConstraints(),
                                       solver.solver().nrInequalityConstraints(),
                                       solver.solver().nrGenInequalityConstraints()};
    const Eigen::VectorXd slack = rolling.lateralSlack(solver.solver().alphaDVec(0));
    solver.removeTask(&posture);
    solver.removeConstraintSet(rolling);
    solver.removeConstraintSet(dynamics);
    return std::make_pair(counts, slack);
  };

  // Compatible data, so both configurations solve and the counts are comparable.
  const auto hard = snapshot(false, false);
  const auto soft = snapshot(true, false);
  BOOST_TEST_MESSAGE("Hard nrVars/bounds/ineq/genIneq: " << hard.first[0] << "/" << hard.first[1] << "/"
                                                         << hard.first[2] << "/" << hard.first[3]);
  BOOST_TEST_MESSAGE("Soft nrVars/bounds/ineq/genIneq: " << soft.first[0] << "/" << soft.first[1] << "/"
                                                         << soft.first[2] << "/" << soft.first[3]);
  BOOST_CHECK(hard.first == soft.first);

  // A bounded slack would show as a one-sided residual; this one takes both
  // signs, so nothing is clamping it in either direction.
  const auto incompatible = snapshot(true, true);
  BOOST_TEST_MESSAGE("Soft lateral slack on incompatible data: [" << incompatible.second.transpose() << "]");
  BOOST_CHECK(incompatible.first == soft.first);
  BOOST_CHECK_GT(incompatible.second.maxCoeff(), 1e-9);
  BOOST_CHECK_LT(incompatible.second.minCoeff(), -1e-9);
}

BOOST_AUTO_TEST_CASE(LateralSlackRowsFollowTheBlockWeightAndRejectInvalidWeights)
{
  constexpr double dt = 0.005;
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, dt);
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.rollingWeight = 1000.0;
  options.lateralSlackWeight = 4e4;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
  solver.addConstraintSet(rolling);
  rolling.update(solver);

  // The realised weight of a soft row is rollingWeight * scale^2, and the scale
  // is read off against the unscaled lateral row the geometry publishes.
  auto realisedLateralWeight = [&]()
  {
    const Eigen::Index row = softRowIndex(rolling, "front_left/lateral");
    const Eigen::RowVectorXd unscaled = rolling.geometryResults()[0].rollingMatrix.row(1);
    const Eigen::Index column = 0;
    Eigen::Index best = column;
    for(Eigen::Index i = 0; i < unscaled.size(); ++i)
    {
      if(std::abs(unscaled(i)) > std::abs(unscaled(best))) { best = i; }
    }
    const double scale = rolling.softMatrix()(row, best) / unscaled(best);
    return rolling.rollingWeight() * scale * scale;
  };

  const double before = realisedLateralWeight();
  const size_t layoutRevision = rolling.layoutRevision();
  rolling.rollingWeight(250.0);
  rolling.update(solver);
  const double after = realisedLateralWeight();
  BOOST_TEST_MESSAGE("Realised lateral-slack weight before rollingWeight(250): " << before << ", after: " << after);
  BOOST_CHECK_CLOSE(before, 4e4, 1e-9);
  BOOST_CHECK_CLOSE(after, 4e4, 1e-9);
  BOOST_CHECK_EQUAL(rolling.layoutRevision(), layoutRevision);
  BOOST_REQUIRE(solve(solver, rolling));
  solver.removeConstraintSet(rolling);

  for(const double weight : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()})
  {
    auto invalid = options;
    invalid.lateralSlackWeight = weight;
    BOOST_CHECK_THROW(invalid.validate(4), std::invalid_argument);
  }
}

namespace
{

/** Tasks/src/GenQPUtils.h::fillQC()'s own unconditional diagonal floor,
 * reproduced here (the library does not expose it) because it lands on the
 * lambda block of the real assembled Hessian regardless of what the
 * generator-regularization term contributes, and QP-01/QP-02 need to report
 * eigenvalues that account for it honestly instead of pretending it away. */
constexpr double tasksLibraryDiagConstant = 1e-4;

/** A representative enabled generatorRegularization value, used by the tests
 * that measure the term's effect once it is turned on. Not
 * RollingContactDynamicsConstraint::defaultGeneratorRegularization, which is
 * 0.0 (see that class's documentation): mc_rtc's Tasks backend already
 * floors the Hessian diagonal unconditionally, so the term is off by
 * default there. This is the value used to measure the trade-off for a
 * caller -- a different backend or solver without such a floor -- who does
 * enable it: four times the library floor, matching what
 * defaultGeneratorRegularization used to be before that measurement showed
 * its physical-force bias was not worth paying by default. */
constexpr double enabledGeneratorRegularization = 2e-4;

Eigen::MatrixXd withTasksLibraryFloor(Eigen::MatrixXd Q)
{
  for(Eigen::Index i = 0; i < Q.rows(); ++i)
  {
    if(std::abs(Q(i, i)) < tasksLibraryDiagConstant) { Q(i, i) += tasksLibraryDiagConstant; }
  }
  return Q;
}

struct GeneratorRegularizationSolution
{
  double normalForce = 0.0;
  double tangentialForce = 0.0;
  Eigen::VectorXd acceleration;
};

/** Solve the same asymmetric, kinematically-compatible acceleration target on
 * the two-wheel differential robot, at a given generatorRegularization. */
GeneratorRegularizationSolution solveWithGeneratorRegularization(double epsilon)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels, false, epsilon);
  mc_solver::RollingContactConstraintOptions rollingOptions;
  rollingOptions.velocityGain = 0.0;
  rollingOptions.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, rollingOptions);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);

  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  const Eigen::VectorXd target = compatibleTarget(solver.robot(0), rolling.hardMatrix(), 2.0, 1.0);
  targetTask.target(target);
  solver.addTask(&targetTask);
  BOOST_REQUIRE(solve(solver, rolling));

  GeneratorRegularizationSolution solution;
  const Eigen::VectorXd solvedLambda = solver.solver().lambdaVec();
  BOOST_REQUIRE_EQUAL(solvedLambda.size(), 16);
  solution.normalForce = dynamics.normalForce("left", solvedLambda.head(8));
  solution.tangentialForce = dynamics.tangentialForce("left", solvedLambda.head(8));
  solution.acceleration = solver.solver().alphaDVec(0);

  solver.removeTask(&targetTask);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  return solution;
}

} // namespace

BOOST_AUTO_TEST_CASE(RollingContactDynamicsConstraintGeneratorRegularizationQCFormula)
{
  // eq:generator-regularization: eps * ||lambda||^2 contributes Q = 2*eps*I,
  // C = 0 to the QP objective over one wheel's eight generator multipliers.
  // This is the exact formula the Tasks-backend per-wheel regularization
  // task bakes into its Q()/C(), shared here so a test failure means the two
  // disagree rather than the test re-deriving the same formula independently.
  const auto formula = mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(8, 0.03);
  const Eigen::MatrixXd & Q = formula.first;
  const Eigen::VectorXd & C = formula.second;
  BOOST_REQUIRE_EQUAL(Q.rows(), 8);
  BOOST_REQUIRE_EQUAL(Q.cols(), 8);
  BOOST_REQUIRE_EQUAL(C.size(), 8);
  BOOST_CHECK_SMALL((Q - 0.06 * Eigen::MatrixXd::Identity(8, 8)).norm(), 1e-15);
  BOOST_CHECK_SMALL(C.norm(), 1e-15);

  // QP-02: epsilon = 0 must give an exactly-zero block, not merely a small
  // one, so the floor measured in
  // RollingContactDynamicsConstraintGeneratorRegularizationQP02ZeroReliesOnlyOnTheLibraryFloor
  // is provably the Tasks library's own DIAG_CONSTANT and not a residual of
  // this formula.
  const auto zero = mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(8, 0.0);
  BOOST_CHECK_EQUAL(zero.first, Eigen::MatrixXd::Zero(8, 8));
  BOOST_CHECK_EQUAL(zero.second, Eigen::VectorXd::Zero(8));

  BOOST_CHECK_THROW(mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(8, -1e-6),
                    std::invalid_argument);
  BOOST_CHECK_THROW(mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(
                        8, std::numeric_limits<double>::quiet_NaN()),
                    std::invalid_argument);
  BOOST_CHECK_THROW(mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(
                        8, std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
  BOOST_CHECK_THROW(mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(-1, 0.03),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(RollingContactDynamicsConstraintGeneratorRegularizationQP02ZeroReliesOnlyOnTheLibraryFloor)
{
  // epsilon = 0 here is not just one point on a sweep: it is
  // defaultGeneratorRegularization (see RollingContactDynamicsConstraint.h),
  // so this test is what justifies that default -- it must show the library
  // floor alone already keeps the Hessian block positive definite.
  BOOST_CHECK_EQUAL(mc_solver::RollingContactDynamicsConstraint::defaultGeneratorRegularization, 0.0);

  // With no regularization, nothing in RollingContactConstraint.cpp or
  // RollingContactDynamicsConstraint.cpp contributes Q/C over lambda: the
  // rolling soft task targets alphaD (Impl::SoftTask in
  // RollingContactConstraint.cpp), and RollingMotionConstr/ModeBound are pure
  // equality/inequality/bound contributors that never implement Q()/C(). So
  // the pre-floor lambda block of the assembled Hessian, for one wheel's
  // eight generator multipliers, is exactly the formula under test at
  // epsilon = 0.
  const auto formula = mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(8, 0.0);
  BOOST_CHECK_EQUAL(formula.first, Eigen::MatrixXd::Zero(8, 8));

  const Eigen::MatrixXd floored = withTasksLibraryFloor(formula.first);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(floored);
  BOOST_REQUIRE_EQUAL(eigen.info(), Eigen::Success);
  BOOST_TEST_MESSAGE(
      "QP-02 smallest eigenvalue without the term (library floor only): " << eigen.eigenvalues().minCoeff());
  // Not "singular" in an absolute sense: the Tasks library's own unconditional
  // DIAG_CONSTANT (Tasks/src/GenQPUtils.h) floors every diagonal entry
  // smaller than 1e-4, so the real solver never sees an exact zero here
  // either way. The claim this term exists for is conditioning quality, not
  // literal invertibility.
  BOOST_CHECK_CLOSE(eigen.eigenvalues().minCoeff(), tasksLibraryDiagConstant, 1e-9);
}

BOOST_AUTO_TEST_CASE(RollingContactDynamicsConstraintGeneratorRegularizationQP01EnabledClearsTheLibraryFloor)
{
  // defaultGeneratorRegularization is 0.0 (see
  // RollingContactDynamicsConstraint.h), so this test does not use it -- it
  // measures the margin a caller gets if they enable the term explicitly,
  // at the representative value enabledGeneratorRegularization.
  const double epsilon = enabledGeneratorRegularization;
  const auto formula = mc_solver::RollingContactDynamicsConstraint::generatorRegularizationQC(8, epsilon);
  const Eigen::MatrixXd floored = withTasksLibraryFloor(formula.first);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(floored);
  BOOST_REQUIRE_EQUAL(eigen.info(), Eigen::Success);
  BOOST_TEST_MESSAGE("QP-01 smallest eigenvalue with the term enabled at (" << epsilon
                     << "): " << eigen.eigenvalues().minCoeff());
  // Enabling the term must make a real difference over the library's own
  // floor, not just restate it: require the achieved margin to be at least
  // double the bare floor. Above that factor the floor branch in the
  // library's fillQC() (`if abs(Q(i,i)) < DIAG_CONSTANT: Q(i,i) +=
  // DIAG_CONSTANT`) never fires, so the margin is a clean multiple of the
  // floor and entirely attributable to this term, not to GenQPUtils.h's
  // incidental safety net. A larger multiple (e.g. ten times) was tried and
  // rejected: see defaultGeneratorRegularization's documentation and
  // ...FRI01BiasAtEnabledAndTenX below for why enabledGeneratorRegularization
  // (and this bound) stop at a factor of four instead -- and why the class
  // default is 0.0 rather than this value, since mc_rtc's Tasks backend
  // already clears the floor without it.
  BOOST_CHECK_GT(eigen.eigenvalues().minCoeff(), 2.0 * tasksLibraryDiagConstant);
  BOOST_CHECK_CLOSE(eigen.eigenvalues().minCoeff(), 2.0 * epsilon, 1e-9);
}

BOOST_AUTO_TEST_CASE(RollingContactDynamicsConstraintGeneratorRegularizationFRI01BiasAtEnabledAndTenX)
{
  // As in QP-01 above, this deliberately does not use
  // defaultGeneratorRegularization (0.0): a bias measurement between 0 and 0
  // would be vacuous. It measures the bias a caller pays if they enable the
  // term at the representative value enabledGeneratorRegularization.
  const double epsilon = enabledGeneratorRegularization;
  const auto atEnabled = solveWithGeneratorRegularization(epsilon);
  const auto atTenX = solveWithGeneratorRegularization(10.0 * epsilon);

  const double normalForceBias = std::abs(atTenX.normalForce - atEnabled.normalForce);
  const double tangentialForceBias = std::abs(atTenX.tangentialForce - atEnabled.tangentialForce);
  const double accelerationBias = (atTenX.acceleration - atEnabled.acceleration).lpNorm<Eigen::Infinity>();
  BOOST_TEST_MESSAGE("FRI-01 solved values at enabled generatorRegularization ("
                     << epsilon << "): normalForce " << atEnabled.normalForce << " N, tangentialForce "
                     << atEnabled.tangentialForce << " N");
  BOOST_TEST_MESSAGE("FRI-01 bias from a 10x generatorRegularization change ("
                     << epsilon << " -> " << 10.0 * epsilon << "): normalForce " << normalForceBias
                     << " N, tangentialForce " << tangentialForceBias << " N, acceleration " << accelerationBias
                     << " rad/s^2 (Linf)");

  // Physical-fidelity budget: a 10x change in the regularization weight must
  // not move the solved contact force by more than 1% of its own magnitude
  // (floored at 1 N for the tangential force, which is legitimately near zero
  // for a mild, close-to-balanced target).
  BOOST_CHECK_LT(normalForceBias, 1e-2 * std::abs(atEnabled.normalForce));
  BOOST_CHECK_LT(tangentialForceBias, 1e-2 * std::max(1.0, std::abs(atEnabled.tangentialForce)));
}

BOOST_AUTO_TEST_CASE(RollingContactDynamicsConstraintGeneratorRegularizationPreservesProblemSizeAcrossModes)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = differentialWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(
      solver.robots(), 0, solver.dt(), wheels, false,
      mc_solver::RollingContactDynamicsConstraint::defaultGeneratorRegularization);
  mc_solver::RollingContactConstraintOptions rollingOptions;
  rollingOptions.velocityGain = 0.0;
  rollingOptions.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, rollingOptions);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addTask(&posture);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  BOOST_REQUIRE(solver.run());
  const int nrVars = solver.data().nrVars();
  const int totalLambda = solver.data().totalLambda();
  const int bounds = solver.solver().nrBoundConstraints();

  BOOST_CHECK_CLOSE(dynamics.generatorRegularization(),
                    mc_solver::RollingContactDynamicsConstraint::defaultGeneratorRegularization, 1e-9);

  dynamics.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  rolling.mode("left", mc_rbdyn::RollingContactMode::Sliding);
  BOOST_REQUIRE(solver.run());
  BOOST_CHECK_EQUAL(solver.data().nrVars(), nrVars);
  BOOST_CHECK_EQUAL(solver.data().totalLambda(), totalLambda);
  BOOST_CHECK_EQUAL(solver.solver().nrBoundConstraints(), bounds);
  BOOST_CHECK_GE(dynamics.frictionMargin("left", solver.solver().lambdaVec().head(8)), -1e-6);

  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
}
