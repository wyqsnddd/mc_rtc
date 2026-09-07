#include <mc_rbdyn/RobotLoader.h>
#include <mc_rbdyn/Robots.h>

#include <mc_solver/EqualityConstraint.h>
#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/KinematicsConstraint.h>
#include <mc_solver/RollingContactConstraint.h>
#include <mc_solver/RollingContactDynamicsConstraint.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_solver/TVMQPSolver.h>

#include <mc_tvm/RollingContactFunction.h>
#include <mc_tvm/Robot.h>

#include <mc_rtc/constants.h>

#include <mc_tasks/PostureTask.h>
#include <mc_tasks/OrientationTask.h>
#include <mc_tasks/PositionTask.h>

#include <boost/test/unit_test.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <RBDyn/CoM.h>
#include <RBDyn/Coriolis.h>
#include <RBDyn/FD.h>
#include <RBDyn/Momentum.h>
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

/** Hard equality rows pinning a chosen block of the floating base's alphaD. */
class PinnedBaseAcceleration : public mc_solver::EqualityConstraintRobot
{
public:
  PinnedBaseAcceleration(unsigned int robotIndex, int nrDof, Eigen::Index rows)
  : mc_solver::EqualityConstraintRobot(robotIndex), A_(Eigen::MatrixXd::Zero(rows, nrDof)),
    b_(Eigen::VectorXd::Zero(rows))
  {
    A_.leftCols(rows).setIdentity();
  }

  void target(Eigen::Index row, double value) { b_(row) = value; }

  const Eigen::MatrixXd & A() const override { return A_; }
  void compute() override {}
  int maxEq() const override { return static_cast<int>(A_.rows()); }
  std::string nameEq() const override { return "PinnedBaseAcceleration"; }
  const Eigen::VectorXd & bEq() const override { return b_; }

private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

/** The quasi-static wheel-force bound r_i |t_i . f_i| <= tau_max, one row per wheel.
 *
 * Deliberately NOT part of the implementation: the report warns against
 * imposing eq:quasistatic-wheel-force alongside a complete wheel row, and
 * DYN-09 exists to show what doing so would cost. It lives in the test so the
 * comparison can be made without shipping the defect.
 *
 * The row is linear in lambda: t_i^T [G_0 | G_1] over the wheel's eight
 * generator multipliers, scaled by the wheel radius.
 */
class QuasiStaticWheelForceBound : public tasks::qp::ConstraintFunction<tasks::qp::GenInequality>
{
public:
  QuasiStaticWheelForceBound(const mc_solver::RollingContactDynamicsConstraint & dynamics,
                             std::vector<std::string> wheels,
                             double radius,
                             double torqueLimit)
  : dynamics_(dynamics), wheels_(std::move(wheels)), radius_(radius),
    lower_(Eigen::VectorXd::Constant(static_cast<Eigen::Index>(wheels_.size()), -torqueLimit)),
    upper_(Eigen::VectorXd::Constant(static_cast<Eigen::Index>(wheels_.size()), torqueLimit))
  {
  }

  void updateNrVars(const std::vector<rbd::MultiBody> &, const tasks::qp::SolverData & data) override
  {
    A_.setZero(static_cast<Eigen::Index>(wheels_.size()), data.nrVars());
  }

  void update(const std::vector<rbd::MultiBody> &,
              const std::vector<rbd::MultiBodyConfig> &,
              const tasks::qp::SolverData &) override
  {
    A_.setZero();
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto row = static_cast<Eigen::Index>(i);
      const Eigen::Vector3d rolling = dynamics_.geometryResult(wheels_[i]).rollingDirection;
      const Eigen::Index begin = dynamics_.lambdaBegin(wheels_[i]);
      for(unsigned int endpoint = 0; endpoint < 2; ++endpoint)
      {
        const auto & generators = dynamics_.forceGenerators(wheels_[i], endpoint);
        A_.block(row, begin + 4 * endpoint, 1, 4) = radius_ * rolling.transpose() * generators;
      }
    }
  }

  int maxGenInEq() const override { return static_cast<int>(wheels_.size()); }
  const Eigen::MatrixXd & AGenInEq() const override { return A_; }
  const Eigen::VectorXd & LowerGenInEq() const override { return lower_; }
  const Eigen::VectorXd & UpperGenInEq() const override { return upper_; }
  std::string nameGenInEq() const override { return "QuasiStaticWheelForceBound"; }
  std::string descGenInEq(const std::vector<rbd::MultiBody> &, int line) override
  {
    return nameGenInEq() + "/" + wheels_[static_cast<size_t>(line)];
  }

private:
  const mc_solver::RollingContactDynamicsConstraint & dynamics_;
  std::vector<std::string> wheels_;
  double radius_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
};

/** Run @p invalid, require it to be rejected, and require the message to name @p key. */
template<typename Callable>
void checkRejectionNames(Callable && invalid, const std::string & key, const std::string & what)
{
  bool rejected = false;
  try
  {
    invalid();
  }
  catch(const std::exception & error)
  {
    rejected = true;
    const std::string message = error.what();
    BOOST_CHECK_MESSAGE(message.find(key) != std::string::npos,
                        what << ": the rejection does not name '" << key << "', so nobody can act on it: " << message);
  }
  BOOST_CHECK_MESSAGE(rejected, what << ": an invalid configuration was accepted");
}

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

// ---------------------------------------------------------------------------
// The planar reduction alpha = P u of eq:reduced-velocity-coordinates, shared
// by the ORC-02, DYN-01, DYN-02 and DYN-06 cards.
// ---------------------------------------------------------------------------

/** The reduced-coordinate lift of eq:reduced-velocity-coordinates.
 *
 * u = [xi^Pi ; thetaDot_i ; deltaDot_i] and alpha = P u, with the planar twist
 * read in the chassis-aligned basis of eq:base-planar-twist. The base block of
 * P is the *lift* of that extraction, [E_Pi | n_Pi] resolved in the
 * floating-base frame, so P has orthonormal columns and is a genuine congruence
 * factor rather than a selection of rows whenever the chassis frame is not
 * already aligned with the contact plane.
 */
struct PlanarReduction
{
  Eigen::MatrixXd lift;
  /** [e_x, e_y, n] of eq:chassis-aligned-tangent-basis, resolved in the floating-base frame. */
  Eigen::Matrix3d bodyBasis = Eigen::Matrix3d::Identity();
  std::vector<Eigen::Index> driveDof;
  std::vector<Eigen::Index> steerDof;
  std::vector<Eigen::Vector2d> offsets;
  /** The naive alternative DYN-01 contrasts the congruence with: the alphaD
   * indices a row selection would keep, in the same order as u. */
  std::vector<Eigen::Index> selection;

  /** u from a whole-body velocity or acceleration; exact only in range(P). */
  Eigen::VectorXd reduce(const Eigen::VectorXd & whole) const { return lift.transpose() * whole; }
  Eigen::VectorXd expand(const Eigen::VectorXd & reduced) const { return lift * reduced; }
};

PlanarReduction planarReduction(const mc_rbdyn::Robot & robot,
                                const std::vector<mc_rbdyn::RollingContactDescription> & wheels,
                                const Eigen::Vector3d & terrainNormal)
{
  PlanarReduction out;
  const auto nrDof = static_cast<Eigen::Index>(robot.mb().nrDof());
  const auto nrWheels = static_cast<Eigen::Index>(wheels.size());
  const sva::PTransformd & X_0_fb = robot.mbc().bodyPosW[0];
  // X_0_b.rotation() maps world coordinates into the body frame, so this is
  // the {}^fb n_Pi of eq:terrain-basis-body-frame.
  const Eigen::Vector3d bodyNormal = X_0_fb.rotation() * terrainNormal.normalized();
  out.bodyBasis = mc_rbdyn::planarContactBasis(bodyNormal, Eigen::Vector3d::UnitX());
  out.lift.setZero(nrDof, 3 + 2 * nrWheels);
  out.lift.block<3, 2>(3, 0) = out.bodyBasis.leftCols<2>();
  out.lift.block<3, 1>(0, 2) = out.bodyBasis.col(2);
  out.selection = {3, 4, 2};
  for(Eigen::Index i = 0; i < nrWheels; ++i)
  {
    const auto & wheel = wheels[static_cast<size_t>(i)];
    const auto drive = static_cast<Eigen::Index>(
        robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheel.driveJoint))));
    const auto steer = static_cast<Eigen::Index>(
        robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheel.steeringJoint))));
    out.driveDof.push_back(drive);
    out.steerDof.push_back(steer);
    out.lift(drive, 3 + i) = 1.0;
    out.lift(steer, 3 + nrWheels + i) = 1.0;
    const Eigen::Vector3d offset =
        X_0_fb.rotation() * (robot.frame(wheel.carrierFrame).position().translation() - X_0_fb.translation());
    out.offsets.push_back(out.bodyBasis.leftCols<2>().transpose() * offset);
  }
  for(const auto drive : out.driveDof) { out.selection.push_back(drive); }
  for(const auto steer : out.steerDof) { out.selection.push_back(steer); }
  return out;
}

/** The planar wheel data the reduced rows are built from, read off the robot. */
std::vector<mc_rbdyn::PlanarWheel> measuredPlanarWheels(const mc_rbdyn::Robot & robot,
                                                        const std::vector<mc_rbdyn::RollingContactDescription> & wheels,
                                                        const PlanarReduction & reduction)
{
  std::vector<mc_rbdyn::PlanarWheel> planar(wheels.size());
  Eigen::VectorXd velocity(robot.mb().nrDof());
  rbd::paramToVector(robot.mbc().alpha, velocity);
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const auto steer = robot.jointIndexByName(wheels[i].steeringJoint);
    planar[i].offset = reduction.offsets[i];
    planar[i].steeringAngle = robot.mbc().q[steer][0];
    planar[i].steeringRate = velocity(reduction.steerDof[i]);
    planar[i].radius = wheels[i].radius;
    planar[i].spinSign = wheels[i].spinSign;
  }
  return planar;
}

/** Whole-body kinetic energy from the body twists and inertias.
 *
 * Independent of the mass matrix: it reads rbd::MultiBodyConfig::bodyVelB and
 * the URDF inertias directly, so DYN-01 can check M_red against it without
 * restating the congruence it is trying to validate.
 */
double kineticEnergy(const mc_rbdyn::Robot & robot)
{
  double energy = 0.0;
  for(size_t i = 0; i < robot.mbc().bodyVelB.size(); ++i)
  {
    const sva::MotionVecd & velocity = robot.mbc().bodyVelB[i];
    energy += 0.5 * velocity.dot(robot.mb().body(static_cast<int>(i)).inertia() * velocity);
  }
  return energy;
}

/** Put @p velocity on the robot and refresh every derived kinematic quantity. */
void setVelocity(mc_rbdyn::Robot & robot, const Eigen::VectorXd & velocity)
{
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.forwardKinematics();
  robot.forwardVelocity();
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

namespace
{

/** The two endpoints' world-frame generator matrices for one wheel, copied out. */
std::array<Eigen::Matrix<double, 3, Eigen::Dynamic>, 2> coneSnapshot(
    const mc_solver::RollingContactDynamicsConstraint & dynamics,
    const std::string & wheel)
{
  return {dynamics.forceGenerators(wheel, 0), dynamics.forceGenerators(wheel, 1)};
}

/** Worst deviation of n . c_k from its cone-consistent value over all generators.
 *
 * Each generator is (n + mu t_k) normalized with t_k orthogonal to n, so
 * n . c_k = 1 / sqrt(1 + mu^2) exactly in the frame the cone was built in. This
 * is the FRI-03 invariant, evaluated against whichever normal is passed.
 */
double normalProjectionDeviation(const std::array<Eigen::Matrix<double, 3, Eigen::Dynamic>, 2> & generators,
                                 const Eigen::Vector3d & normal,
                                 double friction)
{
  const double expected = 1.0 / std::sqrt(1.0 + friction * friction);
  double worst = 0.0;
  for(const auto & endpoint : generators)
  {
    for(Eigen::Index k = 0; k < endpoint.cols(); ++k)
    {
      worst = std::max(worst, std::abs(normal.dot(endpoint.col(k)) - expected));
    }
  }
  return worst;
}

/** Largest tangential-to-normal ratio, i.e. the Coulomb membership test itself. */
double worstConeRatio(const std::array<Eigen::Matrix<double, 3, Eigen::Dynamic>, 2> & generators,
                      const Eigen::Vector3d & normal)
{
  double worst = 0.0;
  for(const auto & endpoint : generators)
  {
    for(Eigen::Index k = 0; k < endpoint.cols(); ++k)
    {
      const Eigen::Vector3d direction = endpoint.col(k);
      const double normalComponent = normal.dot(direction);
      worst = std::max(worst, (direction - normalComponent * normal).norm() / normalComponent);
    }
  }
  return worst;
}

} // namespace

BOOST_AUTO_TEST_CASE(FrictionConeIsRebuiltWhenTheWheelSteersFRI02)
{
  // FRI-02, a staleness test. The cone *set* is rotationally symmetric about the
  // surface normal, so a cone cached at delta = 0 still passes every membership
  // check after the wheel steers. Only the map lambda -> f sees the defect, so
  // this asserts exact rotational equivariance of the generator matrix and,
  // separately, that the membership check is blind to the difference.
  constexpr double steering = 0.6;
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = fourSteeringWheels();
  const double friction = wheels[0].friction;
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  targetTask.target(Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof()));
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

  const Eigen::Vector3d normal = dynamics.terrainNormal();
  const auto stale = coneSnapshot(dynamics, "front_left");

  auto & robot = solver.robot(0);
  robot.mbc().q[robot.jointIndexByName("front_left_steer")][0] = steering;
  robot.forwardKinematics();
  robot.forwardVelocity();
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const auto fresh = coneSnapshot(dynamics, "front_left");

  // The steering axis is the surface normal (assumption A3), so steering by
  // delta must rotate the whole generator matrix by delta about that normal.
  const Eigen::Matrix3d rotation(Eigen::AngleAxisd(steering, normal));
  double equivarianceError = 0.0;
  double staleError = 0.0;
  for(size_t endpoint = 0; endpoint < fresh.size(); ++endpoint)
  {
    equivarianceError = std::max(equivarianceError, (fresh[endpoint] - rotation * stale[endpoint]).norm());
    staleError = std::max(staleError, (fresh[endpoint] - stale[endpoint]).norm());
  }
  BOOST_TEST_MESSAGE("FRI-02 equivariance error " << equivarianceError << ", stale-vs-fresh distance " << staleError);
  BOOST_CHECK_SMALL(equivarianceError, 1e-9);
  // The staleness assertion: a cone cached at delta = 0 is measurably wrong at
  // delta = 0.6, so the equivariance check above cannot be passed by a cache.
  BOOST_CHECK_GT(staleError, 0.5);

  // And the reason a membership test cannot stand in for this one: the stale
  // cone passes the Coulomb check against the same normal just as well.
  BOOST_CHECK_CLOSE(worstConeRatio(stale, normal), friction, 1e-8);
  BOOST_CHECK_CLOSE(worstConeRatio(fresh, normal), friction, 1e-8);

  solver.removeConstraintSet(dynamics);
  solver.removeTask(&targetTask);
}

BOOST_AUTO_TEST_CASE(FrictionConeIsRebuiltWhenTheTerrainNormalChangesFRI03)
{
  // FRI-03, the other half of the staleness pair. Each generator satisfies
  // n . c_k = 1 / sqrt(1 + mu^2) in the frame it was built in; a cone left at
  // the old normal breaks that at first order in the tilt. This is what
  // distinguishes a normal-update fault from the steering-update fault FRI-02
  // targets.
  constexpr double tilt = 20.0 * mc_rtc::constants::PI / 180.0;
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto wheels = fourSteeringWheels();
  const double friction = wheels[0].friction;
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  TargetAccelerationTask targetTask(solver.robot(0).mb(), 0);
  targetTask.target(Eigen::VectorXd::Zero(solver.robot(0).mb().nrDof()));
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

  const Eigen::Vector3d flat = dynamics.terrainNormal();
  const auto stale = coneSnapshot(dynamics, "front_left");
  BOOST_CHECK_SMALL(normalProjectionDeviation(stale, flat, friction), 1e-12);

  // Tilt about the wheel's own lateral axis, so the rolling generators lie in
  // the tilt plane and the stale error is the full first-order one.
  const Eigen::Vector3d lateral = dynamics.geometryResult("front_left").lateralDirection;
  const Eigen::Vector3d ramp = (Eigen::AngleAxisd(tilt, lateral) * flat).normalized();
  dynamics.terrainNormal(ramp);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const auto fresh = coneSnapshot(dynamics, "front_left");

  const double freshDeviation = normalProjectionDeviation(fresh, ramp, friction);
  const double staleDeviation = normalProjectionDeviation(stale, ramp, friction);
  const double firstOrder = (1.0 - std::cos(tilt)) / std::sqrt(1.0 + friction * friction);
  BOOST_TEST_MESSAGE("FRI-03 fresh deviation " << freshDeviation << ", stale deviation " << staleDeviation
                                               << ", first-order floor " << firstOrder);
  BOOST_CHECK_SMALL(freshDeviation, 1e-12);
  BOOST_CHECK_GE(staleDeviation, firstOrder);
  // The rebuilt cone is aligned with the new plane, the cached one is not.
  BOOST_CHECK_SMALL(std::abs(worstConeRatio(fresh, ramp) - friction), 1e-8);
  BOOST_CHECK_GT(std::abs(worstConeRatio(stale, ramp) - friction), 1e-2);

  solver.removeConstraintSet(dynamics);
  solver.removeTask(&targetTask);
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

/** What one solve says about the effort available on each row of the dynamics.
 *
 * In the Tasks assembly the equation of motion is written as the general
 * inequality L <= H alphaD - J^T G lambda <= U with L = tauMin - C and
 * U = tauMax - C, so the width U(k) - L(k) is exactly the actuation the
 * coordinate of row k may absorb. A floating-base coordinate is unactuated: its
 * width is zero, and the actuation map has an identically zero row there.
 */
struct BaseRowEffort
{
  bool baseRowsExactlyEqual = false;
  double planeNormalWidth = 0.0;
  double worstBaseWidth = 0.0;
  double steeringWidth = 0.0;
  double driveWidth = 0.0;
  double baseTorque = 0.0;
  double yawAcceleration = 0.0;
};

} // namespace

BOOST_AUTO_TEST_CASE(SteeringTorqueCannotEnterTheChassisYawRowDYN05)
{
  // DYN-05. The actuation map has identically zero rows for the six
  // floating-base coordinates, so no joint torque -- a steering torque in
  // particular -- can appear in the plane-normal base row. Asserted as exact
  // equality of the two bound vectors on those rows, not as a threshold.
  constexpr Eigen::Index baseDof = 6;
  // The free-flyer alphaD block is [wx, wy, wz, vx, vy, vz] in the body frame,
  // and the chassis is level here, so row 2 is the plane-normal base row.
  constexpr Eigen::Index planeNormalRow = 2;

  auto measure = [](bool infTorque)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    auto & robot = solver.robot(0);
    BOOST_REQUIRE_SMALL((robot.posW().rotation() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    auto wheels = fourSteeringWheels();
    mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels, infTorque);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
    // A chassis yaw acceleration far beyond what the contacts can earn, so a
    // base row that could absorb effort would visibly do so.
    target(planeNormalRow) = 50.0;
    targetTask.target(target);
    solver.addTask(&targetTask);
    solver.addConstraintSet(dynamics);
    BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

    const Eigen::VectorXd & lower = dynamics.motionConstr().LowerGenInEq();
    const Eigen::VectorXd & upper = dynamics.motionConstr().UpperGenInEq();
    BOOST_REQUIRE_EQUAL(lower.size(), robot.mb().nrDof());
    BaseRowEffort out;
    out.baseRowsExactlyEqual = true;
    for(Eigen::Index k = 0; k < baseDof; ++k)
    {
      out.baseRowsExactlyEqual = out.baseRowsExactlyEqual && lower(k) == upper(k);
      out.worstBaseWidth = std::max(out.worstBaseWidth, upper(k) - lower(k));
    }
    out.planeNormalWidth = upper(planeNormalRow) - lower(planeNormalRow);
    const auto steeringDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
    const auto driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_drive")));
    out.steeringWidth = upper(steeringDof) - lower(steeringDof);
    out.driveWidth = upper(driveDof) - lower(driveDof);
    dynamics.motionConstr().computeTorque(solver.solver().alphaDVec(), solver.solver().lambdaVec());
    out.baseTorque = dynamics.motionConstr().torque().head(baseDof).lpNorm<Eigen::Infinity>();
    out.yawAcceleration = solver.solver().alphaDVec(0)(planeNormalRow);
    solver.removeConstraintSet(dynamics);
    solver.removeTask(&targetTask);
    return out;
  };

  const BaseRowEffort shipped = measure(false);
  BOOST_TEST_MESSAGE("DYN-05 shipped: base rows equal=" << shipped.baseRowsExactlyEqual << ", plane-normal width "
                                                        << shipped.planeNormalWidth << ", steering width "
                                                        << shipped.steeringWidth << ", drive width "
                                                        << shipped.driveWidth << ", |tau_base| " << shipped.baseTorque
                                                        << ", yaw alphaD " << shipped.yawAcceleration);
  // Exact equality, not a threshold: no effort at all is admitted on any of the
  // six base rows, and in particular none on the plane-normal one.
  BOOST_CHECK(shipped.baseRowsExactlyEqual);
  BOOST_CHECK_EQUAL(shipped.planeNormalWidth, 0.0);
  BOOST_CHECK_EQUAL(shipped.worstBaseWidth, 0.0);
  // Non-vacuity: the actuated rows of the same vector are wide, at exactly the
  // URDF effort limits (25 Nm steering, 35 Nm drive, two-sided).
  BOOST_CHECK_CLOSE(shipped.steeringWidth, 50.0, 1e-9);
  BOOST_CHECK_CLOSE(shipped.driveWidth, 70.0, 1e-9);
  // The reconstructed base torque is therefore zero however hard the yaw is
  // demanded: the yaw acceleration has to be earned through contact forces.
  BOOST_CHECK_SMALL(shipped.baseTorque, 1e-8);

  // The paired mutant: infTorque hands the same six rows an unbounded effort
  // interval, which is what a nonzero actuation row on the floating base looks
  // like here. The base then absorbs the demand directly.
  const BaseRowEffort unbounded = measure(true);
  BOOST_TEST_MESSAGE("DYN-05 infTorque mutant: plane-normal width " << unbounded.planeNormalWidth << ", |tau_base| "
                                                                    << unbounded.baseTorque << ", yaw alphaD "
                                                                    << unbounded.yawAcceleration);
  BOOST_CHECK(std::isinf(unbounded.planeNormalWidth));
  BOOST_CHECK_GT(unbounded.baseTorque, 1.0);
  BOOST_CHECK_GT(std::abs(unbounded.yawAcceleration), std::abs(shipped.yawAcceleration));
}

BOOST_AUTO_TEST_CASE(PredictedRateRowsCarryNoProportionalStabilizationROW11)
{
  // ROW-11, adapted, and the adaptation is the point.
  //
  // The testcard states an asymmetry between the report's two *planar* QPs:
  // eq:differential-drive-qp is written with -Kp times the measured residual on
  // each of its three rows, and eq:four-steering-wheel-qp has no Kp at all.
  // That asymmetry does NOT carry over to mc_rtc's whole-body form, and
  // reproducing it would be a defect rather than a fix: the general
  // acceleration-level rows of the same report --
  // eq:homogeneous-rolling-acceleration and
  // eq:geometric-wheel-acceleration-task, which are what this implementation
  // assembles -- carry -Kp G u for BOTH chassis. T2's planar QP omits it only
  // because it is stated for the ideal, zero-residual case. A four-steering
  // chassis with a nonzero measured residual needs exactly the stabilization a
  // differential one does, or the residual is frozen instead of removed
  // (ROW-03). So the geometric rows are asserted to be symmetric here.
  //
  // What is true, and is the refactor hazard worth pinning, is the T2-specific
  // half: the predicted rotating-rate rows -- the w_thetaDot and w_deltaDot
  // terms only a four-steering chassis emits -- carry no proportional
  // stabilization. Their right-hand side is (reference - measured) and Kp
  // leaves it bit-identical.
  constexpr double dt = 0.005;
  constexpr double gain = 20.0;

  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, dt);
  auto & robot = solver.robot(0);
  setIncompatibleFourWheelState(robot); // a state with a nonzero measured residual on every row

  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
  solver.addConstraintSet(rolling);
  rolling.rotatingRateReference("front_left", 2.5, -0.4);
  rolling.update(solver);

  // Default options keep the geometric rows hard, so their right-hand side is
  // unscaled and Kp can be read straight off it.
  const Eigen::MatrixXd hardMatrix = rolling.hardMatrix();
  const Eigen::VectorXd hardAtZeroGain = rolling.hardRhs();
  const Eigen::MatrixXd softMatrix = rolling.softMatrix();
  const Eigen::VectorXd softAtZeroGain = rolling.softRhs();
  const auto hardLabels = rolling.hardRowLabels();
  const auto softLabels = rolling.softRowLabels();
  BOOST_REQUIRE_GT(hardAtZeroGain.size(), 0);
  BOOST_REQUIRE_GT(softAtZeroGain.size(), 0);
  for(const auto & label : softLabels)
  {
    BOOST_REQUIRE_MESSAGE(label.find("-rate") != std::string::npos, "unexpected soft row " << label);
  }

  rolling.velocityGain(gain);
  rolling.update(solver);
  BOOST_REQUIRE(rolling.hardRowLabels() == hardLabels);
  BOOST_REQUIRE(rolling.softRowLabels() == softLabels);
  // Kp moves no coefficient, only right-hand sides.
  BOOST_CHECK_SMALL((rolling.hardMatrix() - hardMatrix).norm(), 1e-15);
  BOOST_CHECK_SMALL((rolling.softMatrix() - softMatrix).norm(), 1e-15);

  // The predicted-rate rows are bit-identical: no Kp anywhere in them.
  for(Eigen::Index row = 0; row < softAtZeroGain.size(); ++row)
  {
    BOOST_CHECK_EQUAL(rolling.softRhs()(row), softAtZeroGain(row));
  }
  // Non-vacuity: those right-hand sides are not trivially zero.
  BOOST_CHECK_GT(softAtZeroGain.lpNorm<Eigen::Infinity>(), 1e-3);

  // Every geometric row, by contrast, moved by exactly -Kp times its measured
  // residual -- the symmetric behaviour this implementation deliberately keeps.
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
  const std::array<std::string, 3> axes = {"longitudinal", "lateral", "normal"};
  double worstGeometricError = 0.0;
  // The largest, not the smallest: the normal rows legitimately have a zero
  // measured residual on flat ground, so Kp moves nothing there.
  double largestGeometricShift = 0.0;
  for(Eigen::Index row = 0; row < hardAtZeroGain.size(); ++row)
  {
    const auto & label = hardLabels[static_cast<size_t>(row)];
    const auto wheel = std::distance(names.begin(), std::find(names.begin(), names.end(),
                                                              label.substr(0, label.find('/'))));
    const auto axis = std::distance(axes.begin(), std::find(axes.begin(), axes.end(),
                                                            label.substr(label.find('/') + 1)));
    BOOST_REQUIRE_LT(wheel, 4);
    BOOST_REQUIRE_LT(axis, 3);
    const double residual = rolling.geometryResults()[static_cast<size_t>(wheel)].velocityResidual(axis);
    const double shift = rolling.hardRhs()(row) - hardAtZeroGain(row);
    worstGeometricError = std::max(worstGeometricError, std::abs(shift + gain * residual));
    largestGeometricShift = std::max(largestGeometricShift, std::abs(shift));
  }
  BOOST_TEST_MESSAGE("ROW-11 geometric rows: worst |shift + Kp * residual| " << worstGeometricError
                                                                             << ", largest |shift| "
                                                                             << largestGeometricShift);
  BOOST_CHECK_SMALL(worstGeometricError, 1e-12);
  BOOST_CHECK_GT(largestGeometricShift, 1e-6);
  solver.removeConstraintSet(rolling);

  // And the same on the differential chassis, so the symmetry is asserted on
  // both sides rather than inferred from one.
  auto differentialRobots = loadDifferentialRobot();
  mc_solver::TasksQPSolver differentialSolver(differentialRobots, dt);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(differentialSolver.robot(0).mb().nrDof());
  velocity(2) = 0.2;
  velocity(3) = 0.35;
  velocity(4) = 0.12;
  rbd::vectorToParam(velocity, differentialSolver.robot(0).mbc().alpha);
  differentialSolver.robot(0).forwardVelocity();
  mc_solver::RollingContactConstraintOptions differentialOptions;
  differentialOptions.velocityGain = 0.0;
  differentialOptions.differentialPlanar = true;
  mc_solver::RollingContactConstraint differential(differentialSolver.robots(), 0, differentialWheels(),
                                                   differentialOptions);
  differentialSolver.addConstraintSet(differential);
  differential.update(differentialSolver);
  const Eigen::VectorXd differentialAtZeroGain = differential.hardRhs();
  differential.velocityGain(gain);
  differential.update(differentialSolver);
  const double differentialShift =
      (differential.hardRhs() - differentialAtZeroGain).lpNorm<Eigen::Infinity>();
  BOOST_TEST_MESSAGE("ROW-11 differential geometric rows shift by " << differentialShift << " at Kp = " << gain);
  BOOST_CHECK_GT(differentialShift, 1e-6);
  differentialSolver.removeConstraintSet(differential);
}

BOOST_AUTO_TEST_CASE(QuasiStaticTorqueBoundDuplicatesTheWheelRowDYN09)
{
  // DYN-09. The report warns that eq:quasistatic-wheel-force must not be
  // imposed alongside a complete wheel row. This shows why, on the shipped
  // whole-body QP: the quasi-static inequality neglects the wheel's rotational
  // inertia, so it removes points the full dynamics admits.
  //
  // The state is built so the removal is unambiguous. The chassis angular
  // accelerations and its forward acceleration are pinned hard, which fixes the
  // total longitudinal contact force at m * a_x through the unactuated base
  // rows, and a_x is chosen so that total exceeds what four quasi-static bounds
  // at the drive torque limit can supply. The full row still admits it, because
  // the wheels are free to spin down.
  constexpr double radius = 0.2;
  constexpr double driveTorqueLimit = 35.0; // the URDF effort limit of *_drive
  const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};

  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  const double mass = robot.mass();
  // Twice what the quasi-static bounds could ever supply across four wheels.
  const double quasiStaticCeiling = 4.0 * driveTorqueLimit / radius;
  const double forwardAcceleration = 2.0 * quasiStaticCeiling / mass;

  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  // Rows 0..2 are the base angular accelerations, row 3 the forward one; the
  // vertical and lateral base rows stay free so the normal force can grow.
  PinnedBaseAcceleration pinned(0, robot.mb().nrDof(), 4);
  pinned.target(3, forwardAcceleration);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 1.0);
  solver.addConstraintSet(dynamics);
  solver.addConstraint(&pinned);
  solver.addTask(&posture);

  // The full wheel row alone: feasible. And the shipped assembly carries no
  // quasi-static duplicate -- the dynamics is the only general inequality in
  // the problem, and it contributes exactly one complete row per degree of
  // freedom.
  BOOST_CHECK_EQUAL(solver.solver().nrGenInequalityConstraints(), 1);
  BOOST_CHECK_EQUAL(dynamics.motionConstr().maxGenInEq(), robot.mb().nrDof());
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd lambda = solver.solver().lambdaVec();
  dynamics.motionConstr().computeTorque(solver.solver().alphaDVec(), solver.solver().lambdaVec());
  const Eigen::VectorXd torque = dynamics.motionConstr().torque();

  double totalLongitudinal = 0.0;
  double worstExcess = 0.0;
  for(size_t i = 0; i < names.size(); ++i)
  {
    const auto & result = dynamics.geometryResult(names[i]);
    const auto forces = dynamics.endpointForces(names[i], lambda.segment<8>(static_cast<Eigen::Index>(8 * i)));
    const double longitudinal = result.rollingDirection.dot(forces[0] + forces[1]);
    const auto driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(names[i] + "_drive")));
    totalLongitudinal += longitudinal;
    // The quasi-static inequality |t . f| <= |tau| / r, evaluated at a point the
    // full dynamics admits.
    worstExcess = std::max(worstExcess, std::abs(longitudinal) - std::abs(torque(driveDof)) / radius);
  }
  BOOST_TEST_MESSAGE("DYN-09 demanded m*a_x = " << mass * forwardAcceleration << " N, realised total longitudinal "
                                                << totalLongitudinal << " N, quasi-static ceiling "
                                                << quasiStaticCeiling << " N, worst per-wheel excess of |t.f| over "
                                                << "|tau|/r: " << worstExcess << " N");
  // The base rows are unactuated, so the pinned chassis acceleration is paid for
  // entirely by contact force.
  BOOST_CHECK_CLOSE(totalLongitudinal, mass * forwardAcceleration, 1e-6);
  // And the admitted point violates the quasi-static inequality: the gap is the
  // wheel rotational inertia the inequality drops.
  BOOST_CHECK_GT(worstExcess, 1.0);

  // Now impose the quasi-static bound alongside the complete row. The generator
  // matrices are current from the solve above and the state has not moved.
  QuasiStaticWheelForceBound quasiStatic(dynamics, {names.begin(), names.end()}, radius, driveTorqueLimit);
  solver.addConstraint(&quasiStatic);
  BOOST_CHECK_EQUAL(solver.solver().nrGenInequalityConstraints(), 2);
  BOOST_CHECK(!solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  solver.removeConstraint(&quasiStatic);
  // Removing it again restores feasibility, so the failure above is the bound
  // and not an accumulated solver state.
  BOOST_CHECK(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));

  solver.removeTask(&posture);
  solver.removeConstraint(&pinned);
  solver.removeConstraintSet(dynamics);
}

BOOST_AUTO_TEST_CASE(RejectionContractNamesTheOffendingKeyQP10)
{
  // QP-10. Invalid model data is rejected at assembly rather than propagated,
  // and each rejection names the key that caused it -- a rejection nobody can
  // act on is barely better than silence.
  auto robots = loadRollingRobots();
  mc_solver::TasksQPSolver solver(robots, 0.005);

  // The configuration path, which is where a user actually meets these.
  auto configuredAs = [&](const std::string & key, auto value)
  {
    return [&solver, key, value]()
    {
      auto config = rollingConfiguration("rollingContact", differentialWheels());
      config.add(key, value);
      mc_solver::ConstraintSetLoader::load(solver, config);
    };
  };

  // Removed options: rejected rather than ignored, since ignoring them would
  // silently restore four hard lateral rows on a four-wheel chassis.
  checkRejectionNames(configuredAs("steeringPlanar", true), "steeringPlanar", "steeringPlanar");
  checkRejectionNames(configuredAs("steeringPlanar", true), "softLateralRows", "steeringPlanar replacement");
  checkRejectionNames(configuredAs("steeringPlanarWheels", std::vector<std::string>{"left", "right"}),
                      "steeringPlanarWheels", "steeringPlanarWheels");

  // Non-positive-definite or non-finite weights.
  for(const double weight : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()})
  {
    checkRejectionNames(configuredAs("lateralSlackWeight", weight), "lateralSlackWeight", "lateralSlackWeight");
    checkRejectionNames(configuredAs("rollingWeight", weight), "rollingWeight", "rollingWeight");
  }
  for(const double weight : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()})
  {
    checkRejectionNames(configuredAs("rollingRateWeight", weight), "rollingRateWeight", "rollingRateWeight");
    checkRejectionNames(configuredAs("steeringRateWeight", weight), "steeringRateWeight", "steeringRateWeight");
  }
  checkRejectionNames(configuredAs("velocityGain", -1.0), "velocityGain", "velocityGain");
  checkRejectionNames(configuredAs("velocityGain", std::numeric_limits<double>::infinity()), "velocityGain",
                      "infinite velocityGain");
  checkRejectionNames(configuredAs("terrainNormal", Eigen::Vector3d{0.0, 0.0, 0.0}), "terrainNormal",
                      "terrainNormal");
  const double notANumber = std::numeric_limits<double>::quiet_NaN();
  checkRejectionNames(configuredAs("terrainNormal", Eigen::Vector3d{notANumber, notANumber, notANumber}),
                      "terrainNormal", "non-finite terrainNormal");
  // differentialPlanar is only meaningful on a two-wheel chassis.
  checkRejectionNames(
      [&]()
      {
        auto four = loadFourSteeringRobot();
        mc_solver::RollingContactConstraintOptions differential;
        differential.differentialPlanar = true;
        mc_solver::RollingContactConstraint(*four, 0, fourSteeringWheels(), differential);
      },
      "differentialPlanar", "differentialPlanar on four wheels");
  for(const double epsilon : {-1.0, std::numeric_limits<double>::infinity()})
  {
    checkRejectionNames(
        [&]()
        {
          auto config = rollingConfiguration("rollingContactDynamics", differentialWheels());
          config.add("generatorRegularization", epsilon);
          mc_solver::ConstraintSetLoader::load(solver, config);
        },
        "generatorRegularization", "generatorRegularization");
  }
  checkRejectionNames(configuredAs("longitudinal", std::string{"medium"}), "longitudinal", "longitudinal");

  // A zero wheel radius, at both constraint sets.
  for(const std::string type : {"rollingContact", "rollingContactDynamics"})
  {
    checkRejectionNames(
        [&]()
        {
          auto config = rollingConfiguration(type, differentialWheels());
          config("wheels")[0].add("radius", 0.0);
          mc_solver::ConstraintSetLoader::load(solver, config);
        },
        "radius", type + " radius");
  }

  // A steering joint that is also its own drive joint.
  checkRejectionNames(
      [&]()
      {
        auto four = loadFourSteeringRobot();
        auto steeringWheels = fourSteeringWheels();
        steeringWheels[0].steeringJoint = steeringWheels[0].driveJoint;
        mc_solver::RollingContactConstraint(*four, 0, steeringWheels);
      },
      "steeringJoint", "steeringJoint == driveJoint");

  // Non-finite values that would otherwise reach the QP through a setter.
  auto four = loadFourSteeringRobot();
  mc_solver::RollingContactConstraintOptions options;
  options.softLateralRows = true;
  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint rolling(*four, 0, fourSteeringWheels(), options);
  checkRejectionNames(
      [&]() { rolling.rotatingRateReference("front_left", std::numeric_limits<double>::infinity(), 0.0); },
      "rate reference", "infinite rolling-rate reference");
  checkRejectionNames(
      [&]()
      { rolling.rotatingRateReference("front_left", 0.0, std::numeric_limits<double>::quiet_NaN()); },
      "rate reference", "non-finite steering-rate reference");
  checkRejectionNames([&]() { rolling.activation("front_left", std::numeric_limits<double>::quiet_NaN()); },
                      "activation", "non-finite activation");
  checkRejectionNames([&]() { rolling.terrainNormal(Eigen::Vector3d::Zero()); }, "terrainNormal",
                      "degenerate terrainNormal setter");
  mc_solver::RollingContactDynamicsConstraint dynamics(*four, 0, solver.dt(), fourSteeringWheels());
  checkRejectionNames([&]() { dynamics.terrainNormal(Eigen::Vector3d::Zero()); }, "terrainNormal",
                      "degenerate dynamics terrainNormal setter");
  checkRejectionNames([&]() { dynamics.terrainNormal(Eigen::Vector3d{notANumber, notANumber, notANumber}); },
                      "terrainNormal", "non-finite dynamics terrainNormal setter");

  // A duplicate wheel name names the duplicate.
  checkRejectionNames(
      [&]()
      {
        auto duplicated = fourSteeringWheels();
        duplicated[1].name = duplicated[0].name;
        mc_solver::RollingContactConstraint(*four, 0, duplicated);
      },
      "front_left", "duplicate wheel name");

  // And no command is emitted: every rejection above happened at assembly, so
  // the reference the last valid call stored is still what the object holds.
  BOOST_CHECK_EQUAL(rolling.rollingRateReference("front_left"), 0.0);
  BOOST_CHECK_EQUAL(rolling.steeringRateReference("front_left"), 0.0);
  BOOST_CHECK_EQUAL(rolling.activation("front_left"), 1.0);
  BOOST_CHECK_SMALL((rolling.terrainNormal() - Eigen::Vector3d::UnitZ()).norm(), 1e-12);
}

BOOST_AUTO_TEST_CASE(YawingChassisConsumesNoSteeringRateBudgetBND02)
{
  // BND-02. delta_i is chassis relative, so the steering-rate budget is
  // independent of the chassis yaw rate. Two halves, both discriminating.
  constexpr double dt = 0.005;
  const std::array<double, 4> lockedAngles = {0.31, -0.22, 0.47, -0.13};

  // Half one, structural and exact: the assembled predicted-steering-rate row
  // carries deltaDot + dt * deltaDdot and nothing else. Its coefficient on the
  // base yaw acceleration is exactly zero, as is every coefficient except the
  // steering joint's own.
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, dt);
    auto & robot = solver.robot(0);
    mc_solver::RollingContactConstraintOptions options;
    options.softLateralRows = true;
    options.trackRotatingRates = true;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, fourSteeringWheels(), options);
    solver.addConstraintSet(rolling);
    rolling.update(solver);

    const auto steeringDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
    const Eigen::RowVectorXd row = rolling.softMatrix().row(softRowIndex(rolling, "front_left/steering-rate"));
    BOOST_REQUIRE_EQUAL(row.size(), robot.mb().nrDof());
    // Row 2 of the free-flyer block is the plane-normal (yaw) base acceleration.
    BOOST_CHECK_EQUAL(row(2), 0.0);
    BOOST_CHECK_GT(std::abs(row(steeringDof)), 0.0);
    for(Eigen::Index k = 0; k < row.size(); ++k)
    {
      if(k == steeringDof) { continue; }
      BOOST_CHECK_EQUAL(row(k), 0.0);
    }
    solver.removeConstraintSet(rolling);
  }

  // Half two, behavioural: with the modules locked, the reachable steering
  // acceleration is the joint-velocity box (vu - deltaDot) / dt, and it does not
  // move when the chassis yaws. The mutant reads the absolute heading rate
  // instead, i.e. deltaDot = omega, and loses exactly omega / dt of the box.
  auto reachableSteeringAcceleration = [&](double baseYawRate, double measuredSteeringRate)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, dt);
    auto & robot = solver.robot(0);
    for(size_t i = 0; i < 4; ++i)
    {
      const std::array<std::string, 4> names = {"front_left", "front_right", "rear_left", "rear_right"};
      robot.mbc().q[robot.jointIndexByName(names[i] + "_steer")][0] = lockedAngles[i];
    }
    robot.forwardKinematics();
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
    velocity(2) = baseYawRate;
    const auto steeringDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
    velocity(steeringDof) = measuredSteeringRate;
    rbd::vectorToParam(velocity, robot.mbc().alpha);
    robot.forwardVelocity();

    mc_solver::KinematicsConstraint kinematics(solver.robots(), 0, dt, {0.1, 0.01, 0.5}, 0.5);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(robot.mb().nrDof());
    target(steeringDof) = 1e6; // demand far past the box, so the box is what is measured
    targetTask.target(target);
    // No contact has been declared, so the Tasks decision layout has not been
    // sized yet; KinematicsConstraint assumes it has.
    solver.updateNrVars();
    solver.addConstraintSet(kinematics);
    solver.addTask(&targetTask);
    BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
    const double reached = solver.solver().alphaDVec(0)(steeringDof);
    solver.removeTask(&targetTask);
    solver.removeConstraintSet(kinematics);
    return reached;
  };

  // The velocity limit is 8 rad/s in the URDF, halved by the standard
  // velocityPercent of 0.5, so a still hinge may accelerate to 4 / dt.
  const double budget = 4.0 / dt;
  const double atRest = reachableSteeringAcceleration(0.0, 0.0);
  BOOST_TEST_MESSAGE("BND-02 reachable steering acceleration at rest: " << atRest << " (box " << budget << ")");
  BOOST_REQUIRE_CLOSE(atRest, budget, 1e-6); // the velocity box, not some other limit, is what binds

  for(const double omega : {1.0, 4.0})
  {
    const double yawing = reachableSteeringAcceleration(omega, 0.0);
    const double absolute = reachableSteeringAcceleration(0.0, omega);
    BOOST_TEST_MESSAGE("BND-02 omega=" << omega << ": chassis-relative reach " << yawing
                                       << ", absolute-convention reach " << absolute
                                       << ", budget consumed by the mutant "
                                       << (100.0 * (atRest - absolute) / atRest) << "%");
    // The shipped reading is bit-identical to the omega = 0 case: no budget spent.
    BOOST_CHECK_EQUAL(yawing, atRest);
    // The mutant loses exactly omega / dt of the box.
    BOOST_CHECK_CLOSE(atRest - absolute, omega / dt, 1e-6);
  }
  // At omega equal to the limit the absolute convention has nothing left at all.
  BOOST_CHECK_SMALL(reachableSteeringAcceleration(0.0, 4.0), 1e-9);
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

// ===========================================================================
// Layer C. The reduced dynamics congruence and the two corrected invariants.
// ===========================================================================

BOOST_AUTO_TEST_CASE(ReducedInertiaIsACongruenceNotARowSliceDYN01)
{
  // DYN-01. M_red = P^T M P, checked against an oracle that never touches M:
  // the kinetic energy summed from the body twists and the URDF inertias. A row
  // selection reproduces neither that energy nor, off a chassis-aligned plane,
  // the symmetric positive-definite structure.
  auto robots = loadFourSteeringRobot();
  auto & robot = robots->robot(0);
  const auto wheels = fourSteeringWheels();
  setSteeringAngles(robot, uncoordinatedSteeringAngles());

  rbd::ForwardDynamics forwardDynamics(robot.mb());
  forwardDynamics.computeH(robot.mb(), robot.mbc());
  const Eigen::MatrixXd inertia = forwardDynamics.H();
  BOOST_REQUIRE_EQUAL(inertia.rows(), robot.mb().nrDof());

  std::mt19937 generator(20260907u);
  std::uniform_real_distribution<double> draw(-1.5, 1.5);

  // The flat, level case first: mc_rtc's floating-base alphaD is already a body
  // twist, so P degenerates to a selection there and the congruence coincides
  // with the principal submatrix. That is a property of this codebase's
  // coordinates, not of the reduction, so it is pinned rather than assumed.
  // The tilted-plane case is the one where the two genuinely differ.
  const std::array<Eigen::Vector3d, 2> normals = {Eigen::Vector3d::UnitZ(),
                                                  Eigen::Vector3d(0.35, -0.2, 1.0).normalized()};
  for(const auto & normal : normals)
  {
    const bool chassisAligned = normal.isApprox(Eigen::Vector3d::UnitZ());
    const auto reduction = planarReduction(robot, wheels, normal);
    const Eigen::MatrixXd reduced = reduction.lift.transpose() * inertia * reduction.lift;
    const auto size = reduced.rows();
    BOOST_REQUIRE_EQUAL(size, 11);

    BOOST_CHECK_SMALL((reduced - reduced.transpose()).lpNorm<Eigen::Infinity>(), 1e-9 * inertia.norm());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(reduced);
    BOOST_TEST_MESSAGE("DYN-01 normal [" << normal.transpose() << "] smallest reduced eigenvalue "
                                         << spectrum.eigenvalues().minCoeff());
    BOOST_CHECK_GT(spectrum.eigenvalues().minCoeff(), 0.0);

    // The oracle: 0.5 u^T M_red u must be the physical kinetic energy of the
    // lifted motion, read from bodyVelB and the body inertias.
    for(int sample = 0; sample < 8; ++sample)
    {
      Eigen::VectorXd reducedVelocity(size);
      for(Eigen::Index i = 0; i < size; ++i) { reducedVelocity(i) = draw(generator); }
      setVelocity(robot, reduction.expand(reducedVelocity));
      const double energy = kineticEnergy(robot);
      const double quadratic = 0.5 * reducedVelocity.dot(reduced * reducedVelocity);
      BOOST_CHECK_MESSAGE(std::abs(quadratic - energy) <= 1e-9 * (1.0 + std::abs(energy)),
                          "DYN-01 reduced kinetic energy " << quadratic << " against the body-twist oracle " << energy);
    }

    // The row selection the card contrasts with.
    Eigen::MatrixXd sliced(size, size);
    for(Eigen::Index i = 0; i < size; ++i)
    {
      for(Eigen::Index j = 0; j < size; ++j) { sliced(i, j) = inertia(reduction.selection[static_cast<size_t>(i)],
                                                                     reduction.selection[static_cast<size_t>(j)]); }
    }
    const double gap = (sliced - reduced).lpNorm<Eigen::Infinity>();
    BOOST_TEST_MESSAGE("DYN-01 normal [" << normal.transpose() << "] congruence-minus-selection " << gap);
    if(chassisAligned) { BOOST_CHECK_SMALL(gap, 1e-9 * inertia.norm()); }
    else
    {
      // Off the chassis-aligned plane the selection is a different matrix, and
      // it gets the energy wrong: that is the assertion with teeth.
      BOOST_CHECK_GT(gap, 1e-3 * inertia.norm());
      Eigen::VectorXd reducedVelocity = Eigen::VectorXd::Zero(size);
      reducedVelocity.head<3>() << 0.8, -0.4, 0.6;
      setVelocity(robot, reduction.expand(reducedVelocity));
      const double energy = kineticEnergy(robot);
      BOOST_CHECK_GT(std::abs(0.5 * reducedVelocity.dot(sliced * reducedVelocity) - energy), 1e-3 * (1.0 + energy));
    }
  }
}

BOOST_AUTO_TEST_CASE(ReducedBiasCarriesTheVelocityProductTermDYN02)
{
  // DYN-02, adapted to the whole-body form and reporting the carry-over.
  //
  // h_red = P^T (M Pdot u + h). In mc_rtc's coordinates the first term is
  // identically zero along any planar motion, because RBDyn's floating-base
  // alpha is already the *body* twist of eq:base-body-velocity and the
  // chassis-aligned basis is constant in the floating-base frame. Pdot = 0 is
  // therefore measured here, not assumed. What the card is really about --
  // that the velocity-product part is not optional, because it is where the
  // Coriolis and centripetal terms of the rotating chassis-aligned basis live
  // -- transfers intact: in the whole-body form those terms sit inside the
  // full nonlinear-effects vector h(q, alpha), and using gravity alone is the
  // error the card warns about.
  auto robots = loadFourSteeringRobot();
  auto & robot = robots->robot(0);
  const auto wheels = fourSteeringWheels();
  setSteeringAngles(robot, uncoordinatedSteeringAngles());
  const auto reduction = planarReduction(robot, wheels, Eigen::Vector3d::UnitZ());

  // A turning motion, so omega_Pi != 0 as the card's regime requires.
  Eigen::VectorXd reducedVelocity = Eigen::VectorXd::Zero(reduction.lift.cols());
  reducedVelocity.head<3>() << 0.9, -0.25, 0.7;
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    reducedVelocity(3 + static_cast<Eigen::Index>(i)) = 3.0 + 0.4 * static_cast<double>(i);
    reducedVelocity(7 + static_cast<Eigen::Index>(i)) = 0.5 - 0.3 * static_cast<double>(i);
  }
  const Eigen::VectorXd velocity = reduction.expand(reducedVelocity);
  setVelocity(robot, velocity);
  BOOST_REQUIRE_GT(std::abs(reducedVelocity(2)), 0.5);

  // (1) Pdot along the motion, by central differences. Zero to working
  //     precision for a planar motion; the same probe on a rolling chassis is
  //     shown to return a large value, so the zero is a measurement and not a
  //     blind spot.
  const auto liftAt = [&](const Eigen::VectorXd & motion, double time)
  {
    auto scratch = loadFourSteeringRobot();
    auto & moved = scratch->robot(0);
    setSteeringAngles(moved, uncoordinatedSteeringAngles());
    setVelocity(moved, motion);
    moved.eulerIntegration(time);
    moved.forwardKinematics();
    moved.forwardVelocity();
    return planarReduction(moved, wheels, Eigen::Vector3d::UnitZ()).lift;
  };
  for(const double step : {4e-3, 1e-3})
  {
    const Eigen::MatrixXd rate = (liftAt(velocity, step) - liftAt(velocity, -step)) / (2.0 * step);
    BOOST_TEST_MESSAGE("DYN-02 |Pdot| along the planar motion at h=" << step << ": "
                                                                     << rate.lpNorm<Eigen::Infinity>());
    BOOST_CHECK_SMALL(rate.lpNorm<Eigen::Infinity>(), 1e-9);
  }
  Eigen::VectorXd tilting = velocity;
  tilting(0) = 0.8; // a roll rate: the chassis leaves the plane and P must move
  const Eigen::MatrixXd tiltingRate = (liftAt(tilting, 1e-3) - liftAt(tilting, -1e-3)) / 2e-3;
  BOOST_TEST_MESSAGE("DYN-02 |Pdot| when the chassis leaves the plane: " << tiltingRate.lpNorm<Eigen::Infinity>());
  BOOST_CHECK_GT(tiltingRate.lpNorm<Eigen::Infinity>(), 0.1);

  // (2) h_red = P^T h, with h the full nonlinear effects. Cross-checked against
  //     an independent factorisation: rbd::Coriolis builds the Coriolis matrix
  //     by a different route than ForwardDynamics::computeC.
  rbd::ForwardDynamics forwardDynamics(robot.mb());
  forwardDynamics.computeC(robot.mb(), robot.mbc());
  const Eigen::VectorXd bias = forwardDynamics.C();
  const auto restore = robot.mbc().alpha;
  setVelocity(robot, Eigen::VectorXd::Zero(robot.mb().nrDof()));
  forwardDynamics.computeC(robot.mb(), robot.mbc());
  const Eigen::VectorXd gravityOnly = forwardDynamics.C();
  robot.mbc().alpha = restore;
  robot.forwardKinematics();
  robot.forwardVelocity();

  rbd::Coriolis coriolis(robot.mb());
  const Eigen::MatrixXd coriolisMatrix = coriolis.coriolis(robot.mb(), robot.mbc());
  const double scale = robot.mass() * mc_rtc::constants::GRAVITY;
  BOOST_CHECK_SMALL((coriolisMatrix * velocity + gravityOnly - bias).lpNorm<Eigen::Infinity>(), 1e-4 * scale);

  const Eigen::VectorXd reducedBias = reduction.lift.transpose() * bias;
  const Eigen::VectorXd reducedGravityOnly = reduction.lift.transpose() * gravityOnly;
  const double dropped = (reducedBias - reducedGravityOnly).lpNorm<Eigen::Infinity>();

  // The bias the shipped whole-body constraint actually enforces must be the
  // same h: Tasks' MotionConstr writes M alphaD - J^T f = -h on the six
  // unactuated base rows, so -LowerGenInEq().head(6) is its own copy of h. This
  // is where DYN-02 touches production code rather than the reduction alone.
  mc_solver::TasksQPSolver solver(robots, 0.005);
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addTask(&posture);
  solver.addConstraintSet(dynamics);
  setVelocity(solver.robot(0), velocity);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd shippedBias = -dynamics.motionConstr().LowerGenInEq().head(6);
  BOOST_TEST_MESSAGE("DYN-02 shipped base bias [" << shippedBias.transpose() << "], model [" << bias.head(6).transpose()
                                                  << "]");
  BOOST_CHECK_SMALL((shippedBias - bias.head(6)).lpNorm<Eigen::Infinity>(), 1e-4 * scale);
  // The same rows read through the reduction: the three planar entries of
  // h_red are exactly the ones the whole-body constraint carries, because P's
  // first three columns are supported on the six base rows alone.
  const Eigen::Vector3d shippedPlanarBias = reduction.lift.topRows(6).leftCols<3>().transpose() * shippedBias;
  BOOST_TEST_MESSAGE("DYN-02 planar h_red from the shipped constraint [" << shippedPlanarBias.transpose()
                                                                        << "], from the model ["
                                                                        << reducedBias.head<3>().transpose() << "]");
  BOOST_CHECK_SMALL((shippedPlanarBias - reducedBias.head<3>()).lpNorm<Eigen::Infinity>(), 1e-4 * scale);
  // Non-vacuity of that comparison: the gravity-only bias would have failed it.
  BOOST_CHECK_GT((reducedGravityOnly.head<3>() - reducedBias.head<3>()).lpNorm<Eigen::Infinity>(), 1e-4 * scale);
  solver.removeConstraintSet(dynamics);
  solver.removeTask(&posture);
  BOOST_TEST_MESSAGE("DYN-02 velocity-product part of h_red: " << dropped << " against 1e-4 m g = " << 1e-4 * scale);
  // (3) Dropping the velocity-product part moves h_red by far more than the
  //     tolerance the card states, so the test is discriminating.
  BOOST_CHECK_GT(dropped, 1e-4 * scale);
}

BOOST_AUTO_TEST_CASE(ContactForcesFollowNewtonsLawGeneralAndStaticDYN03)
{
  // DYN-03, the corrected form of claim (1). The whole-body QP does contain a
  // surface-normal force balance -- the planar reduction is what discards it --
  // so all three components are earned by the solver here and the assertion is
  // on the QP, not on a load model.
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.softLateralRows = true;
  options.lateralSlackWeight = 1e7;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  TargetAccelerationTask targetTask(robot.mb(), 0);
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);

  const Eigen::Vector3d gravity = -robot.mbc().gravity;
  BOOST_REQUIRE_LT(gravity.z(), -9.0);
  const double weight = robot.mass() * gravity.norm();
  rbd::CoMJacobian comJacobian(robot.mb());

  struct Branch
  {
    const char * name;
    double linear;
    double yaw;
  };
  double acceleratingMismatch = 0.0;
  for(const Branch & branch : {Branch{"static", 0.0, 0.0}, Branch{"general", 0.9, 0.4}})
  {
    const Eigen::VectorXd target = ackermannTarget(robot, branch.linear, branch.yaw);
    targetTask.target(target);
    rolling.update(solver);
    BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
    const Eigen::VectorXd lambda = solver.solver().lambdaVec();
    const Eigen::VectorXd acceleration = solver.solver().alphaDVec(0);

    Eigen::Vector3d contactForce = Eigen::Vector3d::Zero();
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      const auto forces = dynamics.endpointForces(wheels[i].name, lambda.segment<8>(static_cast<Eigen::Index>(8 * i)));
      contactForce += forces[0] + forces[1];
    }
    const Eigen::Vector3d comAcceleration =
        comJacobian.jacobian(robot.mb(), robot.mbc()) * acceleration + comJacobian.normalAcceleration(robot.mb(), robot.mbc());
    const Eigen::Vector3d newton = robot.mass() * (comAcceleration - gravity);
    BOOST_TEST_MESSAGE("DYN-03 " << branch.name << ": sum f = [" << contactForce.transpose() << "], m (comddot - g) = ["
                                 << newton.transpose() << "], m g = " << weight);
    BOOST_CHECK_SMALL((contactForce - newton).lpNorm<Eigen::Infinity>(), 1e-4 * weight);

    // The static specialisation, asserted only where it is legitimate.
    const double naiveMismatch = (contactForce + robot.mass() * gravity).lpNorm<Eigen::Infinity>();
    if(std::abs(branch.linear) + std::abs(branch.yaw) == 0.0)
    {
      BOOST_CHECK_SMALL(comAcceleration.lpNorm<Eigen::Infinity>(), 1e-6);
      BOOST_CHECK_SMALL(naiveMismatch, 1e-4 * weight);
    }
    else
    {
      acceleratingMismatch = naiveMismatch;
      BOOST_CHECK_GT(comAcceleration.lpNorm<Eigen::Infinity>(), 1e-2);
    }
  }
  // The correction itself: under acceleration the naive "contact forces equal
  // the weight" statement is wrong by m ||comddot||, and that error is far
  // above the tolerance the general law is asserted at. Encoding the naive
  // form would produce a false failure of exactly this size.
  BOOST_TEST_MESSAGE("DYN-03 naive static-law mismatch while accelerating: " << acceleratingMismatch);
  BOOST_CHECK_GT(acceleratingMismatch, 1e-2 * weight);

  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.removeTask(&targetTask);
}

BOOST_AUTO_TEST_CASE(YawMomentumEqualsTheContactMomentDYN04)
{
  // DYN-04, the corrected form of claim (3). The plane-normal component of the
  // centroidal angular-momentum rate is the net moment of the *contact* forces
  // about the CoM; gravity contributes none and the steering torques, being
  // internal, contribute none either. The steering torques are shown to be
  // large in this fixture, so "they contribute nothing" is a measurement.
  auto robots = loadFourSteeringRobot();
  mc_solver::TasksQPSolver solver(robots, 0.005);
  auto & robot = solver.robot(0);
  auto wheels = fourSteeringWheels();
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.softLateralRows = true;
  options.lateralSlackWeight = 1e7;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  TargetAccelerationTask targetTask(robot.mb(), 0);
  const Eigen::VectorXd target = ackermannTarget(robot, 0.7, 0.5);
  targetTask.target(target);
  solver.addTask(&targetTask);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);

  // A turning state so the yaw balance is not trivially 0 = 0.
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
  velocity(2) = 0.6;
  velocity(3) = 0.8;
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    velocity(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheels[i].steeringJoint)))) =
        0.4 - 0.2 * static_cast<double>(i);
  }
  setVelocity(robot, velocity);
  rolling.update(solver);
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd lambda = solver.solver().lambdaVec();
  const Eigen::VectorXd acceleration = solver.solver().alphaDVec(0);

  const Eigen::Vector3d com = rbd::computeCoM(robot.mb(), robot.mbc());
  const Eigen::Vector3d comVelocity = rbd::computeCoMVelocity(robot.mb(), robot.mbc());
  rbd::CentroidalMomentumMatrix centroidal(robot.mb());
  centroidal.computeMatrix(robot.mb(), robot.mbc(), com);
  const sva::ForceVecd normalDot = centroidal.normalMomentumDot(robot.mb(), robot.mbc(), com, comVelocity);
  const Eigen::VectorXd momentumRate = centroidal.matrix() * acceleration + normalDot.vector();
  const Eigen::Vector3d angularRate = momentumRate.head<3>();

  Eigen::Vector3d contactMoment = Eigen::Vector3d::Zero();
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const auto forces = dynamics.endpointForces(wheels[i].name, lambda.segment<8>(static_cast<Eigen::Index>(8 * i)));
    const auto & geometry = dynamics.geometryResult(wheels[i].name);
    contactMoment += (geometry.lineStart - com).cross(forces[0]) + (geometry.lineEnd - com).cross(forces[1]);
  }

  const Eigen::Vector3d normal = options.terrainNormal.normalized();
  const double largestOffset = 0.45 + 0.3;
  const double scale = robot.mass() * mc_rtc::constants::GRAVITY * largestOffset;
  const double residual = std::abs(normal.dot(angularRate) - normal.dot(contactMoment));
  BOOST_TEST_MESSAGE("DYN-04 n.Ldot = " << normal.dot(angularRate) << ", n.(contact moment) = "
                                        << normal.dot(contactMoment) << ", residual " << residual << " against "
                                        << 1e-4 * scale);
  BOOST_CHECK_SMALL(residual, 1e-4 * scale);
  BOOST_CHECK_GT(std::abs(normal.dot(contactMoment)), 1e-3 * scale);

  // "Actuator torques contribute nothing": the steering torques are far from
  // zero here, and the balance above closed without any of them appearing.
  dynamics.motionConstr().computeTorque(solver.solver().alphaDVec(), lambda);
  const Eigen::VectorXd torque = dynamics.motionConstr().torque();
  double steeringTorqueSum = 0.0;
  for(const auto & wheel : wheels)
  {
    steeringTorqueSum += torque(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheel.steeringJoint))));
  }
  BOOST_TEST_MESSAGE("DYN-04 sum of steering torques (which must not appear above): " << steeringTorqueSum
                                                                                      << ", n.Ldot "
                                                                                      << normal.dot(angularRate));
  // The steering torques are not negligible: they are orders of magnitude above
  // the residual the balance closed to, so "actuator torques contribute
  // nothing" is a measurement here rather than an absence of signal.
  BOOST_CHECK_GT(std::abs(steeringTorqueSum), 100.0 * std::max(residual, 1e-12));
  // And the naive claim the report calls false -- that the steering torques sum
  // to the yaw-momentum derivative -- is measurably wrong on the same solve.
  BOOST_CHECK_GT(std::abs(normal.dot(angularRate) - steeringTorqueSum), 1e-2 * std::abs(normal.dot(angularRate)));
  BOOST_CHECK_SMALL(torque.head<6>().lpNorm<Eigen::Infinity>(), 1e-8);

  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  solver.removeTask(&targetTask);
}

// ===========================================================================
// Layer G. Independent oracles.
// ===========================================================================

BOOST_AUTO_TEST_CASE(ReducedPlanarQPMatchesItsWholeBodyPreimageORC02)
{
  // ORC-02, the card the suite calls the single most valuable oracle. mc_rtc
  // ships no planar reduction, so the reduced problem is built here from
  // mc_rbdyn::steeringRollingMatrix() and the documented reduction
  // eq:reduced-velocity-coordinates, solved independently, and lifted back
  // through P. Nothing on the reduced side reads a row, a right-hand side or a
  // weight out of RollingContactConstraint: the two problems share only the
  // robot state and the numeric options.
  auto robots = loadFourSteeringRobot();
  constexpr double dt = 0.005;
  constexpr double gain = 20.0;
  constexpr double slackWeight = 1e7;
  mc_solver::TasksQPSolver solver(robots, dt);
  auto & robot = solver.robot(0);
  const auto wheels = fourSteeringWheels();

  // Deliberately *not* an ICR-coordinated state. Coordinated angles put every
  // carrier velocity along its own rolling direction, which makes the rolling
  // row's direction-derivative bias tdot . v_c identically zero and leaves that
  // term untested -- a mutation dropping it passed against a coordinated
  // fixture. The generic uncoordinated state is also the one the softened
  // lateral rows exist for, so the slacks are active here as well.
  constexpr double linear = 0.8;
  constexpr double lateral = 0.15;
  constexpr double yaw = 0.5;
  setSteeringAngles(robot, uncoordinatedSteeringAngles());
  const std::array<double, 4> steeringRates = {0.5, -0.3, 0.7, -0.9};
  const std::array<double, 4> driveRates = {3.1, 2.4, -1.2, 0.8};
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
  velocity(2) = yaw;
  velocity(3) = linear;
  velocity(4) = lateral;
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    velocity(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheels[i].driveJoint)))) = driveRates[i];
    velocity(robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName(wheels[i].steeringJoint)))) =
        steeringRates[i];
  }
  setVelocity(robot, velocity);

  const auto reduction = planarReduction(robot, wheels, Eigen::Vector3d::UnitZ());
  const auto reducedSize = reduction.lift.cols();
  BOOST_REQUIRE_EQUAL(reducedSize, 11);
  // The reduction really is the state's preimage: the measured motion is in range(P).
  BOOST_REQUIRE_SMALL((reduction.expand(reduction.reduce(velocity)) - velocity).lpNorm<Eigen::Infinity>(), 1e-12);
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    BOOST_CHECK_SMALL((reduction.offsets[i] - fourSteeringOffsets()[i]).lpNorm<Eigen::Infinity>(), 1e-9);
  }

  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = gain;
  options.softLateralRows = true;
  options.lateralSlackWeight = slackWeight;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  TargetAccelerationTask targetTask(robot.mb(), 0);
  Eigen::VectorXd reducedTarget(reducedSize);
  reducedTarget << 0.4, -0.2, 0.3, 1.5, -2.2, 0.9, 3.3, 0.6, -0.4, 0.2, -0.7;
  const Eigen::VectorXd target = reduction.expand(reducedTarget);
  targetTask.target(target);
  solver.addTask(&targetTask);
  solver.addConstraintSet(rolling);
  rolling.update(solver);

  // ---- The structural claims of the reduction, on the shipped rows ---------
  const Eigen::MatrixXd hard = rolling.hardMatrix();
  const Eigen::VectorXd hardRhs = rolling.hardRhs();
  const auto & hardLabels = rolling.hardRowLabels();
  const Eigen::MatrixXd normalRows = labelledRows(hard, hardLabels, "/normal");
  BOOST_REQUIRE_EQUAL(normalRows.rows(), 4);
  // eq:reduced-normal-annihilated, measured on mc_rtc's own whole-body rows:
  // every normal row is annihilated by the lift, so it reduces to 0 = 0 and
  // must not appear in the planar QP. It is *not* redundant in the whole-body
  // QP, where it pins the three out-of-plane base accelerations.
  const double annihilated = (normalRows * reduction.lift).lpNorm<Eigen::Infinity>();
  const Eigen::Index normalRank = rankAt(normalRows, 1e-8);
  BOOST_TEST_MESSAGE("ORC-02 |n^T J P| = " << annihilated << ", rank of the normal rows " << normalRank);
  BOOST_CHECK_SMALL(annihilated, 1e-12);
  BOOST_CHECK_EQUAL(normalRank, 3);

  // ---- The reduced problem, built independently ---------------------------
  const auto planar = measuredPlanarWheels(robot, wheels, reduction);
  const Eigen::Vector3d planarTwist = reduction.reduce(velocity).head<3>();
  const auto planarRows = mc_rbdyn::steeringRollingMatrix(planar, planarTwist);
  const Eigen::VectorXd measuredReduced = reduction.reduce(velocity);
  Eigen::MatrixXd reducedRolling = Eigen::MatrixXd::Zero(4, reducedSize);
  Eigen::MatrixXd reducedLateral = Eigen::MatrixXd::Zero(4, reducedSize);
  Eigen::VectorXd reducedRollingRhs(4);
  Eigen::VectorXd reducedLateralRhs(4);
  for(Eigen::Index i = 0; i < 4; ++i)
  {
    reducedRolling.row(i).head<7>() = planarRows.matrix.row(2 * i);
    reducedLateral.row(i).head<7>() = planarRows.matrix.row(2 * i + 1);
    // G alphaD = -Gdot alpha - Kp G alpha, exactly as the shipped assembler
    // writes it, but assembled from the planar rows rather than read from them.
    reducedRollingRhs(i) =
        -planarRows.accelerationBias(2 * i) - gain * planarRows.matrix.row(2 * i).dot(measuredReduced.head<7>());
    reducedLateralRhs(i) =
        -planarRows.accelerationBias(2 * i + 1) - gain * planarRows.matrix.row(2 * i + 1).dot(measuredReduced.head<7>());
  }

  // The rows the whole-body assembler prints, composed with P, must BE the
  // planar rows. This is the reduction itself, and it is where a mismatch
  // between the two forms would surface.
  const Eigen::MatrixXd wholeBodyRolling = labelledRows(hard, hardLabels, "/longitudinal") * reduction.lift;
  BOOST_TEST_MESSAGE("ORC-02 rolling row gap " << (wholeBodyRolling - reducedRolling).lpNorm<Eigen::Infinity>());
  BOOST_CHECK_SMALL((wholeBodyRolling - reducedRolling).lpNorm<Eigen::Infinity>(), 1e-9);
  const auto unscaled = unscaledLateralRows(rolling, robot.mb().nrDof());
  BOOST_CHECK_SMALL((unscaled.first * reduction.lift - reducedLateral).lpNorm<Eigen::Infinity>(), 1e-9);

  Eigen::VectorXd wholeBodyRollingRhs(4);
  Eigen::Index found = 0;
  for(size_t i = 0; i < hardLabels.size(); ++i)
  {
    if(hardLabels[i].find("/longitudinal") != std::string::npos) { wholeBodyRollingRhs(found++) = hardRhs(static_cast<Eigen::Index>(i)); }
  }
  BOOST_REQUIRE_EQUAL(found, 4);
  BOOST_TEST_MESSAGE("ORC-02 rolling rhs gap " << (wholeBodyRollingRhs - reducedRollingRhs).lpNorm<Eigen::Infinity>());
  BOOST_CHECK_SMALL((wholeBodyRollingRhs - reducedRollingRhs).lpNorm<Eigen::Infinity>(), 1e-9);
  BOOST_CHECK_SMALL((unscaled.second - reducedLateralRhs).lpNorm<Eigen::Infinity>(), 1e-9);
  // The slacks are genuinely active, so the softened block is exercised.
  BOOST_CHECK_GT((reducedLateral * reducedTarget - reducedLateralRhs).lpNorm<Eigen::Infinity>(), 1e-3);
  // Both halves of each row's bias are non-negligible in this fixture, so
  // neither can be dropped without the right-hand-side comparison noticing.
  BOOST_TEST_MESSAGE("ORC-02 planar bias [" << planarRows.accelerationBias.transpose() << "]");
  for(Eigen::Index i = 0; i < 4; ++i)
  {
    BOOST_CHECK_GT(std::abs(planarRows.accelerationBias(2 * i)), 1e-2);
    BOOST_CHECK_GT(std::abs(planarRows.accelerationBias(2 * i + 1)), 1e-2);
  }
  // The degeneracy this fixture exists to avoid, recorded so it is not
  // reintroduced. On an ICR-coordinated state every carrier velocity is along
  // its own rolling direction, so tdot . v_c -- the entire rolling-row bias --
  // is identically zero and a mutation deleting that term goes unnoticed. Same
  // steering rates, same speeds; only the angles are coordinated.
  {
    auto coordinated = planar;
    const auto icr = icrSteeringAngles(linear, lateral, yaw);
    for(size_t i = 0; i < coordinated.size(); ++i)
    {
      coordinated[i].steeringAngle = icr[i];
      coordinated[i].steeringRate = steeringRates[i];
    }
    const auto coordinatedRows = mc_rbdyn::steeringRollingMatrix(coordinated, planarTwist);
    double coordinatedRolling = 0.0;
    double uncoordinatedRolling = 0.0;
    for(Eigen::Index i = 0; i < 4; ++i)
    {
      coordinatedRolling = std::max(coordinatedRolling, std::abs(coordinatedRows.accelerationBias(2 * i)));
      uncoordinatedRolling = std::max(uncoordinatedRolling, std::abs(planarRows.accelerationBias(2 * i)));
    }
    BOOST_TEST_MESSAGE("ORC-02 largest rolling-row bias: this fixture " << uncoordinatedRolling
                                                                       << ", ICR-coordinated " << coordinatedRolling);
    BOOST_CHECK_SMALL(coordinatedRolling, 1e-12);
    BOOST_CHECK_GT(uncoordinatedRolling, 1e-1);
  }

  // ---- Solve both and compare ---------------------------------------------
  BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
  const Eigen::VectorXd wholeBody = solver.solver().alphaDVec(0);
  const Eigen::VectorXd reducedSolution =
      quadraticOracle(reducedTarget + slackWeight * reducedLateral.transpose() * reducedLateralRhs, reducedLateral,
                      slackWeight, reducedRolling, reducedRollingRhs);

  // The whole-body optimum lies in range(P): the four normal rows pin the three
  // out-of-plane base accelerations, so nothing is lost by projecting.
  const Eigen::VectorXd projected = reduction.expand(reduction.reduce(wholeBody));
  BOOST_TEST_MESSAGE("ORC-02 out-of-plane part of the whole-body optimum "
                     << (wholeBody - projected).lpNorm<Eigen::Infinity>());
  BOOST_CHECK_SMALL((wholeBody - projected).lpNorm<Eigen::Infinity>(), 1e-8);

  const Eigen::VectorXd shared = reduction.reduce(wholeBody);
  const double relative =
      (reducedSolution - shared).lpNorm<Eigen::Infinity>() / (1.0 + shared.lpNorm<Eigen::Infinity>());
  BOOST_TEST_MESSAGE("ORC-02 shared variables: reduced [" << reducedSolution.transpose() << "], whole-body ["
                                                          << shared.transpose() << "], relative gap " << relative);
  BOOST_CHECK_LT(relative, 1e-6);

  // The lifted reduced solution is feasible for the full problem.
  const Eigen::VectorXd lifted = reduction.expand(reducedSolution);
  BOOST_TEST_MESSAGE("ORC-02 feasibility residual of the lifted point "
                     << (hard * lifted - hardRhs).lpNorm<Eigen::Infinity>());
  BOOST_CHECK_SMALL((hard * lifted - hardRhs).lpNorm<Eigen::Infinity>(), 1e-8);

  // ... and its objective value matches the full optimum.
  const auto objective = [&](const Eigen::VectorXd & x)
  {
    return 0.5 * (x - target).squaredNorm() + 0.5 * slackWeight * (unscaled.first * x - unscaled.second).squaredNorm();
  };
  BOOST_TEST_MESSAGE("ORC-02 objective: whole-body " << objective(wholeBody) << ", lifted reduced "
                                                     << objective(lifted));
  BOOST_CHECK_LT(std::abs(objective(lifted) - objective(wholeBody)), 1e-6 * (1.0 + std::abs(objective(wholeBody))));

  solver.removeConstraintSet(rolling);
  solver.removeTask(&targetTask);
}

// ===========================================================================
// The proportional-stabilization recursion: ROW-03 and QP-08.
// ===========================================================================

/** Measured lateral-residual series of a closed differential-drive loop.
 *
 * Runs the shipped constraint through TasksQPSolver::run(), which rebuilds the
 * rows from the current state, solves and Euler integrates. Entry k is the
 * residual the constraint itself measured at the start of cycle k, so the ratio
 * of consecutive entries is the realised discrete decay. Nothing here restates
 * the recursion; it is read off the closed loop.
 */
std::vector<double> lateralResidualSeries(double gain, double dt, size_t cycles)
{
  auto robots = loadDifferentialRobot();
  mc_solver::TasksQPSolver solver(robots, dt);
  const auto wheels = differentialWheels();
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = gain;
  options.differentialPlanar = true;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  solver.addTask(&posture);
  solver.addConstraintSet(rolling);

  auto & robot = solver.robot(0);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(robot.mb().nrDof());
  velocity(4) = 0.2; // the measured lateral velocity the gain has to remove
  rbd::vectorToParam(velocity, robot.mbc().alpha);
  robot.forwardKinematics();
  robot.forwardVelocity();

  std::vector<double> series;
  series.reserve(cycles);
  for(size_t cycle = 0; cycle < cycles; ++cycle)
  {
    if(!solver.run()) { throw std::runtime_error("Rolling-contact stabilization loop failed to solve"); }
    series.push_back(rolling.geometryResults()[0].velocityResidual.y());
  }
  solver.removeConstraintSet(rolling);
  solver.removeTask(&posture);
  return series;
}

BOOST_AUTO_TEST_CASE(LateralResidualDecaysAtTheDiscreteRateROW03)
{
  // ROW-03. One cycle multiplies the residual by 1 - Kp dt, not by exp(-Kp dt).
  // Both models are evaluated against the same measured series so the
  // false-failure trap the card names is asserted away rather than avoided.
  constexpr double dt = 0.005;
  for(const double gain : {20.0, 50.0, 120.0})
  {
    const auto series = lateralResidualSeries(gain, dt, 12);
    BOOST_REQUIRE_GT(std::abs(series.front()), 1e-3);
    const double discrete = 1.0 - gain * dt;
    const double continuous = std::exp(-gain * dt);
    double worstDiscrete = 0.0;
    double worstContinuous = 0.0;
    size_t compared = 0;
    for(size_t k = 0; k + 1 < series.size(); ++k)
    {
      // Once the residual has decayed into round-off the ratio is noise, not a
      // measurement of the recursion.
      if(std::abs(series[k]) < 1e-12) { continue; }
      ++compared;
      const double ratio = series[k + 1] / series[k];
      worstDiscrete = std::max(worstDiscrete, std::abs(ratio - discrete));
      worstContinuous = std::max(worstContinuous, std::abs(ratio - continuous));
    }
    BOOST_REQUIRE_GT(compared, 4u);
    BOOST_TEST_MESSAGE("ROW-03 Kp=" << gain << " Kp dt=" << gain * dt << ": worst |ratio - (1 - Kp dt)| "
                                    << worstDiscrete << ", worst |ratio - exp(-Kp dt)| " << worstContinuous
                                    << " over " << compared << " cycles");
    BOOST_CHECK_LT(worstDiscrete, 1e-9);
    // The false-failure trap the card names: the continuous-time model
    // exp(-Kp dt) is measurably wrong on the same series.
    BOOST_CHECK_GT(worstContinuous, 1e-3);
  }

  // At Kp = 0 the residual is frozen, not removed. This is the clause that a
  // "stabilization is always on" implementation fails.
  const auto frozen = lateralResidualSeries(0.0, dt, 40);
  const double drift = std::abs(frozen.back() - frozen.front());
  BOOST_TEST_MESSAGE("ROW-03 Kp=0: residual " << frozen.front() << " -> " << frozen.back() << ", drift " << drift);
  BOOST_CHECK_GT(std::abs(frozen.front()), 1e-3);
  BOOST_CHECK_SMALL(drift, 1e-9);
}

BOOST_AUTO_TEST_CASE(StabilityWindowOfTheProportionalStabilizationQP08)
{
  // QP-08. Sweep Kp dt over [0, 2.5] and require the observed regime to match
  // the prediction: monotone decay for Kp dt <= 1, oscillatory decay up to 2,
  // divergence above. This is what fixes an upper limit on velocityGain.
  constexpr double dt = 0.005;
  const std::array<double, 8> products = {0.0, 0.25, 0.5, 1.0, 1.4, 1.9, 2.1, 2.5};
  for(const double product : products)
  {
    const auto series = lateralResidualSeries(product / dt, dt, 24);
    const double first = series.front();
    const double last = series.back();
    const double growth = std::abs(last) / std::abs(first);
    bool alternates = false;
    for(size_t k = 0; k + 1 < series.size(); ++k)
    {
      if(series[k] * series[k + 1] < 0.0) { alternates = true; }
    }
    BOOST_TEST_MESSAGE("QP-08 Kp dt=" << product << ": |e_N| / |e_0| = " << growth
                                      << ", sign alternation " << alternates);
    if(product == 0.0)
    {
      BOOST_CHECK_CLOSE(growth, 1.0, 1e-6);
      BOOST_CHECK(!alternates);
    }
    else if(product <= 1.0)
    {
      BOOST_CHECK_LT(growth, 1.0);
      BOOST_CHECK(!alternates); // monotone: no overshoot through zero
    }
    else if(product < 2.0)
    {
      BOOST_CHECK_LT(growth, 1.0);
      BOOST_CHECK(alternates); // oscillatory decay
    }
    else
    {
      BOOST_CHECK_GT(growth, 1.0); // divergence
      BOOST_CHECK(alternates);
    }
  }
}

// ===========================================================================
// QP-03. Primal uniqueness across warm starts and solvers.
// ===========================================================================

namespace
{

/** One four-steering solve, reduced to the quantities both backends express. */
struct SteeringPrimal
{
  Eigen::VectorXd acceleration;
  std::vector<Eigen::Vector3d> forces;
};

/** Solve one four-steering cycle with dynamics at @p regularization on @p SolverT.
 *
 * The objective is a PostureTask plus a chassis PositionTask/OrientationTask,
 * all of which exist on both backends, so the same physical problem is posed to
 * the Tasks QLD solver and to TVM's.
 */
template<typename SolverT>
SteeringPrimal fourSteeringPrimal(double regularization)
{
  auto robots = loadFourSteeringRobot();
  SolverT solver(robots, 0.005);
  const auto wheels = fourSteeringWheels();
  // An uncoordinated, moving state: on an ICR-coordinated state at rest every
  // row bias is identically zero, which would leave most of the assembly out of
  // the comparison (see the degeneracy recorded in ORC-02).
  setSteeringAngles(solver.robot(0), uncoordinatedSteeringAngles());
  setIncompatibleFourWheelState(solver.robot(0));
  mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels, false,
                                                       regularization);
  mc_solver::RollingContactConstraintOptions options;
  options.velocityGain = 0.0;
  options.softLateralRows = true;
  options.lateralSlackWeight = 1e7;
  mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
  mc_tasks::PostureTask posture(solver, 0, 5.0, 100.0);
  mc_tasks::PositionTask position("chassis", solver.robots(), 0, 20.0, 2000.0);
  position.position(Eigen::Vector3d{0.05, 0.02, 0.2});
  mc_tasks::OrientationTask orientation("chassis", solver.robots(), 0, 20.0, 500.0);
  solver.addConstraintSet(dynamics);
  solver.addConstraintSet(rolling);
  solver.addTask(&posture);
  solver.addTask(&position);
  solver.addTask(&orientation);
  if(!solver.run()) { throw std::runtime_error("QP-03 four-steering solve failed"); }

  SteeringPrimal primal;
  if constexpr(std::is_same_v<SolverT, mc_solver::TasksQPSolver>)
  {
    primal.acceleration = solver.solver().alphaDVec(0);
    const Eigen::VectorXd lambda = solver.solver().lambdaVec();
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      const auto forces = dynamics.endpointForces(wheels[i].name, lambda.segment<8>(static_cast<Eigen::Index>(8 * i)));
      primal.forces.push_back(forces[0]);
      primal.forces.push_back(forces[1]);
    }
  }
  else
  {
    primal.acceleration = solver.robot(0).tvmRobot().alphaD()->value();
    for(const auto & wheel : wheels)
    {
      const auto forces = dynamics.endpointForces(wheel.name);
      primal.forces.push_back(forces[0]);
      primal.forces.push_back(forces[1]);
    }
  }
  solver.removeTask(&orientation);
  solver.removeTask(&position);
  solver.removeTask(&posture);
  solver.removeConstraintSet(rolling);
  solver.removeConstraintSet(dynamics);
  return primal;
}

} // namespace

BOOST_AUTO_TEST_CASE(PrimalSolutionIsUniqueAcrossWarmStartsAndSolversQP03)
{
  // QP-03, in the regime the card names: epsilon_lambda > 0. The generator
  // block is only epsilon-strongly convex, so lambda is compared at the
  // epsilon-aware tolerance sqrt(eps_alg / epsilon_lambda) and alphaD at the
  // tight one. A uniform tight tolerance on lambda is the false-failure trap.
  constexpr double regularization = 2e-4;
  const double lambdaTolerance = std::sqrt(1e-9 / regularization);

  // --- Warm starts ---------------------------------------------------------
  // The same problem is posed to the same solver after three different
  // histories: cold, after a different target, and after two. If the primal
  // depended on the solver's internal state these would not coincide.
  const Eigen::VectorXd decoyA = Eigen::VectorXd::Constant(14, 0.4);
  const Eigen::VectorXd decoyB = Eigen::VectorXd::Constant(14, -1.7);
  std::vector<Eigen::VectorXd> accelerations;
  std::vector<Eigen::VectorXd> multipliers;
  for(int history = 0; history < 3; ++history)
  {
    auto robots = loadFourSteeringRobot();
    mc_solver::TasksQPSolver solver(robots, 0.005);
    auto & robot = solver.robot(0);
    const auto wheels = fourSteeringWheels();
    mc_solver::RollingContactDynamicsConstraint dynamics(solver.robots(), 0, solver.dt(), wheels, false,
                                                         regularization);
    mc_solver::RollingContactConstraintOptions options;
    options.velocityGain = 0.0;
    options.softLateralRows = true;
    options.lateralSlackWeight = 1e7;
    mc_solver::RollingContactConstraint rolling(solver.robots(), 0, wheels, options);
    TargetAccelerationTask targetTask(robot.mb(), 0);
    solver.addTask(&targetTask);
    solver.addConstraintSet(dynamics);
    solver.addConstraintSet(rolling);
    const Eigen::VectorXd target = ackermannTarget(robot, 0.35, 0.12);
    const auto restoreQ = robot.mbc().q;
    const auto restoreAlpha = robot.mbc().alpha;
    // The decoy cycles integrate, so the solver reaches the final problem from a
    // different internal state *and* from a different robot state. Restoring the
    // state before the final solve is what makes this a warm start rather than a
    // different problem, and it is what a stale row cache would fail.
    for(int step = 0; step < history; ++step)
    {
      targetTask.target(step == 0 ? decoyA : decoyB);
      BOOST_REQUIRE(solver.run());
    }
    robot.mbc().q = restoreQ;
    robot.mbc().alpha = restoreAlpha;
    robot.forwardKinematics();
    robot.forwardVelocity();
    targetTask.target(target);
    rolling.update(solver);
    BOOST_REQUIRE(solver.solver().solveNoMbcUpdate(solver.robots().mbs(), solver.robots().mbcs()));
    accelerations.push_back(solver.solver().alphaDVec(0));
    multipliers.push_back(solver.solver().lambdaVec());
    solver.removeConstraintSet(rolling);
    solver.removeConstraintSet(dynamics);
    solver.removeTask(&targetTask);
  }
  for(size_t i = 1; i < accelerations.size(); ++i)
  {
    const double motion = (accelerations[i] - accelerations[0]).lpNorm<Eigen::Infinity>()
                          / (1.0 + accelerations[0].lpNorm<Eigen::Infinity>());
    const double generators = (multipliers[i] - multipliers[0]).lpNorm<Eigen::Infinity>();
    BOOST_TEST_MESSAGE("QP-03 warm start " << i << ": relative alphaD gap " << motion << ", lambda gap " << generators
                                           << " against " << lambdaTolerance);
    BOOST_CHECK_LT(motion, 1e-6);
    BOOST_CHECK_LT(generators, lambdaTolerance);
  }
  // Non-vacuity: the decoy problems really do move the solution, so the three
  // histories are distinct states of the solver.
  BOOST_CHECK_GT((decoyA - accelerations[0]).lpNorm<Eigen::Infinity>(), 0.1);

  // --- Two independent solvers ---------------------------------------------
  const auto tasksPrimal = fourSteeringPrimal<mc_solver::TasksQPSolver>(regularization);
  const auto tvmPrimal = fourSteeringPrimal<mc_solver::TVMQPSolver>(regularization);
  BOOST_REQUIRE_EQUAL(tasksPrimal.forces.size(), tvmPrimal.forces.size());
  const Eigen::VectorXd backendGap = tasksPrimal.acceleration - tvmPrimal.acceleration;
  const double scale = 1.0 + tasksPrimal.acceleration.lpNorm<Eigen::Infinity>();
  // The card states 1e-6 relative on the planar twist and the epsilon-aware
  // tolerance on the generator-driven quantities. The wheel accelerations are
  // in the second class: they are fixed by the force block, which is only
  // epsilon_lambda-strongly convex, so two independent solvers legitimately
  // land a few 1e-6 apart there.
  const double twistGap =
      std::max(std::abs(backendGap(2)), std::max(std::abs(backendGap(3)), std::abs(backendGap(4)))) / scale;
  const double wholeGap = backendGap.lpNorm<Eigen::Infinity>() / scale;
  BOOST_TEST_MESSAGE("QP-03 Tasks/TVM relative gap: planar twist " << twistGap << ", whole alphaD " << wholeGap);
  BOOST_CHECK_LT(twistGap, 1e-6);
  BOOST_CHECK_LT(wholeGap, lambdaTolerance);
  double forceGap = 0.0;
  double forceScale = 0.0;
  for(size_t i = 0; i < tasksPrimal.forces.size(); ++i)
  {
    forceGap = std::max(forceGap, (tasksPrimal.forces[i] - tvmPrimal.forces[i]).lpNorm<Eigen::Infinity>());
    forceScale = std::max(forceScale, tasksPrimal.forces[i].lpNorm<Eigen::Infinity>());
  }
  // The two backends do not share a force parameterisation -- Tasks uses the
  // polyhedral generators, TVM a 3D vector -- so only the physical force is
  // comparable, and only at the epsilon-aware tolerance.
  BOOST_TEST_MESSAGE("QP-03 Tasks/TVM endpoint-force gap " << forceGap << " on a scale of " << forceScale);
  BOOST_CHECK_GT(forceScale, 1.0);
  BOOST_CHECK_LT(forceGap, lambdaTolerance * forceScale);
}
