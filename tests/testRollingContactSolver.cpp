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

#include <Eigen/QR>

#include <RBDyn/MultiBodyConfig.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

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
  options.steeringPlanar = true;
  options.steeringPlanarWheels = {"front_left", "rear_left"};
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
  options.steeringPlanar = true;
  options.steeringPlanarWheels = {"front_left", "rear_left"};
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
  options.steeringPlanar = true;
  options.steeringPlanarWheels = {"front_left", "rear_left"};
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
  options.steeringPlanar = true;
  options.steeringPlanarWheels = {"front_left", "rear_left"};
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
    options.steeringPlanar = true;
    options.steeringPlanarWheels = {"front_left", "rear_left"};
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
