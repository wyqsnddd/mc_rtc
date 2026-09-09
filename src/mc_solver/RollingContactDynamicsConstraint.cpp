/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_solver/RollingContactDynamicsConstraint.h>

#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_solver/TVMQPSolver.h>

#include <mc_rbdyn/Robots.h>

#include <mc_tvm/DynamicFunction.h>

#include <mc_rtc/void_ptr.h>

#include <Tasks/Bounds.h>
#include <Tasks/QPContacts.h>
#include <Tasks/QPMotionConstr.h>

#include <RBDyn/Jacobian.h>

#include <tvm/function/abstract/LinearFunction.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "RollingContactConfig.h"

namespace mc_solver
{

namespace
{

constexpr int generatorsPerPoint = 4;
constexpr int pointsPerWheel = 2;
constexpr int lambdasPerWheel = generatorsPerPoint * pointsPerWheel;
constexpr double pi = 3.14159265358979323846;
constexpr const char * terrainBody = "__rolling_terrain__";

/** Linearized Coulomb cone for one contact point, TVM side.
 *
 * The four rows are mu * (n.f) - s_r * (t.f) - s_l * (l.f) >= 0 over both sign
 * combinations, which is exactly |t.f| + |l.f| <= mu * (n.f). That is the L1
 * ball: an *inscribed* approximation of the exact circular cone, conservative
 * everywhere and tight only on the rolling/lateral axes. The Tasks side uses
 * the equivalent inscribed four-generator pyramid (see the generator loop in
 * update()), so neither backend can plan a force outside the true cone.
 *
 * frictionMargin() is nevertheless a *diagnostic*, not the constrained
 * quantity: it grades the resultant of the wheel's two endpoint forces against
 * the exact circle, and it is read after the solve, once the controller has
 * already integrated the robot state that defined these frames. When the
 * tangential force is large the small frame inconsistency that introduces is
 * visible in the resultant. Measured on the crab profile: the margin is
 * -1.04 N on a 180 N normal force (0.6%) during the four friction-saturated
 * start-up ticks, where the tangential force is ~200 N, and +146 N for the
 * remaining 996 cycles, where it is ~0 N. This is why
 * rolling-contact-report/scripts/check-controller-log.py deliberately asserts
 * on the normal force and the drive-torque margin but not on this one.
 */
class TVMRollingForceCone final : public tvm::function::abstract::LinearFunction
{
public:
  SET_UPDATES(TVMRollingForceCone, Jacobian, B)

  explicit TVMRollingForceCone(tvm::VariablePtr force)
  : tvm::function::abstract::LinearFunction(4), force_(std::move(force)), matrix_(Eigen::MatrixXd::Zero(4, 3))
  {
    registerUpdates(Update::Jacobian, &TVMRollingForceCone::updateJacobian, Update::B,
                    &TVMRollingForceCone::updateB);
    addOutputDependency<TVMRollingForceCone>(Output::Jacobian, Update::Jacobian);
    addOutputDependency<TVMRollingForceCone>(Output::B, Update::B);
    addVariable(force_, true);
    velocity_.setZero();
  }

  void frame(const mc_rbdyn::RollingContactGeometryResult & result,
             double friction,
             const Eigen::Matrix3d & worldToBody)
  {
    const Eigen::Vector3d normal = worldToBody * result.normalDirection;
    const Eigen::Vector3d rolling = worldToBody * result.rollingDirection;
    const Eigen::Vector3d lateral = worldToBody * result.lateralDirection;
    int row = 0;
    for(const double rollingSign : {-1.0, 1.0})
    {
      for(const double lateralSign : {-1.0, 1.0})
      {
        matrix_.row(row++) = friction * normal.transpose() - rollingSign * rolling.transpose()
                             - lateralSign * lateral.transpose();
      }
    }
    jacobian_[force_.get()] = matrix_;
  }

private:
  void updateJacobian() { jacobian_[force_.get()] = matrix_; }
  void updateB() { b_.setZero(); }

  tvm::VariablePtr force_;
  Eigen::MatrixXd matrix_;
};

class TVMRollingForceMode final : public tvm::function::abstract::LinearFunction
{
public:
  SET_UPDATES(TVMRollingForceMode, Jacobian, B)

  explicit TVMRollingForceMode(tvm::VariablePtr force)
  : tvm::function::abstract::LinearFunction(3), force_(std::move(force)), matrix_(Eigen::Matrix3d::Zero())
  {
    registerUpdates(Update::Jacobian, &TVMRollingForceMode::updateJacobian, Update::B,
                    &TVMRollingForceMode::updateB);
    addOutputDependency<TVMRollingForceMode>(Output::Jacobian, Update::Jacobian);
    addOutputDependency<TVMRollingForceMode>(Output::B, Update::B);
    addVariable(force_, true);
    velocity_.setZero();
  }

  void unrestricted()
  {
    matrix_.setZero();
    jacobian_[force_.get()] = matrix_;
  }

  void detached()
  {
    matrix_.setIdentity();
    jacobian_[force_.get()] = matrix_;
  }

  void sliding(const Eigen::Vector3d & generatorInBody)
  {
    const Eigen::Vector3d direction = generatorInBody.normalized();
    Eigen::Vector3d seed = Eigen::Vector3d::UnitX();
    if(std::abs(direction.y()) <= std::abs(direction.x()) && std::abs(direction.y()) <= std::abs(direction.z()))
    {
      seed = Eigen::Vector3d::UnitY();
    }
    else if(std::abs(direction.z()) <= std::abs(direction.x())
            && std::abs(direction.z()) <= std::abs(direction.y()))
    {
      seed = Eigen::Vector3d::UnitZ();
    }
    const Eigen::Vector3d first = direction.cross(seed).normalized();
    const Eigen::Vector3d second = direction.cross(first);
    matrix_.row(0) = first.transpose();
    matrix_.row(1) = second.transpose();
    matrix_.row(2).setZero();
    jacobian_[force_.get()] = matrix_;
  }

private:
  void updateJacobian() { jacobian_[force_.get()] = matrix_; }
  void updateB() { b_.setZero(); }

  tvm::VariablePtr force_;
  Eigen::Matrix3d matrix_;
};

std::vector<std::vector<double>> torqueLower(const mc_rbdyn::Robot & robot, bool infinite)
{
  auto out = robot.tl();
  if(infinite)
  {
    for(auto & joint : out)
    {
      for(auto & value : joint) { value = -std::numeric_limits<double>::infinity(); }
    }
  }
  return out;
}

std::vector<std::vector<double>> torqueUpper(const mc_rbdyn::Robot & robot, bool infinite)
{
  auto out = robot.tu();
  if(infinite)
  {
    for(auto & joint : out)
    {
      for(auto & value : joint) { value = std::numeric_limits<double>::infinity(); }
    }
  }
  return out;
}

std::vector<std::vector<double>> torqueRateLower(const mc_rbdyn::Robot & robot, bool infinite)
{
  auto out = robot.tdl();
  if(infinite)
  {
    for(auto & joint : out)
    {
      for(auto & value : joint) { value = -std::numeric_limits<double>::infinity(); }
    }
  }
  return out;
}

std::vector<std::vector<double>> torqueRateUpper(const mc_rbdyn::Robot & robot, bool infinite)
{
  auto out = robot.tdu();
  if(infinite)
  {
    for(auto & joint : out)
    {
      for(auto & value : joint) { value = std::numeric_limits<double>::infinity(); }
    }
  }
  return out;
}

} // namespace

struct RollingContactDynamicsConstraint::Impl
{
  struct ModeBound final : public tasks::qp::Bound
  {
    ModeBound(int & begin, const Eigen::VectorXd & lower, const Eigen::VectorXd & upper, std::string name)
    : begin_(begin), lower_(lower), upper_(upper), name_(std::move(name))
    {
    }

    int beginVar() const override { return begin_; }
    const Eigen::VectorXd & Lower() const override { return lower_; }
    const Eigen::VectorXd & Upper() const override { return upper_; }
    std::string nameBound() const override { return "RollingContactForceMode/" + name_; }
    std::string descBound(const std::vector<rbd::MultiBody> &, int i) override
    {
      return nameBound() + "/lambda" + std::to_string(i);
    }

    int & begin_;
    const Eigen::VectorXd & lower_;
    const Eigen::VectorXd & upper_;
    std::string name_;
  };

  // Defined below, after WheelData: it needs a complete WheelData& to look up
  // its wheel's contact id. WheelData only needs a non-owning pointer to it
  // (ownership lives in Impl::regularizationTasks), so the forward
  // declaration here is enough.
  struct GeneratorRegularizationTask;

  struct WheelData
  {
    WheelData(const mc_rbdyn::Robot & robot, const mc_rbdyn::RollingContactDescription & descriptionIn, int ambiguityIn)
    : description(descriptionIn), geometry(robot, description), wheelJacobian(robot.mb(), description.wheelBody),
      ambiguity(ambiguityIn), translatedJacobian(6, wheelJacobian.dof()),
      generatorJacobian(generatorsPerPoint, wheelJacobian.dof()),
      fullGeneratorJacobian(generatorsPerPoint, robot.mb().nrDof()),
      generalizedForce(Eigen::MatrixXd::Zero(robot.mb().nrDof(), lambdasPerWheel)),
      lambdaLower(Eigen::VectorXd::Zero(lambdasPerWheel)),
      lambdaUpper(Eigen::VectorXd::Constant(lambdasPerWheel, std::numeric_limits<double>::infinity())),
      tvmPoints(pointsPerWheel)
    {
      generators[0].setZero(3, generatorsPerPoint);
      generators[1].setZero(3, generatorsPerPoint);
      if(description.width <= 0.0)
      {
        throw std::invalid_argument("Rolling contact line width must be positive for wheel: " + description.name);
      }
    }

    mc_rbdyn::RollingContactDescription description;
    mc_rbdyn::RollingContactRobotGeometry geometry;
    mc_rbdyn::RollingContactGeometryResult result;
    rbd::Jacobian wheelJacobian;
    int ambiguity;
    int lambdaBegin = -1;
    Eigen::MatrixXd translatedJacobian;
    Eigen::MatrixXd generatorJacobian;
    Eigen::MatrixXd fullGeneratorJacobian;
    Eigen::MatrixXd generalizedForce;
    Eigen::VectorXd lambdaLower;
    Eigen::VectorXd lambdaUpper;
    std::unique_ptr<ModeBound> modeBound;
    // Non-owning: Impl::regularizationTasks owns it (see GeneratorRegularizationTask).
    GeneratorRegularizationTask * regularizationTask = nullptr;
    int slidingGenerator = 2;
    bool slidingDirectionEstablished = false;
    std::array<Eigen::Matrix<double, 3, Eigen::Dynamic>, pointsPerWheel> generators;
    tvm::VariableVector tvmForces;
    std::array<std::shared_ptr<TVMRollingForceCone>, pointsPerWheel> tvmCones;
    std::array<tvm::TaskWithRequirementsPtr, pointsPerWheel> tvmConeTasks;
    std::array<std::shared_ptr<TVMRollingForceMode>, pointsPerWheel> tvmModes;
    std::array<tvm::TaskWithRequirementsPtr, pointsPerWheel> tvmModeTasks;
    std::vector<sva::PTransformd> tvmPoints;
  };

  /** Tikhonov term eps * ||lambda_i||^2 on one wheel's eight generator
   * multipliers, see the RollingContactDynamicsConstraint class
   * documentation. Q_/C_ never change after construction: unlike
   * RollingContactConstraint.cpp's Impl::SoftTask, this term carries no
   * state dependent on robot geometry, so update() is a no-op.
   *
   * updateNrVars() duplicates RollingMotionConstr::updateNrVars()'s contact
   * lookup rather than reading wheel.lambdaBegin once that has run, because
   * Tasks::QPSolver::updateNrVars() calls every Task's updateNrVars() before
   * any Constraint's (Tasks/src/QPSolver.cpp): this Task cannot assume
   * RollingMotionConstr, a Constraint, has already resolved it this cycle.
   */
  struct GeneratorRegularizationTask final : public tasks::qp::Task
  {
    GeneratorRegularizationTask(Impl & owner, WheelData & wheel, double epsilon)
    : tasks::qp::Task(1.0), owner_(owner), wheel_(wheel)
    {
      const auto qc = RollingContactDynamicsConstraint::generatorRegularizationQC(lambdasPerWheel, epsilon);
      Q_ = qc.first;
      C_ = qc.second;
    }

    std::pair<int, int> begin() const override { return {lambdaBegin_, lambdaBegin_}; }

    void updateNrVars(const std::vector<rbd::MultiBody> &, const tasks::qp::SolverData & data) override
    {
      lambdaBegin_ = -1;
      const auto & contacts = data.allContacts();
      for(size_t contactIndex = 0; contactIndex < contacts.size(); ++contactIndex)
      {
        const auto & id = contacts[contactIndex].contactId;
        if(id.r1Index == static_cast<int>(owner_.robotIndex) && id.r2Index == -1
           && id.r1BodyName == wheel_.description.wheelBody && id.r2BodyName == terrainBody
           && id.ambiguityId == wheel_.ambiguity)
        {
          lambdaBegin_ = data.lambdaBegin(static_cast<int>(contactIndex));
        }
      }
      if(lambdaBegin_ < 0)
      {
        throw std::runtime_error("Rolling contact generator-regularization variables are missing for wheel: "
                                 + wheel_.description.name);
      }
    }

    void update(const std::vector<rbd::MultiBody> &,
                const std::vector<rbd::MultiBodyConfig> &,
                const tasks::qp::SolverData &) override
    {
    }

    const Eigen::MatrixXd & Q() const override { return Q_; }
    const Eigen::VectorXd & C() const override { return C_; }

    Impl & owner_;
    WheelData & wheel_;
    int lambdaBegin_ = -1;
    Eigen::MatrixXd Q_;
    Eigen::VectorXd C_;
  };

  struct RollingMotionConstr final : public tasks::qp::MotionConstr
  {
    RollingMotionConstr(Impl & owner,
                        const std::vector<rbd::MultiBody> & mbs,
                        int robotIndex,
                        const tasks::TorqueBound & torque,
                        const tasks::TorqueDBound & torqueRate,
                        double dt)
    : tasks::qp::MotionConstr(mbs, robotIndex, torque, torqueRate, dt), owner_(owner)
    {
    }

    void updateNrVars(const std::vector<rbd::MultiBody> & mbs, const tasks::qp::SolverData & data) override
    {
      tasks::qp::MotionConstr::updateNrVars(mbs, data);
      for(auto & wheel : owner_.wheelData) { wheel.lambdaBegin = -1; }
      const auto & contacts = data.allContacts();
      for(size_t contactIndex = 0; contactIndex < contacts.size(); ++contactIndex)
      {
        const auto & id = contacts[contactIndex].contactId;
        for(auto & wheel : owner_.wheelData)
        {
          if(id.r1Index == static_cast<int>(owner_.robotIndex) && id.r2Index == -1
             && id.r1BodyName == wheel.description.wheelBody && id.r2BodyName == terrainBody
             && id.ambiguityId == wheel.ambiguity)
          {
            if(contacts[contactIndex].nrLambda() != lambdasPerWheel)
            {
              throw std::runtime_error("Rolling contact lambda count changed for wheel: " + wheel.description.name);
            }
            wheel.lambdaBegin = data.lambdaBegin(static_cast<int>(contactIndex));
          }
        }
      }
      for(const auto & wheel : owner_.wheelData)
      {
        if(wheel.lambdaBegin < 0)
        {
          throw std::runtime_error("Rolling contact variables are missing for wheel: " + wheel.description.name);
        }
      }
    }

    void update(const std::vector<rbd::MultiBody> & mbs,
                const std::vector<rbd::MultiBodyConfig> & mbcs,
                const tasks::qp::SolverData & data) override
    {
      tasks::qp::MotionConstr::update(mbs, mbcs, data);
      owner_.updateForceColumns(A_);
    }

    Impl & owner_;
  };

  Impl(const mc_rbdyn::Robots & robotsIn,
       unsigned int robotIndexIn,
       std::vector<mc_rbdyn::RollingContactDescription> wheelsIn,
       double generatorRegularizationIn)
  : robots(robotsIn), robotIndex(robotIndexIn), robot(robots.robot(robotIndex)), wheels(std::move(wheelsIn)),
    ownerKey("RollingContactDynamics/" + robot.name()), generatorRegularization(generatorRegularizationIn)
  {
    if(wheels.empty()) { throw std::invalid_argument("RollingContactDynamicsConstraint requires at least one wheel"); }
    if(!std::isfinite(generatorRegularization) || generatorRegularization < 0.0)
    {
      throw std::invalid_argument(
          "RollingContactDynamicsConstraint generatorRegularization must be finite and non-negative");
    }
    std::unordered_set<std::string> names;
    wheelData.reserve(wheels.size());
    contacts.reserve(wheels.size());
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      auto & wheel = wheels[i];
      wheel.validate();
      if(!names.insert(wheel.name).second)
      {
        throw std::invalid_argument("Duplicate rolling contact name: " + wheel.name);
      }
      const int ambiguity = 10000 + static_cast<int>(i);
      wheelData.emplace_back(robot, wheel, ambiguity);
      std::vector<Eigen::Vector3d> placeholderPoints(pointsPerWheel, Eigen::Vector3d::Zero());
      tasks::qp::ContactId id(static_cast<int>(robotIndex), -1, wheel.wheelBody, terrainBody, ambiguity);
      contacts.emplace_back(id, std::move(placeholderPoints), Eigen::Matrix3d::Identity(), sva::PTransformd::Identity(),
                            generatorsPerPoint, wheel.friction);
    }
    regularizationTasks.reserve(wheelData.size());
    for(auto & wheel : wheelData)
    {
      wheel.modeBound = std::make_unique<ModeBound>(wheel.lambdaBegin, wheel.lambdaLower, wheel.lambdaUpper,
                                                    wheel.description.name);
      updateModeBound(wheel);
      regularizationTasks.push_back(
          std::make_unique<GeneratorRegularizationTask>(*this, wheel, generatorRegularization));
      wheel.regularizationTask = regularizationTasks.back().get();
    }
  }

  WheelData & wheel(const std::string & name)
  {
    for(auto & value : wheelData)
    {
      if(value.description.name == name) { return value; }
    }
    throw std::out_of_range("Unknown rolling contact wheel: " + name);
  }

  const WheelData & wheel(const std::string & name) const
  {
    for(const auto & value : wheelData)
    {
      if(value.description.name == name) { return value; }
    }
    throw std::out_of_range("Unknown rolling contact wheel: " + name);
  }

  void selectSlidingGenerator(WheelData & wheel)
  {
    const Eigen::Vector3d slip = wheel.result.slipVelocity;
    // Establish a direction on sliding entry, then retain it close to rest.
    // Re-selecting a full cone-edge force from measurement noise creates a
    // self-excited sign flip and prevents deterministic rolling recovery.
    const double slipDirectionEpsilon = wheel.slidingDirectionEstablished ? 5e-2 : 1e-6;
    if(slip.norm() <= slipDirectionEpsilon) { return; }
    const Eigen::Vector3d oppositeSlip = -slip.normalized();
    double best = -std::numeric_limits<double>::infinity();
    for(int generator = 0; generator < generatorsPerPoint; ++generator)
    {
      const double theta = 0.5 * pi * static_cast<double>(generator);
      const Eigen::Vector3d tangent = std::cos(theta) * wheel.result.rollingDirection
                                      + std::sin(theta) * wheel.result.lateralDirection;
      const double alignment = tangent.dot(oppositeSlip);
      if(alignment > best)
      {
        best = alignment;
        wheel.slidingGenerator = generator;
      }
    }
    wheel.slidingDirectionEstablished = true;
  }

  void updateModeBound(WheelData & wheel)
  {
    wheel.lambdaUpper.setConstant(std::numeric_limits<double>::infinity());
    if(wheel.description.mode == mc_rbdyn::RollingContactMode::Detached) { wheel.lambdaUpper.setZero(); }
    else if(wheel.description.mode == mc_rbdyn::RollingContactMode::Sliding)
    {
      wheel.lambdaUpper.setZero();
      for(int point = 0; point < pointsPerWheel; ++point)
      {
        wheel.lambdaUpper(point * generatorsPerPoint + wheel.slidingGenerator) =
            std::numeric_limits<double>::infinity();
      }
    }
  }

  void updateTVMMode(WheelData & wheel, const Eigen::Matrix3d & worldToBody)
  {
    for(int point = 0; point < pointsPerWheel; ++point)
    {
      auto & mode = wheel.tvmModes[static_cast<size_t>(point)];
      if(!mode) { continue; }
      if(wheel.description.mode == mc_rbdyn::RollingContactMode::Detached) { mode->detached(); }
      else if(wheel.description.mode == mc_rbdyn::RollingContactMode::Sliding)
      {
        mode->sliding(worldToBody * wheel.generators[point].col(wheel.slidingGenerator));
      }
      else { mode->unrestricted(); }
    }
  }

  void updateWheel(WheelData & wheel)
  {
    const auto & mb = robot.mb();
    const auto & mbc = robot.mbc();
    wheel.result = wheel.geometry.update(robot, terrainNormal, 0.0);
    const auto & bodyJacobian = wheel.wheelJacobian.bodyJacobian(mb, mbc);
    const unsigned int bodyIndex = robot.bodyIndexByName(wheel.description.wheelBody);
    const auto & X_0_b = mbc.bodyPosW[bodyIndex];
    const Eigen::Vector3d bodyOrigin = X_0_b.translation();
    const std::array<Eigen::Vector3d, pointsPerWheel> points = {wheel.result.lineStart, wheel.result.lineEnd};
    for(int pointIndex = 0; pointIndex < pointsPerWheel; ++pointIndex)
    {
      const Eigen::Vector3d pointInBody = X_0_b.rotation() * (points[pointIndex] - bodyOrigin);
      wheel.tvmPoints[static_cast<size_t>(pointIndex)] = sva::PTransformd(pointInBody);
      wheel.wheelJacobian.translateBodyJacobian(bodyJacobian, mbc, pointInBody, wheel.translatedJacobian);
      for(int generator = 0; generator < generatorsPerPoint; ++generator)
      {
        const double theta = 0.5 * pi * static_cast<double>(generator);
        const Eigen::Vector3d tangent = std::cos(theta) * wheel.result.rollingDirection
                                        + std::sin(theta) * wheel.result.lateralDirection;
        wheel.generators[pointIndex].col(generator) =
            (wheel.result.normalDirection + wheel.description.friction * tangent).normalized();
      }
      const Eigen::Matrix<double, 3, Eigen::Dynamic> generatorsInBody =
          X_0_b.rotation() * wheel.generators[pointIndex];
      wheel.generatorJacobian.noalias() = generatorsInBody.transpose() * wheel.translatedJacobian.bottomRows<3>();
      wheel.fullGeneratorJacobian.setZero();
      wheel.wheelJacobian.fullJacobian(mb, wheel.generatorJacobian, wheel.fullGeneratorJacobian);
      wheel.generalizedForce.middleCols(pointIndex * generatorsPerPoint, generatorsPerPoint) =
          wheel.fullGeneratorJacobian.transpose();
      if(wheel.tvmCones[static_cast<size_t>(pointIndex)])
      {
        wheel.tvmCones[static_cast<size_t>(pointIndex)]->frame(wheel.result, wheel.description.friction,
                                                               X_0_b.rotation());
      }
    }
    if(wheel.description.mode == mc_rbdyn::RollingContactMode::Sliding) { selectSlidingGenerator(wheel); }
    updateModeBound(wheel);
    updateTVMMode(wheel, X_0_b.rotation());
  }

  void updateAll()
  {
    for(auto & wheel : wheelData) { updateWheel(wheel); }
  }

  void updateForceColumns(Eigen::MatrixXd & dynamicsMatrix)
  {
    updateAll();
    const auto & mb = robot.mb();
    for(const auto & wheel : wheelData)
    {
      dynamicsMatrix.block(0, wheel.lambdaBegin, mb.nrDof(), lambdasPerWheel) = -wheel.generalizedForce;
    }
  }

  const mc_rbdyn::Robots & robots;
  unsigned int robotIndex;
  const mc_rbdyn::Robot & robot;
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  std::string ownerKey;
  double generatorRegularization;
  std::vector<WheelData> wheelData;
  std::vector<tasks::qp::UnilateralContact> contacts;
  // Owns each wheel's regularization task; WheelData::regularizationTask is a
  // non-owning pointer into this, kept index-aligned with wheelData (built in
  // lockstep in the constructor and never reordered afterwards).
  std::vector<std::unique_ptr<GeneratorRegularizationTask>> regularizationTasks;
  Eigen::Vector3d terrainNormal = Eigen::Vector3d::UnitZ();
};

RollingContactDynamicsConstraint::RollingContactDynamicsConstraint(
    const mc_rbdyn::Robots & robots,
    unsigned int robotIndex,
    double timeStep,
    std::vector<mc_rbdyn::RollingContactDescription> wheels,
    bool infTorque,
    double generatorRegularization)
: DynamicsConstraint(robots, robotIndex, timeStep, {0.1, 0.01, 0.5}, 0.5, infTorque, true),
  impl_(std::make_unique<Impl>(robots, robotIndex, std::move(wheels), generatorRegularization))
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    const auto & robot = robots.robot(robotIndex);
    tasks::TorqueBound torque(torqueLower(robot, infTorque), torqueUpper(robot, infTorque));
    tasks::TorqueDBound torqueRate(torqueRateLower(robot, infTorque), torqueRateUpper(robot, infTorque));
    auto motion = std::make_unique<Impl::RollingMotionConstr>(*impl_, robots.mbs(), static_cast<int>(robotIndex),
                                                              torque, torqueRate, timeStep);
    motion_constr_ = mc_rtc::make_void_ptr(std::move(motion));
  }
  else if(backend_ != QPSolver::Backend::TVM)
  {
    throw std::runtime_error("RollingContactDynamicsConstraint requires a concrete solver backend");
  }
}

RollingContactDynamicsConstraint::~RollingContactDynamicsConstraint() = default;

double RollingContactDynamicsConstraint::generatorRegularization() const noexcept
{
  return impl_->generatorRegularization;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> RollingContactDynamicsConstraint::generatorRegularizationQC(int count,
                                                                                                        double epsilon)
{
  if(count < 0)
  {
    throw std::invalid_argument(
        "RollingContactDynamicsConstraint::generatorRegularizationQC count must be non-negative");
  }
  if(!std::isfinite(epsilon) || epsilon < 0.0)
  {
    throw std::invalid_argument(
        "RollingContactDynamicsConstraint::generatorRegularizationQC epsilon must be finite and non-negative");
  }
  return {2.0 * epsilon * Eigen::MatrixXd::Identity(count, count), Eigen::VectorXd::Zero(count)};
}

void RollingContactDynamicsConstraint::update(QPSolver & solver)
{
  if(solver.backend() != backend_) { throw std::logic_error("RollingContactDynamicsConstraint backend mismatch"); }
  if(backend_ != QPSolver::Backend::TVM) { return; }
  impl_->updateAll();
  auto & dynamics = dynamicFunction();
  for(auto & wheel : impl_->wheelData)
  {
    const auto & frame = impl_->robot.frame(wheel.description.wheelBody);
    dynamics.updateContact(frame, wheel.tvmPoints);
  }
}

void RollingContactDynamicsConstraint::addToSolverImpl(QPSolver & solver)
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    auto & tasksSolver = TasksQPSolver::from_solver(solver);
    tasksSolver.registerRollingContacts(impl_->ownerKey, impl_->contacts);
    try
    {
      DynamicsConstraint::addToSolverImpl(solver);
      for(auto & wheel : impl_->wheelData)
      {
        tasksSolver.solver().addBoundConstraint(wheel.modeBound.get());
        tasksSolver.addTask(wheel.regularizationTask);
      }
    }
    catch(...)
    {
      tasksSolver.unregisterRollingContacts(impl_->ownerKey);
      throw;
    }
  }
  else
  {
    impl_->updateAll();
    auto & dynamics = dynamicFunction();
    for(auto & wheel : impl_->wheelData)
    {
      const auto & frame = impl_->robot.frame(wheel.description.wheelBody);
      const auto bodyIndex = impl_->robot.bodyIndexByName(wheel.description.wheelBody);
      if(wheel.tvmForces.numberOfVariables() == 0)
      {
        wheel.tvmForces = dynamics.addContact(frame, wheel.tvmPoints, 1.0);
        if(wheel.tvmForces.numberOfVariables() != pointsPerWheel)
        {
          throw std::runtime_error("TVM rolling contact must expose exactly two point forces");
        }
        for(int point = 0; point < pointsPerWheel; ++point)
        {
          wheel.tvmCones[static_cast<size_t>(point)] =
              std::make_shared<TVMRollingForceCone>(wheel.tvmForces[point]);
          wheel.tvmModes[static_cast<size_t>(point)] =
              std::make_shared<TVMRollingForceMode>(wheel.tvmForces[point]);
        }
      }
      else
      {
        dynamics.updateContact(frame, wheel.tvmPoints);
      }
      for(auto & cone : wheel.tvmCones)
      {
        cone->frame(wheel.result, wheel.description.friction, impl_->robot.mbc().bodyPosW[bodyIndex].rotation());
      }
      impl_->updateTVMMode(wheel, impl_->robot.mbc().bodyPosW[bodyIndex].rotation());
    }
    DynamicsConstraint::addToSolverImpl(solver);
    auto & problem = TVMQPSolver::from_solver(solver).problem();
    for(auto & wheel : impl_->wheelData)
    {
      for(int point = 0; point < pointsPerWheel; ++point)
      {
        wheel.tvmConeTasks[static_cast<size_t>(point)] =
            problem.add(wheel.tvmCones[static_cast<size_t>(point)] >= 0.0,
                        {tvm::requirements::PriorityLevel(0)});
        wheel.tvmModeTasks[static_cast<size_t>(point)] =
            problem.add(wheel.tvmModes[static_cast<size_t>(point)] == 0.0,
                        {tvm::requirements::PriorityLevel(0)});
      }
    }
  }
}

void RollingContactDynamicsConstraint::removeFromSolverImpl(QPSolver & solver)
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    auto & tasksSolver = TasksQPSolver::from_solver(solver);
    for(auto & wheel : impl_->wheelData)
    {
      tasksSolver.solver().removeBoundConstraint(wheel.modeBound.get());
      tasksSolver.removeTask(wheel.regularizationTask);
    }
    DynamicsConstraint::removeFromSolverImpl(solver);
    tasksSolver.unregisterRollingContacts(impl_->ownerKey);
  }
  else
  {
    auto & problem = TVMQPSolver::from_solver(solver).problem();
    for(auto & wheel : impl_->wheelData)
    {
      for(auto & task : wheel.tvmModeTasks)
      {
        if(task)
        {
          problem.remove(*task);
          TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(task);
          task.reset();
        }
      }
      for(auto & task : wheel.tvmConeTasks)
      {
        if(task)
        {
          problem.remove(*task);
          TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(task);
          task.reset();
        }
      }
    }
    DynamicsConstraint::removeFromSolverImpl(solver);
  }
}

const std::vector<mc_rbdyn::RollingContactDescription> & RollingContactDynamicsConstraint::wheels() const noexcept
{
  return impl_->wheels;
}

void RollingContactDynamicsConstraint::mode(const std::string & wheel, mc_rbdyn::RollingContactMode mode)
{
  auto & data = impl_->wheel(wheel);
  if(mode == mc_rbdyn::RollingContactMode::Sliding
     && data.description.mode != mc_rbdyn::RollingContactMode::Sliding)
  {
    data.slidingDirectionEstablished = false;
  }
  data.description.mode = mode;
  for(auto & description : impl_->wheels)
  {
    if(description.name == wheel)
    {
      description.mode = mode;
      break;
    }
  }
  impl_->updateModeBound(data);
}

mc_rbdyn::RollingContactMode RollingContactDynamicsConstraint::mode(const std::string & wheel) const
{
  return impl_->wheel(wheel).description.mode;
}

int RollingContactDynamicsConstraint::slidingGenerator(const std::string & wheel) const
{
  return impl_->wheel(wheel).slidingGenerator;
}

Eigen::Vector3d RollingContactDynamicsConstraint::slidingDirection(const std::string & wheel) const
{
  const auto & data = impl_->wheel(wheel);
  const double theta = 0.5 * pi * static_cast<double>(data.slidingGenerator);
  return std::cos(theta) * data.result.rollingDirection + std::sin(theta) * data.result.lateralDirection;
}

void RollingContactDynamicsConstraint::terrainNormal(const Eigen::Vector3d & normal)
{
  if(!normal.allFinite() || normal.norm() < 1e-12)
  {
    // Spelled as the configuration key, so the message names what the caller set.
    throw std::invalid_argument("RollingContactDynamicsConstraint terrainNormal must be finite and non-zero");
  }
  impl_->terrainNormal = normal.normalized();
}

const Eigen::Vector3d & RollingContactDynamicsConstraint::terrainNormal() const noexcept
{
  return impl_->terrainNormal;
}

int RollingContactDynamicsConstraint::lambdaBegin(const std::string & wheel) const
{
  if(backend_ != QPSolver::Backend::Tasks)
  {
    throw std::logic_error("lambdaBegin is only available for the Tasks backend");
  }
  return impl_->wheel(wheel).lambdaBegin;
}

int RollingContactDynamicsConstraint::lambdaCount(const std::string &) const
{
  if(backend_ != QPSolver::Backend::Tasks)
  {
    throw std::logic_error("lambdaCount is only available for the Tasks backend");
  }
  return lambdasPerWheel;
}

const Eigen::MatrixXd & RollingContactDynamicsConstraint::generalizedForceMatrix(const std::string & wheel) const
{
  return impl_->wheel(wheel).generalizedForce;
}

std::array<Eigen::Vector3d, 2> RollingContactDynamicsConstraint::endpointForces(
    const std::string & wheel,
    const Eigen::Ref<const Eigen::VectorXd> & lambda) const
{
  if(lambda.size() != lambdasPerWheel)
  {
    throw std::invalid_argument("Rolling wheel lambda must contain exactly eight values");
  }
  const auto & data = impl_->wheel(wheel);
  return {data.generators[0] * lambda.head<generatorsPerPoint>(),
          data.generators[1] * lambda.tail<generatorsPerPoint>()};
}

std::array<Eigen::Vector3d, 2> RollingContactDynamicsConstraint::endpointForces(const std::string & wheel) const
{
  if(backend_ != QPSolver::Backend::TVM)
  {
    throw std::logic_error("The no-argument endpointForces overload is only available for the TVM backend");
  }
  const auto & forces = impl_->wheel(wheel).tvmForces;
  if(forces.numberOfVariables() != pointsPerWheel)
  {
    throw std::logic_error("TVM rolling contact is not currently in the solver");
  }
  const auto & description = impl_->wheel(wheel).description;
  const auto bodyIndex = impl_->robot.bodyIndexByName(description.wheelBody);
  const Eigen::Matrix3d bodyToWorld = impl_->robot.mbc().bodyPosW[bodyIndex].rotation().transpose();
  return {bodyToWorld * forces[0]->value(), bodyToWorld * forces[1]->value()};
}

const Eigen::Matrix<double, 3, Eigen::Dynamic> & RollingContactDynamicsConstraint::forceGenerators(
    const std::string & wheel,
    unsigned int endpoint) const
{
  if(endpoint >= pointsPerWheel) { throw std::out_of_range("Rolling contact endpoint index must be zero or one"); }
  return impl_->wheel(wheel).generators[endpoint];
}

double RollingContactDynamicsConstraint::normalForce(const std::string & wheel,
                                                     const Eigen::Ref<const Eigen::VectorXd> & lambda) const
{
  const auto forces = endpointForces(wheel, lambda);
  const auto & normal = impl_->wheel(wheel).result.normalDirection;
  return normal.dot(forces[0] + forces[1]);
}

double RollingContactDynamicsConstraint::tangentialForce(const std::string & wheel,
                                                         const Eigen::Ref<const Eigen::VectorXd> & lambda) const
{
  const auto forces = endpointForces(wheel, lambda);
  const Eigen::Vector3d total = forces[0] + forces[1];
  const auto & normal = impl_->wheel(wheel).result.normalDirection;
  return (total - normal.dot(total) * normal).norm();
}

double RollingContactDynamicsConstraint::frictionMargin(const std::string & wheel,
                                                        const Eigen::Ref<const Eigen::VectorXd> & lambda) const
{
  return impl_->wheel(wheel).description.friction * normalForce(wheel, lambda) - tangentialForce(wheel, lambda);
}

double RollingContactDynamicsConstraint::normalForce(const std::string & wheel) const
{
  const auto forces = endpointForces(wheel);
  const auto & normal = impl_->wheel(wheel).result.normalDirection;
  return normal.dot(forces[0] + forces[1]);
}

double RollingContactDynamicsConstraint::tangentialForce(const std::string & wheel) const
{
  const auto forces = endpointForces(wheel);
  const Eigen::Vector3d total = forces[0] + forces[1];
  const auto & normal = impl_->wheel(wheel).result.normalDirection;
  return (total - normal.dot(total) * normal).norm();
}

double RollingContactDynamicsConstraint::frictionMargin(const std::string & wheel) const
{
  return impl_->wheel(wheel).description.friction * normalForce(wheel) - tangentialForce(wheel);
}

Eigen::Matrix<double, 6, 1> RollingContactDynamicsConstraint::resultantWrenchAtCarrier(
    const std::string & wheel,
    const Eigen::Ref<const Eigen::VectorXd> & lambda) const
{
  const auto & data = impl_->wheel(wheel);
  const auto forces = endpointForces(wheel, lambda);
  return mc_rbdyn::contactWrenchAtCarrier(data.result.carrierCenter, data.result.lineStart, forces[0])
         + mc_rbdyn::contactWrenchAtCarrier(data.result.carrierCenter, data.result.lineEnd, forces[1]);
}

Eigen::Matrix<double, 6, 1> RollingContactDynamicsConstraint::resultantWrenchAtCarrier(
    const std::string & wheel) const
{
  const auto & data = impl_->wheel(wheel);
  const auto forces = endpointForces(wheel);
  return mc_rbdyn::contactWrenchAtCarrier(data.result.carrierCenter, data.result.lineStart, forces[0])
         + mc_rbdyn::contactWrenchAtCarrier(data.result.carrierCenter, data.result.lineEnd, forces[1]);
}

const mc_rbdyn::RollingContactGeometryResult & RollingContactDynamicsConstraint::geometryResult(
    const std::string & wheel) const
{
  return impl_->wheel(wheel).result;
}

} // namespace mc_solver

namespace
{

static auto rolling_contact_dynamics_registered = mc_solver::ConstraintSetLoader::register_load_function(
    "rollingContactDynamics",
    [](mc_solver::QPSolver & solver, const mc_rtc::Configuration & config)
    {
      const auto robotIndex = mc_rbdyn::robotIndexFromConfig(config, solver.robots(), "rollingContactDynamics");
      auto constraint = std::make_shared<mc_solver::RollingContactDynamicsConstraint>(
          solver.robots(), robotIndex, solver.dt(), mc_solver::details::loadRollingWheels(config),
          config("infTorque", false),
          config("generatorRegularization",
                 mc_solver::RollingContactDynamicsConstraint::defaultGeneratorRegularization));
      constraint->terrainNormal(config("terrainNormal", Eigen::Vector3d{0.0, 0.0, 1.0}));
      return constraint;
    });

} // namespace
