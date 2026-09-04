/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_solver/RollingContactConstraint.h>

#include <mc_solver/EqualityConstraint.h>
#include <mc_solver/ConstraintSetLoader.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_solver/TVMQPSolver.h>

#include <mc_rbdyn/Robots.h>

#include <mc_tvm/RollingContactFunction.h>

#include <mc_rtc/logging.h>

#include <Tasks/QPSolver.h>

#include <tvm/task_dynamics/None.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "RollingContactConfig.h"

namespace mc_solver
{

void RollingContactConstraintOptions::validate(size_t wheelCount) const
{
  if(wheelCount == 0) { throw std::invalid_argument("RollingContactConstraint requires at least one wheel"); }
  if(!terrainNormal.allFinite() || terrainNormal.norm() <= 1e-12)
  {
    throw std::invalid_argument("RollingContactConstraint terrainNormal must be finite and non-degenerate");
  }
  if(!std::isfinite(velocityGain) || velocityGain < 0.0)
  {
    throw std::invalid_argument("RollingContactConstraint velocityGain cannot be negative");
  }
  if(!std::isfinite(rollingWeight) || rollingWeight <= 0.0)
  {
    throw std::invalid_argument("RollingContactConstraint rollingWeight must be positive");
  }
  if(differentialPlanar && wheelCount != 2)
  {
    throw std::invalid_argument("RollingContactConstraint differentialPlanar requires exactly two wheels");
  }
  if(steeringPlanar && wheelCount < 3)
  {
    throw std::invalid_argument("RollingContactConstraint steeringPlanar requires at least three wheels");
  }
  if(!steeringPlanarWheels.empty() && (!steeringPlanar || steeringPlanarWheels.size() != 2
                                      || steeringPlanarWheels[0] == steeringPlanarWheels[1]))
  {
    throw std::invalid_argument(
        "RollingContactConstraint steeringPlanarWheels requires steeringPlanar and two distinct wheel names");
  }
  if(differentialPlanar && steeringPlanar)
  {
    throw std::invalid_argument("RollingContactConstraint planar specializations are mutually exclusive");
  }
  // isfinite first so NaN and +/-inf are both rejected; a bare x < 0.0 would let NaN through.
  if(!std::isfinite(rollingRateWeight) || rollingRateWeight < 0.0 || !std::isfinite(steeringRateWeight)
     || steeringRateWeight < 0.0)
  {
    throw std::invalid_argument("RollingContactConstraint rate weights must be non-negative and finite");
  }
}

struct RollingContactConstraint::Impl
{
  struct HardConstraint : public EqualityConstraintRobot
  {
    explicit HardConstraint(Impl & owner) : EqualityConstraintRobot(owner.robotIndex), owner_(owner) {}

    const Eigen::MatrixXd & A() const override { return owner_.hardA; }
    void compute() override {}
    int maxEq() const override { return static_cast<int>(4 * owner_.wheels.size()); }
    int nrEq() const override { return static_cast<int>(owner_.hardA.rows()); }
    std::string nameEq() const override { return "RollingContactConstraint"; }
    const Eigen::VectorXd & bEq() const override { return owner_.hardB; }
    int alphaDBegin() const noexcept { return this->ABegin_; }

    Impl & owner_;
  };

  struct SoftTask : public tasks::qp::Task
  {
    explicit SoftTask(Impl & owner) : tasks::qp::Task(owner.options.rollingWeight), owner_(owner)
    {
      const auto nrDof = static_cast<Eigen::Index>(owner.robot.mb().nrDof());
      Q_.setZero(nrDof, nrDof);
      C_.setZero(nrDof);
    }

    std::pair<int, int> begin() const override { return {alphaDBegin_, alphaDBegin_}; }
    void updateNrVars(const std::vector<rbd::MultiBody> &, const tasks::qp::SolverData & data) override
    {
      alphaDBegin_ = data.alphaDBegin(static_cast<int>(owner_.robotIndex));
    }
    void update(const std::vector<rbd::MultiBody> &,
                const std::vector<rbd::MultiBodyConfig> &,
                const tasks::qp::SolverData &) override
    {
      Q_.noalias() = owner_.softA.transpose() * owner_.softA;
      C_.noalias() = -owner_.softA.transpose() * owner_.softB;
    }
    const Eigen::MatrixXd & Q() const override { return Q_; }
    const Eigen::VectorXd & C() const override { return C_; }

    Impl & owner_;
    int alphaDBegin_ = 0;
    Eigen::MatrixXd Q_;
    Eigen::VectorXd C_;
  };

  Impl(const mc_rbdyn::Robots & robotsIn,
       unsigned int robotIndexIn,
       std::vector<mc_rbdyn::RollingContactDescription> descriptions,
       RollingContactConstraintOptions optionsIn,
       QPSolver::Backend backend)
  : robots(robotsIn), robotIndex(robotIndexIn), robot(robots.robot(robotIndex)), wheels(std::move(descriptions)),
    options(std::move(optionsIn))
  {
    options.validate(wheels.size());
    std::unordered_set<std::string> names;
    geometries.reserve(wheels.size());
    results.resize(wheels.size());
    for(const auto & wheel : wheels)
    {
      wheel.validate();
      if(!names.insert(wheel.name).second)
      {
        throw std::invalid_argument("Duplicate rolling contact name: " + wheel.name);
      }
      geometries.emplace_back(robot, wheel);
    }
    for(const auto & name : options.steeringPlanarWheels)
    {
      if(names.count(name) == 0)
      {
        throw std::invalid_argument("RollingContactConstraint steeringPlanarWheels contains unknown wheel: " + name);
      }
    }
    activations.resize(wheels.size(), 1.0);
    rollingRateReferences.assign(wheels.size(), 0.0);
    steeringRateReferences.assign(wheels.size(), 0.0);
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      if(wheels[i].mode == mc_rbdyn::RollingContactMode::Detached) { activations[i] = 0.0; }
    }
    buildRowLayout(false);
    if(backend == QPSolver::Backend::Tasks)
    {
      hardConstraint = std::make_unique<HardConstraint>(*this);
      softTask = std::make_unique<SoftTask>(*this);
    }
    else if(backend == QPSolver::Backend::TVM)
    {
      rebuildTVMFunctions();
    }
    else { throw std::runtime_error("RollingContactConstraint requires a concrete solver backend"); }
    updateGeometry();
  }

  struct Row
  {
    size_t wheel;
    int axis;
    bool fixedLongitudinal;
    bool driveLock;
    double scale;
  };

  static const char * axisName(int axis)
  {
    if(axis == 0) { return "longitudinal"; }
    if(axis == 1) { return "lateral"; }
    if(axis == 2) { return "normal"; }
    return "drive-lock";
  }

  void addRow(size_t wheel, int axis, bool hard, bool fixedLongitudinal = false, bool driveLock = false)
  {
    const double scale = hard ? 1.0 : std::sqrt(activations[wheel]);
    auto & rows = hard ? hardRows : softRows;
    auto & labels = hard ? hardLabels : softLabels;
    rows.push_back({wheel, axis, fixedLongitudinal, driveLock, scale});
    labels.push_back(wheels[wheel].name + "/" + axisName(axis));
  }

  void addModeRow(size_t wheel, int axis, bool nominallyHard, bool fixedLongitudinal = false, bool driveLock = false)
  {
    const double activation = activations[wheel];
    if(activation <= 0.0) { return; }
    const bool hard = nominallyHard && activation >= 1.0 - 1e-12;
    addRow(wheel, axis, hard, fixedLongitudinal, driveLock);
  }

  void buildRowLayout(bool markDirty = true)
  {
    const auto previousHardLabels = hardLabels;
    const auto previousSoftLabels = softLabels;
    hardRows.clear();
    softRows.clear();
    hardLabels.clear();
    softLabels.clear();
    // Keep the specialization's canonical order: all longitudinal rows, then
    // independent lateral rows, then normal rows handled by this object.
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      const auto mode = wheels[i].mode;
      if(mode == mc_rbdyn::RollingContactMode::Rolling)
      {
        addModeRow(i, 0, options.longitudinal == RollingContactLongitudinal::Hard);
      }
      else if(mode == mc_rbdyn::RollingContactMode::Fixed) { addModeRow(i, 0, true, true); }
    }
    if(options.differentialPlanar)
    {
      size_t selected = wheels.size();
      double selectedActivation = -1.0;
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        const auto mode = wheels[i].mode;
        if((mode == mc_rbdyn::RollingContactMode::Rolling || mode == mc_rbdyn::RollingContactMode::Fixed)
           && activations[i] > selectedActivation)
        {
          selected = i;
          selectedActivation = activations[i];
        }
      }
      if(selected != wheels.size()) { addModeRow(selected, 1, true); }
    }
    else if(options.steeringPlanar)
    {
      std::vector<size_t> active;
      active.reserve(wheels.size());
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        const auto mode = wheels[i].mode;
        if((mode == mc_rbdyn::RollingContactMode::Rolling || mode == mc_rbdyn::RollingContactMode::Fixed)
           && activations[i] > 0.0)
        {
          active.push_back(i);
        }
      }
      std::vector<size_t> selected;
      selected.reserve(2);
      for(const auto & name : options.steeringPlanarWheels)
      {
        const auto wheel = std::find_if(active.begin(), active.end(), [this, &name](size_t i)
                                        { return wheels[i].name == name; });
        if(wheel != active.end()) { selected.push_back(*wheel); }
      }
      for(const size_t wheel : active)
      {
        if(selected.size() == 2) { break; }
        if(std::find(selected.begin(), selected.end(), wheel) == selected.end()) { selected.push_back(wheel); }
      }
      for(const size_t wheel : selected) { addModeRow(wheel, 1, true); }
    }
    else
    {
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        const auto mode = wheels[i].mode;
        if(mode == mc_rbdyn::RollingContactMode::Rolling || mode == mc_rbdyn::RollingContactMode::Fixed)
        {
          addModeRow(i, 1, true);
        }
      }
    }
    if(options.constrainNormal)
    {
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        if(wheels[i].mode != mc_rbdyn::RollingContactMode::Detached) { addModeRow(i, 2, true); }
      }
    }
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      if(wheels[i].mode == mc_rbdyn::RollingContactMode::Fixed) { addModeRow(i, 3, true, false, true); }
    }
    hardA.setZero(static_cast<Eigen::Index>(hardRows.size()), robot.mb().nrDof());
    hardB.setZero(static_cast<Eigen::Index>(hardRows.size()));
    softA.setZero(static_cast<Eigen::Index>(softRows.size()), robot.mb().nrDof());
    softB.setZero(static_cast<Eigen::Index>(softRows.size()));
    if(markDirty)
    {
      const bool topologyChanged = hardLabels != previousHardLabels || softLabels != previousSoftLabels;
      if(topologyChanged)
      {
        ++layoutRevision;
        tvmLayoutDirty = true;
      }
    }
  }

  void fillRow(const Row & row, Eigen::MatrixXd & matrix, Eigen::VectorXd & rhs, Eigen::Index rowIndex) const
  {
    auto A = matrix.row(rowIndex);
    if(row.driveLock)
    {
      A = geometries[row.wheel].kinematics().wheelSelector;
      const double rate = geometries[row.wheel].kinematics().wheelSelector.dot(
          geometries[row.wheel].kinematics().generalizedVelocity);
      rhs(rowIndex) = -options.velocityGain * rate;
    }
    else
    {
      const auto & result = results[row.wheel];
      A = result.rollingMatrix.row(row.axis);
      rhs(rowIndex) = result.rhs(row.axis);
    }
    if(row.fixedLongitudinal)
    {
      const auto & result = results[row.wheel];
      const auto & wheel = wheels[row.wheel];
      A.noalias() += wheel.radius * wheel.spinSign * geometries[row.wheel].kinematics().wheelSelector;
      rhs(rowIndex) = -result.accelerationBias(row.axis)
                      - options.velocityGain * result.rollingDirection.dot(result.carrierVelocity);
    }
    A *= row.scale;
    rhs(rowIndex) *= row.scale;
  }

  size_t wheelIndex(const std::string & name) const
  {
    const auto it =
        std::find_if(wheels.begin(), wheels.end(), [&name](const auto & wheel) { return wheel.name == name; });
    if(it == wheels.end()) { throw std::out_of_range("Unknown rolling contact: " + name); }
    return static_cast<size_t>(std::distance(wheels.begin(), it));
  }

  void rebuildTVMFunctions()
  {
    tvmHard.reset();
    tvmSoft.reset();
    if(hardA.rows() != 0) { tvmHard = std::make_shared<mc_tvm::RollingContactFunction>(robot, hardA.rows()); }
    if(softA.rows() != 0) { tvmSoft = std::make_shared<mc_tvm::RollingContactFunction>(robot, softA.rows()); }
    tvmLayoutDirty = false;
  }

  void updateGeometry()
  {
    for(size_t i = 0; i < geometries.size(); ++i)
    {
      results[i] = geometries[i].update(robot, options.terrainNormal, options.velocityGain);
    }
    for(Eigen::Index i = 0; i < hardA.rows(); ++i)
    {
      fillRow(hardRows[static_cast<size_t>(i)], hardA, hardB, i);
    }
    for(Eigen::Index i = 0; i < softA.rows(); ++i)
    {
      fillRow(softRows[static_cast<size_t>(i)], softA, softB, i);
    }
    if(tvmHard) { tvmHard->set(hardA, hardB); }
    if(tvmSoft) { tvmSoft->set(softA, softB); }
  }

  const mc_rbdyn::Robots & robots;
  unsigned int robotIndex;
  const mc_rbdyn::Robot & robot;
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  RollingContactConstraintOptions options;
  std::vector<mc_rbdyn::RollingContactRobotGeometry> geometries;
  std::vector<mc_rbdyn::RollingContactGeometryResult> results;
  std::vector<double> activations;
  std::vector<double> rollingRateReferences;
  std::vector<double> steeringRateReferences;
  /** Control period cached from solver.dt() in update().
   *
   * fillRow() is const and also runs from the constructor, where no solver is
   * available, so the period cannot be read at the point of use. Currently
   * unused: the predicted-rate rows that consume it are not emitted yet.
   */
  double dt = 0.0;
  std::vector<Row> hardRows;
  std::vector<Row> softRows;
  std::vector<std::string> hardLabels;
  std::vector<std::string> softLabels;
  Eigen::MatrixXd hardA;
  Eigen::VectorXd hardB;
  Eigen::MatrixXd softA;
  Eigen::VectorXd softB;
  std::unique_ptr<HardConstraint> hardConstraint;
  std::unique_ptr<SoftTask> softTask;
  mc_tvm::RollingContactFunctionPtr tvmHard;
  mc_tvm::RollingContactFunctionPtr tvmSoft;
  tvm::TaskWithRequirementsPtr tvmHardTask;
  tvm::TaskWithRequirementsPtr tvmSoftTask;
  size_t layoutRevision = 0;
  bool tvmLayoutDirty = false;
  bool inSolver = false;
};

RollingContactConstraint::RollingContactConstraint(const mc_rbdyn::Robots & robots,
                                                   unsigned int robotIndex,
                                                   std::vector<mc_rbdyn::RollingContactDescription> wheels,
                                                   RollingContactConstraintOptions options)
: impl_(std::make_unique<Impl>(robots, robotIndex, std::move(wheels), std::move(options), backend_))
{
}

RollingContactConstraint::~RollingContactConstraint() = default;

void RollingContactConstraint::update(QPSolver & solver)
{
  if(solver.backend() != backend_) { throw std::logic_error("RollingContactConstraint backend mismatch"); }
  if(backend_ == QPSolver::Backend::TVM && impl_->tvmLayoutDirty)
  {
    auto & problem = TVMQPSolver::from_solver(solver).problem();
    if(impl_->tvmSoftTask)
    {
      problem.remove(*impl_->tvmSoftTask);
      TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(impl_->tvmSoftTask);
      impl_->tvmSoftTask.reset();
    }
    if(impl_->tvmHardTask)
    {
      problem.remove(*impl_->tvmHardTask);
      TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(impl_->tvmHardTask);
      impl_->tvmHardTask.reset();
    }
    impl_->rebuildTVMFunctions();
    if(impl_->inSolver)
    {
      if(impl_->tvmHard)
      {
        impl_->tvmHardTask = problem.add(impl_->tvmHard == 0.0, tvm::task_dynamics::None(),
                                         {tvm::requirements::PriorityLevel(0)});
      }
      if(impl_->tvmSoft)
      {
        impl_->tvmSoftTask = problem.add(
            impl_->tvmSoft == 0.0, tvm::task_dynamics::None(),
            {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(impl_->options.rollingWeight)});
      }
    }
  }
  impl_->updateGeometry();
}

void RollingContactConstraint::addToSolverImpl(QPSolver & solver)
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    auto & tasksSolver = TasksQPSolver::from_solver(solver);
    // A standalone/headless solver may not have called setContacts yet. Tasks'
    // lightweight updateNrVars path assumes SolverData was initialized once.
    if(tasksSolver.data().nrVars() == 0) { tasksSolver.updateNrVars(); }
    tasksSolver.addConstraint(impl_->hardConstraint.get());
    tasksSolver.addTask(impl_->softTask.get());
  }
  else
  {
    auto & problem = TVMQPSolver::from_solver(solver).problem();
    if(impl_->tvmHard)
    {
      impl_->tvmHardTask = problem.add(impl_->tvmHard == 0.0, tvm::task_dynamics::None(),
                                       {tvm::requirements::PriorityLevel(0)});
    }
    if(impl_->tvmSoft)
    {
      impl_->tvmSoftTask = problem.add(
          impl_->tvmSoft == 0.0, tvm::task_dynamics::None(),
          {tvm::requirements::PriorityLevel(1), tvm::requirements::Weight(impl_->options.rollingWeight)});
    }
  }
  impl_->inSolver = true;
}

void RollingContactConstraint::removeFromSolverImpl(QPSolver & solver)
{
  if(backend_ == QPSolver::Backend::Tasks)
  {
    auto & tasksSolver = TasksQPSolver::from_solver(solver);
    tasksSolver.removeTask(impl_->softTask.get());
    tasksSolver.removeConstraint(impl_->hardConstraint.get());
  }
  else
  {
    auto & problem = TVMQPSolver::from_solver(solver).problem();
    if(impl_->tvmSoftTask)
    {
      problem.remove(*impl_->tvmSoftTask);
      TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(impl_->tvmSoftTask);
      impl_->tvmSoftTask.reset();
    }
    if(impl_->tvmHardTask)
    {
      problem.remove(*impl_->tvmHardTask);
      TVMQPSolver::from_solver(solver).deferDestructionUntilNextSolve(impl_->tvmHardTask);
      impl_->tvmHardTask.reset();
    }
  }
  impl_->inSolver = false;
}

unsigned int RollingContactConstraint::robotIndex() const noexcept
{
  return impl_->robotIndex;
}

const RollingContactConstraintOptions & RollingContactConstraint::options() const noexcept
{
  return impl_->options;
}

const std::vector<mc_rbdyn::RollingContactDescription> & RollingContactConstraint::wheels() const noexcept
{
  return impl_->wheels;
}

void RollingContactConstraint::terrainNormal(const Eigen::Vector3d & normal)
{
  auto next = impl_->options;
  next.terrainNormal = normal;
  next.validate(impl_->wheels.size());
  impl_->options.terrainNormal = normal;
}

const Eigen::Vector3d & RollingContactConstraint::terrainNormal() const noexcept
{
  return impl_->options.terrainNormal;
}

void RollingContactConstraint::velocityGain(double gain)
{
  auto next = impl_->options;
  next.velocityGain = gain;
  next.validate(impl_->wheels.size());
  impl_->options.velocityGain = gain;
}

void RollingContactConstraint::rollingWeight(double weight)
{
  auto next = impl_->options;
  next.rollingWeight = weight;
  next.validate(impl_->wheels.size());
  impl_->options.rollingWeight = weight;
  if(impl_->softTask) { impl_->softTask->weight(weight); }
  if(impl_->tvmSoftTask) { impl_->tvmSoftTask->requirements.weight() = weight; }
}

double RollingContactConstraint::rollingWeight() const noexcept
{
  return impl_->options.rollingWeight;
}

void RollingContactConstraint::mode(const std::string & wheel, mc_rbdyn::RollingContactMode mode, double activation)
{
  if(!std::isfinite(activation) || activation < 0.0 || activation > 1.0)
  {
    throw std::invalid_argument("Rolling contact activation must be finite and in [0, 1]");
  }
  const size_t index = impl_->wheelIndex(wheel);
  if(mode == mc_rbdyn::RollingContactMode::Detached) { activation = 0.0; }
  if(impl_->wheels[index].mode == mode && impl_->activations[index] == activation) { return; }
  impl_->wheels[index].mode = mode;
  impl_->activations[index] = activation;
  impl_->buildRowLayout();
}

mc_rbdyn::RollingContactMode RollingContactConstraint::mode(const std::string & wheel) const
{
  return impl_->wheels[impl_->wheelIndex(wheel)].mode;
}

void RollingContactConstraint::activation(const std::string & wheel, double activation)
{
  const size_t index = impl_->wheelIndex(wheel);
  mode(wheel, impl_->wheels[index].mode, activation);
}

double RollingContactConstraint::activation(const std::string & wheel) const
{
  return impl_->activations[impl_->wheelIndex(wheel)];
}

void RollingContactConstraint::rotatingRateReference(const std::string & wheel, double rollingRate, double steeringRate)
{
  if(!std::isfinite(rollingRate) || !std::isfinite(steeringRate))
  {
    throw std::invalid_argument("Rolling contact rate references must be finite");
  }
  const size_t index = impl_->wheelIndex(wheel);
  impl_->rollingRateReferences[index] = rollingRate;
  impl_->steeringRateReferences[index] = steeringRate;
}

double RollingContactConstraint::rollingRateReference(const std::string & wheel) const
{
  return impl_->rollingRateReferences[impl_->wheelIndex(wheel)];
}

double RollingContactConstraint::steeringRateReference(const std::string & wheel) const
{
  return impl_->steeringRateReferences[impl_->wheelIndex(wheel)];
}

size_t RollingContactConstraint::layoutRevision() const noexcept
{
  return impl_->layoutRevision;
}

const Eigen::MatrixXd & RollingContactConstraint::hardMatrix() const noexcept
{
  return impl_->hardA;
}

const Eigen::VectorXd & RollingContactConstraint::hardRhs() const noexcept
{
  return impl_->hardB;
}

const Eigen::MatrixXd & RollingContactConstraint::softMatrix() const noexcept
{
  return impl_->softA;
}

const Eigen::VectorXd & RollingContactConstraint::softRhs() const noexcept
{
  return impl_->softB;
}

const std::vector<std::string> & RollingContactConstraint::hardRowLabels() const noexcept
{
  return impl_->hardLabels;
}

const std::vector<std::string> & RollingContactConstraint::softRowLabels() const noexcept
{
  return impl_->softLabels;
}

const Eigen::MatrixXd & RollingContactConstraint::tasksFullHardMatrix() const
{
  if(backend_ != QPSolver::Backend::Tasks)
  {
    throw std::logic_error("tasksFullHardMatrix is only available for the Tasks backend");
  }
  return impl_->hardConstraint->AEq();
}

int RollingContactConstraint::tasksAlphaDBegin() const
{
  if(backend_ != QPSolver::Backend::Tasks)
  {
    throw std::logic_error("tasksAlphaDBegin is only available for the Tasks backend");
  }
  return impl_->hardConstraint->alphaDBegin();
}

const mc_tvm::RollingContactFunction & RollingContactConstraint::tvmHardFunction() const
{
  if(backend_ != QPSolver::Backend::TVM)
  {
    throw std::logic_error("tvmHardFunction is only available for the TVM backend");
  }
  if(!impl_->tvmHard) { throw std::logic_error("tvmHardFunction has no active hard rows"); }
  return *impl_->tvmHard;
}

const mc_tvm::RollingContactFunction * RollingContactConstraint::tvmSoftFunction() const
{
  if(backend_ != QPSolver::Backend::TVM)
  {
    throw std::logic_error("tvmSoftFunction is only available for the TVM backend");
  }
  return impl_->tvmSoft.get();
}

const std::vector<mc_rbdyn::RollingContactGeometryResult> & RollingContactConstraint::geometryResults() const noexcept
{
  return impl_->results;
}

} // namespace mc_solver

namespace
{

static auto rolling_contact_registered = mc_solver::ConstraintSetLoader::register_load_function(
    "rollingContact",
    [](mc_solver::QPSolver & solver, const mc_rtc::Configuration & config)
    {
      const auto robotIndex = mc_rbdyn::robotIndexFromConfig(config, solver.robots(), "rollingContact");
      mc_solver::RollingContactConstraintOptions options;
      options.terrainNormal = config("terrainNormal", Eigen::Vector3d{0.0, 0.0, 1.0});
      const std::string longitudinal = config("longitudinal", std::string{"hard"});
      if(longitudinal == "hard") { options.longitudinal = mc_solver::RollingContactLongitudinal::Hard; }
      else if(longitudinal == "soft") { options.longitudinal = mc_solver::RollingContactLongitudinal::Soft; }
      else { throw std::invalid_argument("Rolling contact longitudinal must be hard or soft"); }
      options.velocityGain = config("velocityGain", 20.0);
      options.rollingWeight = config("rollingWeight", 1000.0);
      options.constrainNormal = config("constrainNormal", true);
      options.differentialPlanar = config("differentialPlanar", false);
      options.steeringPlanar = config("steeringPlanar", false);
      options.steeringPlanarWheels = config("steeringPlanarWheels", std::vector<std::string>{});
      options.trackRotatingRates = config("trackRotatingRates", false);
      options.rollingRateWeight = config("rollingRateWeight", 200.0);
      options.steeringRateWeight = config("steeringRateWeight", 200.0);
      return std::make_shared<mc_solver::RollingContactConstraint>(
          solver.robots(), robotIndex, mc_solver::details::loadRollingWheels(config), options);
    });

} // namespace
