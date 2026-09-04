/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_solver/ConstraintSet.h>

#include <mc_rbdyn/RollingContact.h>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace mc_rbdyn
{
struct Robots;
}

namespace mc_tvm
{
struct RollingContactFunction;
}

namespace mc_solver
{

/** Longitudinal enforcement used by RollingContactConstraint. */
enum class RollingContactLongitudinal
{
  Hard,
  Soft
};

/** Solver-independent settings for a group of rolling contacts. */
struct MC_SOLVER_DLLAPI RollingContactConstraintOptions
{
  Eigen::Vector3d terrainNormal = Eigen::Vector3d::UnitZ();
  RollingContactLongitudinal longitudinal = RollingContactLongitudinal::Hard;
  double velocityGain = 20.0;
  double rollingWeight = 1000.0;
  bool constrainNormal = true;
  /** Keep only the first lateral row for the differential-drive specialization. */
  bool differentialPlanar = false;
  /** Keep a stable two-row lateral basis for a planar multi-steering chassis.
   *
   * Four or more exact lateral rows are rank-redundant for compatible
   * steering and become spuriously full rank under arbitrarily small steering
   * tracking errors. `steeringPlanarWheels` should name a pair that remains
   * independent throughout the intended maneuvers. If it is empty, the first
   * and last active wheels are used. Residuals for every wheel remain
   * available through geometryResults().
   */
  bool steeringPlanar = false;
  std::vector<std::string> steeringPlanarWheels;

  /** Track the eight rotating velocities of a four-steering chassis.
   *
   * When true, each wheel contributes a soft row tracking its predicted rolling
   * rate, and each steering wheel a second soft row tracking its predicted
   * steering rate. These are the w_thetaDot and w_deltaDot objective terms of
   * the four-steering-wheel QP; the corresponding predicted-rate bounds are
   * already supplied by the joint velocity limits of KinematicsConstraint.
   */
  bool trackRotatingRates = false;
  /** Objective weight on (thetaDot^+ - thetaDot^ref). */
  double rollingRateWeight = 200.0;
  /** Objective weight on (deltaDot^+ - deltaDot^ref). */
  double steeringRateWeight = 200.0;

  void validate(size_t wheelCount) const;
};

/** Rolling acceleration rows and optional longitudinal quadratic objective.
 *
 * The Tasks implementation uses the native [alphaD, lambda] decision layout.
 * Hard rows occupy only the selected robot's alphaD block. Soft rolling is
 * represented by 1/2 ||A alphaD - b||^2 at `rollingWeight`; it does not invent
 * a slack decision variable. Contact-force dynamics are added separately by
 * the rolling force adapter.
 */
class MC_SOLVER_DLLAPI RollingContactConstraint : public ConstraintSet
{
public:
  RollingContactConstraint(const mc_rbdyn::Robots & robots,
                           unsigned int robotIndex,
                           std::vector<mc_rbdyn::RollingContactDescription> wheels,
                           RollingContactConstraintOptions options = {});
  ~RollingContactConstraint() override;

  /** Update rolling geometry and backend coefficients for the current cycle.
   * @param solver Solver that owns this constraint.
   */
  void update(QPSolver & solver) override;
  void addToSolverImpl(QPSolver & solver) override;
  void removeFromSolverImpl(QPSolver & solver) override;

  unsigned int robotIndex() const noexcept;
  const RollingContactConstraintOptions & options() const noexcept;
  const std::vector<mc_rbdyn::RollingContactDescription> & wheels() const noexcept;

  void terrainNormal(const Eigen::Vector3d & normal);
  const Eigen::Vector3d & terrainNormal() const noexcept;
  void velocityGain(double gain);
  void rollingWeight(double weight);
  double rollingWeight() const noexcept;

  /** Set the predicted-rate references for one wheel.
   *
   * @throws std::out_of_range if the wheel is unknown.
   * @throws std::invalid_argument if either value is not finite.
   */
  void rotatingRateReference(const std::string & wheel, double rollingRate, double steeringRate);
  double rollingRateReference(const std::string & wheel) const;
  double steeringRateReference(const std::string & wheel) const;

  /** Change one wheel's active contact mode without changing solver variables.
   *
   * An activation strictly between zero and one moves that wheel's kinematic
   * rows to the soft objective and scales their residual by sqrt(activation).
   * Detached contacts always have zero activation.
   */
  void mode(const std::string & wheel, mc_rbdyn::RollingContactMode mode, double activation = 1.0);
  mc_rbdyn::RollingContactMode mode(const std::string & wheel) const;
  void activation(const std::string & wheel, double activation);
  double activation(const std::string & wheel) const;

  /** Incremented whenever the active row policy changes. Decision-variable
   * dimensions are independent of this revision.
   */
  size_t layoutRevision() const noexcept;

  /** Logical matrices over only this robot's alphaD block. */
  const Eigen::MatrixXd & hardMatrix() const noexcept;
  const Eigen::VectorXd & hardRhs() const noexcept;
  const Eigen::MatrixXd & softMatrix() const noexcept;
  const Eigen::VectorXd & softRhs() const noexcept;
  const std::vector<std::string> & hardRowLabels() const noexcept;
  const std::vector<std::string> & softRowLabels() const noexcept;

  /** Full Tasks equality matrix, including structural-zero robot/lambda columns.
   *
   * @throws std::logic_error when this object was not created for Tasks.
   */
  const Eigen::MatrixXd & tasksFullHardMatrix() const;
  int tasksAlphaDBegin() const;

  /** TVM hard/soft linear functions over the common logical matrices. */
  const mc_tvm::RollingContactFunction & tvmHardFunction() const;
  const mc_tvm::RollingContactFunction * tvmSoftFunction() const;

  const std::vector<mc_rbdyn::RollingContactGeometryResult> & geometryResults() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mc_solver
