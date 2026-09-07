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
  /** Keep only the first lateral row for the differential-drive specialization.
   *
   * A two-wheel chassis contributes two lateral rows to three planar DOF and is
   * not over-determined, so selecting one independent row remains correct there.
   * This is not the case for four steering wheels: see softLateralRows.
   */
  bool differentialPlanar = false;
  /** Soften the lateral rows with one slack per wheel.
   *
   * Four lateral rows act on three planar chassis DOF. For steering angles not
   * coordinated on a common ICR the block has full row rank 3, so hard rows
   * admit only the zero twist and freeze the chassis; when they are
   * coordinated, consistency further needs c in range(A), a condition on the
   * measured steering rates that no controller can guarantee. The slacks make
   * the QP always feasible and report the incompatibility through
   * lateralSlack().
   *
   * The Tasks decision layout has no free slack columns, so the softening is
   * realised by moving the lateral rows into the soft objective at
   * `lateralSlackWeight`. That is the explicit-slack problem written out:
   * minimising w ||A x - c||^2 is minimising w ||sigma||^2 subject to
   * A x - c = sigma. No decision variable and no bound is added either way.
   */
  bool softLateralRows = false;
  /** Objective weight on each lateral slack, in the units of an acceleration task.
   *
   * A single common weight across wheels, so the least-squares solution does not
   * preferentially skid one wheel. Set it large relative to the tracking weights
   * so the lateral rows behave as a high-priority task. Unlike the rate weights
   * below this one carries no dt, because a lateral row acts on alphaD directly.
   *
   * Must be positive and finite whether or not `softLateralRows` is set. This
   * default is a placeholder for a caller that does not tune it; the
   * RollingContact sample controller documents and sets its own swept value.
   */
  double lateralSlackWeight = 1e5;

  /** Track the eight rotating velocities of a four-steering chassis.
   *
   * When true, each wheel contributes a soft row tracking its predicted rolling
   * rate, and each steering wheel a second soft row tracking its predicted
   * steering rate. These are the w_thetaDot and w_deltaDot objective terms of
   * the four-steering-wheel QP; the corresponding predicted-rate bounds are
   * already supplied by the joint velocity limits of KinematicsConstraint.
   *
   * Like every other soft row, a rate row is additionally scaled by
   * sqrt(activation), so a wheel at partial activation tracks its rates at a
   * proportionally reduced weight.
   */
  bool trackRotatingRates = false;
  /** Objective weight on (thetaDot^+ - thetaDot^ref); zero emits no rolling-rate row.
   *
   * A rate row's coefficients carry dt, because the tracked quantity is the
   * prediction rate^+ = rate + dt * S * alphaD. The weight therefore enters the
   * QP objective multiplied by dt^2, and is NOT comparable to a task weight on
   * alphaD: authority equivalent to an acceleration-task weight of 1000 at
   * dt = 0.005 is 1000 / dt^2 = 4e7, not 1000. At a plain 200 the row
   * contributes 200 * dt^2 = 5e-3 and any ordinary task outranks it.
   */
  double rollingRateWeight = 200.0;
  /** Objective weight on (deltaDot^+ - deltaDot^ref); zero emits no steering-rate row.
   *
   * Scales with dt^2 exactly as rollingRateWeight does; see its documentation.
   */
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
   * `steeringRate` is ignored for wheels without a steering joint.
   *
   * The reference is stored unconditionally, but it only reaches the QP while the
   * wheel actually carries a rate row: a Detached wheel, a wheel at zero
   * activation, or a zero rate weight silently drops it, and `trackRotatingRates`
   * must be on at all. Inspect softRowLabels() if the caller needs to know.
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

  /** Realised lateral slack sigma = A_lat * alphaD - c_lat, one entry per active lateral row.
   *
   * Unscaled: the entries are in the acceleration units of the lateral rows, not
   * in whatever objective scale the rows carry, so they stay comparable across
   * weight changes. Entry order follows lateralSlackLabels().
   *
   * With `softLateralRows` this is the incompatibility the QP chose to accept;
   * with hard rows it is a constraint violation and the solver drives it to
   * zero, so the same accessor reads both configurations.
   *
   * @throws std::invalid_argument if @p alphaD does not have one entry per DoF.
   */
  Eigen::VectorXd lateralSlack(const Eigen::VectorXd & alphaD) const;
  /** Names of the wheels carrying a lateral row, in slack order. */
  const std::vector<std::string> & lateralSlackLabels() const noexcept;

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
