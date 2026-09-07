/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_solver/DynamicsConstraint.h>

#include <mc_rbdyn/RollingContact.h>

#include <Eigen/Core>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mc_solver
{

/** Whole-body dynamics with stable, state-dependent rolling line forces.
 *
 * Each wheel owns two line endpoints ordered [lineStart, lineEnd]. Each point
 * uses four positive friction-pyramid multipliers ordered by tangent azimuth
 * [0, pi/2, pi, 3*pi/2] in the (t,l) plane. The force direction is
 * normalize(n + mu * (cos(theta) t + sin(theta) l)).
 *
 * The Tasks implementation reuses native contact lambda variables and a
 * MotionConstr-derived dynamics object. It overwrites only the registered
 * rolling-contact generalized-force columns each cycle, preserving ordinary
 * contact behavior, torque reconstruction and torque bounds.
 *
 * Each point's four positive generators are a V-representation of the same
 * friction pyramid on the Tasks backend: opposite-sign generator pairs
 * normalize to the same multiple of the normal direction
 * (generators[0] + generators[2] == generators[1] + generators[3]), so
 * (1,-1,1,-1) is an exact null direction of the map from one point's four
 * multipliers to its 3D force. Nothing else in this file or in
 * RollingContactConstraint.cpp contributes an objective term over lambda, so
 * in principle that leaves the assembled Hessian's lambda block singular
 * along that direction -- except that the Tasks library applies its own
 * unconditional diagonal floor (Tasks/src/GenQPUtils.h's DIAG_CONSTANT =
 * 1e-4, added to every decision variable's diagonal entry regardless of what
 * any task contributes, whenever that entry is smaller than 1e-4).
 * Measured directly (see
 * RollingContactDynamicsConstraintGeneratorRegularizationQP02ZeroReliesOnlyOnTheLibraryFloor
 * in tests/testRollingContactSolver.cpp): at generatorRegularization = 0 the
 * smallest eigenvalue of that block is exactly the floor, 1e-4 -- positive,
 * not singular. So on the Tasks backend the term this class calls
 * generatorRegularization (a Tikhonov penalty eps * ||lambda||^2 per wheel,
 * see generatorRegularizationQC) is not needed for conditioning: the solver
 * already prevents the singularity the formulation would otherwise have.
 *
 * The option and its machinery are kept, and are exercised by the
 * RollingContactDynamicsConstraintGeneratorRegularization* tests in
 * tests/testRollingContactSolver.cpp, for a backend or solver that does not
 * apply a diagonal floor of its own; there the term would be needed and this
 * is where it would be configured. But enabling it is not free: measurement
 * (RollingContactDynamicsConstraintGeneratorRegularizationFRI01BiasAtEnabledAndTenX)
 * shows the conditioning margin and the bias it puts on the solved contact
 * force both grow with epsilon at roughly the same rate, so there is no
 * epsilon that is simultaneously a real conditioning win and negligible
 * against the physical force -- at epsilon = 2e-4 (four times the library
 * floor) the solved tangential force already moved by 0.008936 N on a
 * solved 2.17860 N (41% of a 1% budget), and the four-steering
 * target-tracking error degraded from 1.67e-4 (at epsilon = 0) to 6.692e-4
 * (see RollingTasksFourSteeringAckermannTargetWithDynamics). Because mc_rtc's
 * Tasks backend already floors the Hessian diagonal unconditionally,
 * defaultGeneratorRegularization is 0.0: the default configuration should
 * not pay a measurable physical-force bias for a singularity its solver
 * already prevents.
 *
 * The TVM backend parameterizes each point's contact force directly as a 3D
 * vector (TVMRollingForceCone / TVMRollingForceMode), which has no
 * generator-basis redundancy to remove, so generatorRegularization is stored
 * for both backends but only has an effect on Tasks.
 */
class MC_SOLVER_DLLAPI RollingContactDynamicsConstraint : public DynamicsConstraint
{
public:
  /** Default Tikhonov weight on each wheel's eight generator multipliers
   * (see the class documentation and generatorRegularizationQC). Zero: the
   * Tasks backend floors the Hessian diagonal at 1e-4 unconditionally (see
   * the class documentation and
   * ...GeneratorRegularizationQP02ZeroReliesOnlyOnTheLibraryFloor), so the
   * term is not needed for conditioning there, and it is not free to enable
   * -- measured at epsilon = 2e-4 (four times that floor), it biased the
   * solved tangential force by 0.008936 N on a solved 2.17860 N (41% of a 1%
   * budget) and degraded the four-steering target-tracking error from
   * 1.67e-4 to 6.692e-4 (RollingTasksFourSteeringAckermannTargetWithDynamics).
   * Margin and bias both grow with epsilon at roughly the same rate, so no
   * positive value is simultaneously an unambiguous conditioning win and
   * negligible against the physical force.
   *
   * The option is kept, and is not dead: a solver or backend without an
   * unconditional diagonal floor of its own would need it for conditioning,
   * and RollingContactDynamicsConstraintGeneratorRegularizationQP01EnabledClearsTheLibraryFloor
   * and ...FRI01BiasAtEnabledAndTenX in tests/testRollingContactSolver.cpp
   * measure exactly that trade-off at an explicit enabled value (2e-4) so a
   * caller who does need it has a starting point and a measured cost. See
   * those tests and the class documentation for the numbers.
   */
  static constexpr double defaultGeneratorRegularization = 0.0;

  RollingContactDynamicsConstraint(const mc_rbdyn::Robots & robots,
                                   unsigned int robotIndex,
                                   double timeStep,
                                   std::vector<mc_rbdyn::RollingContactDescription> wheels,
                                   bool infTorque = false,
                                   double generatorRegularization = defaultGeneratorRegularization);
  ~RollingContactDynamicsConstraint() override;

  /** Configured Tikhonov weight on the generator multipliers, see the class
   * documentation and generatorRegularizationQC. */
  double generatorRegularization() const noexcept;

  /** Q (2*epsilon*Identity(count)) and C (Zero(count)) for the Tikhonov term
   * epsilon * ||lambda||^2 added to the QP objective over `count` generator
   * multipliers. Shared by the Tasks-backend per-wheel regularization task
   * and by tests/testRollingContactSolver.cpp, so a test failure means the
   * two disagree, not that the test re-derived the formula it is checking.
   * @throws std::invalid_argument if count is negative, or epsilon is
   * negative or not finite.
   */
  static std::pair<Eigen::MatrixXd, Eigen::VectorXd> generatorRegularizationQC(int count, double epsilon);

  /** Update force points, friction generators and mode coefficients.
   * @param solver Solver that owns this constraint.
   */
  void update(QPSolver & solver) override;
  void addToSolverImpl(QPSolver & solver) override;
  void removeFromSolverImpl(QPSolver & solver) override;

  const std::vector<mc_rbdyn::RollingContactDescription> & wheels() const noexcept;

  /** Update one wheel's force policy without removing its force variables.
   *
   * Fixed and rolling use the full cone, sliding freezes the nearest
   * polyhedral generator opposing established measured slip and retains that
   * direction below 50 mm/s to avoid a near-zero sign discontinuity, and
   * detached fixes every force component/multiplier to zero.
   */
  void mode(const std::string & wheel, mc_rbdyn::RollingContactMode mode);
  mc_rbdyn::RollingContactMode mode(const std::string & wheel) const;

  /** Selected tangent azimuth index [0, 3] for the sliding approximation. */
  int slidingGenerator(const std::string & wheel) const;

  /** World-frame tangent direction of the selected sliding generator. */
  Eigen::Vector3d slidingDirection(const std::string & wheel) const;

  /** Set the world-frame terrain normal used by all rolling force cones. */
  void terrainNormal(const Eigen::Vector3d & normal);

  /** Current normalized world-frame terrain normal. */
  const Eigen::Vector3d & terrainNormal() const noexcept;

  /** Absolute column in the complete Tasks decision vector. */
  int lambdaBegin(const std::string & wheel) const;
  int lambdaCount(const std::string & wheel) const;

  /** Positive generalized force map J_p^T G for one wheel (nrDof x 8). */
  const Eigen::MatrixXd & generalizedForceMatrix(const std::string & wheel) const;

  /** World forces at [lineStart, lineEnd] for eight wheel-local multipliers. */
  std::array<Eigen::Vector3d, 2> endpointForces(const std::string & wheel,
                                               const Eigen::Ref<const Eigen::VectorXd> & lambda) const;

  /** Solved endpoint forces for the TVM backend. */
  std::array<Eigen::Vector3d, 2> endpointForces(const std::string & wheel) const;

  /** World-frame force generators for endpoint 0 (lineStart) or 1 (lineEnd). */
  const Eigen::Matrix<double, 3, Eigen::Dynamic> & forceGenerators(const std::string & wheel,
                                                                   unsigned int endpoint) const;

  /** Total normal force, total tangential-force magnitude and circular-cone margin. */
  double normalForce(const std::string & wheel, const Eigen::Ref<const Eigen::VectorXd> & lambda) const;
  double tangentialForce(const std::string & wheel, const Eigen::Ref<const Eigen::VectorXd> & lambda) const;
  double frictionMargin(const std::string & wheel, const Eigen::Ref<const Eigen::VectorXd> & lambda) const;
  double normalForce(const std::string & wheel) const;
  double tangentialForce(const std::string & wheel) const;
  double frictionMargin(const std::string & wheel) const;

  /** Resultant SpaceVecAlg-order wrench about the carrier center. */
  Eigen::Matrix<double, 6, 1> resultantWrenchAtCarrier(
      const std::string & wheel,
      const Eigen::Ref<const Eigen::VectorXd> & lambda) const;
  Eigen::Matrix<double, 6, 1> resultantWrenchAtCarrier(const std::string & wheel) const;

  const mc_rbdyn::RollingContactGeometryResult & geometryResult(const std::string & wheel) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mc_solver
