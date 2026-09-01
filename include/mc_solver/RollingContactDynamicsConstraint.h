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
 */
class MC_SOLVER_DLLAPI RollingContactDynamicsConstraint : public DynamicsConstraint
{
public:
  RollingContactDynamicsConstraint(const mc_rbdyn::Robots & robots,
                                   unsigned int robotIndex,
                                   double timeStep,
                                   std::vector<mc_rbdyn::RollingContactDescription> wheels,
                                   bool infTorque = false);
  ~RollingContactDynamicsConstraint() override;

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
