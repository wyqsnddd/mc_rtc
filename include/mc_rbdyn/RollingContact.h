/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_rbdyn/api.h>

#include <Eigen/Core>

#include <memory>
#include <string>
#include <vector>

namespace mc_rbdyn
{

struct Robot;

/** Contact modes supported by the rolling-contact model. */
enum class RollingContactMode
{
  Fixed,
  Rolling,
  Sliding,
  Detached
};

/** Return the stable configuration spelling for a rolling contact mode. */
MC_RBDYN_DLLAPI const char * to_string(RollingContactMode mode) noexcept;

/** Parse a rolling contact mode.
 *
 * @throws std::invalid_argument if @p mode is not one of fixed, rolling,
 * sliding or detached.
 */
MC_RBDYN_DLLAPI RollingContactMode rollingContactModeFromString(const std::string & mode);

/** Backend-neutral description of one conventional rigid wheel. */
struct MC_RBDYN_DLLAPI RollingContactDescription
{
  std::string name;
  std::string carrierFrame;
  std::string wheelBody;
  std::string driveJoint;
  std::string steeringJoint;
  double radius = 0.0;
  double width = 0.0;
  double friction = 0.7;
  double spinSign = 1.0;
  RollingContactMode mode = RollingContactMode::Rolling;
  double activation = 1.0;

  /** Validate scalar values and required names.
   *
   * Robot-specific name resolution is intentionally performed by the solver
   * adapter after the robot is loaded.
   */
  void validate() const;
};

/** Deterministic hysteresis and transition settings for one rolling contact. */
struct MC_RBDYN_DLLAPI RollingContactModeThresholds
{
  double slipEnter = 0.05;
  double slipExit = 0.02;
  double residualEnter = 0.05;
  double residualExit = 0.02;
  double normalForceEnter = 5.0;
  double normalForceExit = 1.0;
  double frictionMarginEnter = 0.1;
  double frictionMarginExit = 0.0;
  double torqueMarginEnter = 0.1;
  double torqueMarginExit = 0.0;
  double minimumDwell = 0.1;
  double transitionTime = 0.2;
  double filterTimeConstant = 0.02;

  void validate() const;
};

/** Measurements consumed by RollingContactModeManager for the current cycle. */
struct MC_RBDYN_DLLAPI RollingContactModeObservation
{
  bool valid = false;
  double slipSpeed = 0.0;
  double rollingResidual = 0.0;
  double normalForce = 0.0;
  double frictionMargin = 0.0;
  double torqueMargin = 0.0;
};

/** Current requested/estimated mode and transition diagnostics. */
struct MC_RBDYN_DLLAPI RollingContactModeState
{
  RollingContactMode requested = RollingContactMode::Rolling;
  RollingContactMode estimated = RollingContactMode::Detached;
  double activation = 0.0;
  double filteredSlip = 0.0;
  double dwellTime = 0.0;
  bool measurementValid = false;
  std::string invalidReason;
};

/** CPU-only rolling/sliding/detached mode estimator with deterministic time input.
 *
 * Mode selection remains outside the QP. Unsafe/invalid input and requested
 * detachment take effect immediately. Other changes must persist for
 * `minimumDwell`; activation is ramped over `transitionTime`.
 */
class MC_RBDYN_DLLAPI RollingContactModeManager
{
public:
  explicit RollingContactModeManager(RollingContactModeThresholds thresholds = {});

  void requestedMode(RollingContactMode mode) noexcept;
  RollingContactMode requestedMode() const noexcept;
  const RollingContactModeState & state() const noexcept;
  const RollingContactModeThresholds & thresholds() const noexcept;

  void reset(RollingContactMode requested = RollingContactMode::Rolling,
             RollingContactMode estimated = RollingContactMode::Detached,
             double activation = 0.0);

  const RollingContactModeState & update(const RollingContactModeObservation & observation, double dt);

private:
  RollingContactMode desiredMode(const RollingContactModeObservation & observation) const;
  void transitionTo(RollingContactMode mode);

  RollingContactModeThresholds thresholds_;
  RollingContactModeState state_;
  RollingContactMode pendingMode_ = RollingContactMode::Detached;
  double pendingTime_ = 0.0;
};

/** Inputs used by the per-wheel rolling geometry update.
 *
 * The carrier Jacobian follows the carrier-center convention: it must not
 * contain a second rim-speed contribution. `carrierNormalAcceleration` is
 * Jdot(q, alpha) * alpha at the carrier center. The terrain normal is assumed
 * constant during one update. `wheelAxleRate` is the inertial derivative of
 * `wheelAxle`. `steeringRate` is an optional additional rotation of t toward
 * l for callers that provide a static axle direction.
 */
struct MC_RBDYN_DLLAPI RollingContactKinematics
{
  Eigen::Vector3d carrierCenter = Eigen::Vector3d::Zero();
  Eigen::Vector3d wheelAxle = Eigen::Vector3d::UnitY();
  Eigen::Vector3d wheelAxleRate = Eigen::Vector3d::Zero();
  Eigen::Vector3d terrainNormal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d carrierNormalAcceleration = Eigen::Vector3d::Zero();
  Eigen::MatrixXd carrierJacobian;
  Eigen::RowVectorXd wheelSelector;
  /** Row selecting the steering (yaw) joint rate, empty for a non-steering wheel.
   *
   * When non-empty it must have the same size as wheelSelector. It addresses the
   * second rotating velocity of a steering wheel: see the rotational coupling
   * omega_{w/c} = theta_dot * l + delta_dot * n of the rolling-contact report.
   */
  Eigen::RowVectorXd steeringSelector;
  Eigen::VectorXd generalizedVelocity;
  double radius = 0.0;
  double width = 0.0;
  double spinSign = 1.0;
  double steeringRate = 0.0;
  double velocityGain = 0.0;
};

/** Result of one rolling geometry update.
 *
 * All vectors are expressed in the inertial frame. The rolling matrix is the
 * homogeneous block G from the rolling-contact report and `rhs` is the
 * stabilized acceleration-level right-hand side:
 *
 *   G * alphaD = -Gdot * alpha - Kp * G * alpha.
 */
struct MC_RBDYN_DLLAPI RollingContactGeometryResult
{
  Eigen::Vector3d rollingDirection = Eigen::Vector3d::UnitX();
  Eigen::Vector3d lateralDirection = Eigen::Vector3d::UnitY();
  Eigen::Vector3d normalDirection = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d rollingDirectionRate = Eigen::Vector3d::Zero();
  Eigen::Vector3d lateralDirectionRate = Eigen::Vector3d::Zero();
  Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d tangentProjector = Eigen::Matrix3d::Identity();
  Eigen::Vector3d carrierCenter = Eigen::Vector3d::Zero();
  Eigen::Vector3d contactPoint = Eigen::Vector3d::Zero();
  Eigen::Vector3d lineStart = Eigen::Vector3d::Zero();
  Eigen::Vector3d lineEnd = Eigen::Vector3d::Zero();
  Eigen::Vector3d carrierVelocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangentialCarrierVelocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d slipVelocity = Eigen::Vector3d::Zero();
  Eigen::MatrixXd rollingMatrix;
  Eigen::Vector3d velocityResidual = Eigen::Vector3d::Zero();
  Eigen::Vector3d accelerationBias = Eigen::Vector3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  double orthonormalError = 0.0;
  double rightHandedError = 0.0;
  /** Measured rolling (pitch) rate, wheelSelector * generalizedVelocity. */
  double measuredRollingRate = 0.0;
  /** Measured steering (yaw) rate, steeringSelector * generalizedVelocity; zero without a steering joint. */
  double measuredSteeringRate = 0.0;
};

/** Allocation-stable CPU rolling-geometry calculator for one wheel. */
class MC_RBDYN_DLLAPI RollingContactGeometry
{
public:
  explicit RollingContactGeometry(int nrDof);

  int nrDof() const noexcept { return nrDof_; }

  /** Update all geometry, rows and diagnostics.
   *
   * The dynamic matrices are sized by the constructor and are not resized by
   * this function.
   */
  const RollingContactGeometryResult & update(const RollingContactKinematics & input);

  const RollingContactGeometryResult & result() const noexcept { return result_; }

private:
  int nrDof_;
  RollingContactGeometryResult result_;
};

/** Name-resolved rolling geometry for one wheel on an mc_rbdyn Robot.
 *
 * This class resolves the carrier frame, wheel body and drive joint once. Its
 * update builds the carrier-center Jacobian, drive selector, generalized
 * velocity, axle direction/rate and normal acceleration directly from the
 * current RBDyn state before invoking RollingContactGeometry. Matrix sizes and
 * decision-vector offsets remain stable after construction.
 */
class MC_RBDYN_DLLAPI RollingContactRobotGeometry
{
public:
  RollingContactRobotGeometry(const Robot & robot, RollingContactDescription description);
  ~RollingContactRobotGeometry();

  RollingContactRobotGeometry(const RollingContactRobotGeometry &) = delete;
  RollingContactRobotGeometry & operator=(const RollingContactRobotGeometry &) = delete;
  RollingContactRobotGeometry(RollingContactRobotGeometry &&) noexcept;
  RollingContactRobotGeometry & operator=(RollingContactRobotGeometry &&) noexcept;

  /** Update from the robot's current kinematic and velocity state.
   *
   * The caller must have run forwardKinematics and forwardVelocity after the
   * latest state change. The terrain normal is inertial and constant for the
   * duration of this update.
   */
  const RollingContactGeometryResult & update(const Robot & robot,
                                              const Eigen::Vector3d & terrainNormal,
                                              double velocityGain = 0.0);

  const RollingContactDescription & description() const noexcept;
  const RollingContactKinematics & kinematics() const noexcept;
  const RollingContactGeometryResult & result() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** One wheel in an explicit planar chassis specialization. */
struct MC_RBDYN_DLLAPI PlanarWheel
{
  Eigen::Vector2d offset = Eigen::Vector2d::Zero();
  double steeringAngle = 0.0;
  double steeringRate = 0.0;
  double radius = 0.0;
  double spinSign = 1.0;

  void validate() const;
};

/** Explicit planar rolling rows and their known time-derivative bias. */
struct MC_RBDYN_DLLAPI PlanarRollingResult
{
  /** Columns are [vx, vy, omega, drive rates...]. */
  Eigen::MatrixXd matrix;
  /** Gdot * velocity for the current steering rates. */
  Eigen::VectorXd accelerationBias;
};

/** Build the three independent differential-drive rows.
 *
 * Columns are [vx, vy, omega, phiDotLeft, phiDotRight]. Rows are left
 * longitudinal, right longitudinal and the single independent lateral row.
 */
MC_RBDYN_DLLAPI PlanarRollingResult differentialDriveRollingMatrix(double trackWidth,
                                                                   double leftRadius,
                                                                   double rightRadius,
                                                                   double leftSpinSign = 1.0,
                                                                   double rightSpinSign = 1.0);

/** Build longitudinal/lateral rows for a four-steering or general N-wheel chassis.
 *
 * The result contains two rows per wheel. `planarVelocity` is [vx, vy, omega]
 * and is used only to compute the known direction-derivative bias.
 */
MC_RBDYN_DLLAPI PlanarRollingResult steeringRollingMatrix(const std::vector<PlanarWheel> & wheels,
                                                          const Eigen::Vector3d & planarVelocity);

/** Spatial wrench at the carrier center generated by a force at the contact point.
 *
 * The result uses SpaceVecAlg ordering: angular moment followed by linear force.
 */
MC_RBDYN_DLLAPI Eigen::Matrix<double, 6, 1> contactWrenchAtCarrier(const Eigen::Vector3d & carrierCenter,
                                                                 const Eigen::Vector3d & contactPoint,
                                                                 const Eigen::Vector3d & force);

} // namespace mc_rbdyn
