/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_control/api.h>
#include <mc_control/mc_controller.h>

#include <mc_rbdyn/RollingContact.h>
#include <mc_solver/RollingContactConstraint.h>
#include <mc_solver/RollingContactDynamicsConstraint.h>
#include <mc_tasks/OrientationTask.h>
#include <mc_tasks/PositionTask.h>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mc_control
{

/** CPU rolling-contact sample for differential, four-steering, and Ranger Mini V3 robots. */
struct MC_CONTROL_DLLAPI MCRollingContactController : public MCController
{
  MCRollingContactController(mc_rbdyn::RobotModulePtr robot,
                             double dt,
                             const mc_rtc::Configuration & config);
  MCRollingContactController(mc_rbdyn::RobotModulePtr robot,
                             double dt,
                             const mc_rtc::Configuration & config,
                             Backend backend);
  ~MCRollingContactController() override;

  bool run() override;
  void reset(const ControllerResetData & data) override;

  /** Set the commanded planar chassis twist [vx, vy, omega] in the chassis frame.
   *
   * This is the single entry point used by the keyboard, the GUI and the
   * scripted scenarios. It only stores the command; the references are rebuilt
   * on the next run().
   *
   * @throws std::invalid_argument if the twist is not finite.
   */
  void setCommandedTwist(const Eigen::Vector3d & twist);
  const Eigen::Vector3d & commandedTwist() const noexcept { return commandedTwist_; }
  double maxLateralResidual() const noexcept { return maxLateralResidual_; }

private:
  struct KeyboardInput;

  struct ExternalContactMeasurement
  {
    double rollingSlip = 0.0;
    double lateralSlip = 0.0;
    double normalForce = 0.0;
    double tangentialForce = 0.0;
    double age = std::numeric_limits<double>::infinity();
    bool valid = false;
  };

  /** Rebuild every reference the QP reads, in three phases. */
  void updateReference();
  /** Phase 1: resolve this cycle's commanded planar twist [vx, vy, omega].
   *
   * Turns scenario_ into a twist - polling the keyboard when that is the active
   * scenario - republishes it through setCommandedTwist() on a four-steering
   * chassis, records it in the reference_* diagnostics, and zeroes it while a
   * contact fallback is latched.
   */
  Eigen::Vector3d resolveCommandedTwist();
  /** Phase 2: integrate the chassis pose targets and feed the two chassis tasks. */
  void updateChassisReference(const Eigen::Vector3d & twist);
  /** Phase 3: build the per-wheel drive and steering references and the posture target. */
  void updateWheelReferences(const Eigen::Vector3d & twist);
  /** True world heading of the chassis' +X axis, +psi for a chassis yawed by
   * psi in the world.
   *
   * robot().posW().rotation() is the inertial-to-body map E_0_b (E_0_b =
   * Rz(psi)^T for a chassis yawed by psi), so its transpose is the
   * body-to-inertial map and E_0_b^T * e_x is the true world direction the
   * chassis' +X axis points in. This is the single convention baseYawTarget_
   * uses everywhere except the closed-loop keyboard branch of
   * updateChassisReference(), which mirrors the measured heading directly
   * (see the long comment there) and negates this value at its point of use.
   *
   * May return NaN in the same atan2(0, 0) degenerate case every caller here
   * already guards against.
   */
  double measuredWorldYaw() const;
  /** Bound the accumulated heading target against the measured chassis heading.
   *
   * Applies to every branch that integrates the commanded yaw rate open loop;
   * closed-loop keyboard already copies the measured heading. See the
   * definition for why an unbounded heading target is a hard failure and not
   * merely a large error.
   */
  void saturateYawTargetAgainstMeasuredHeading();
  /** Push this cycle's terrain normal into both rolling-contact constraints.
   *
   * Called at the top of every run(), before the constraints' own update().
   * The value is currently the configured constant, normalised once in the
   * constructor, so this is a no-op in effect; it exists so the per-cycle
   * setter path is exercised and proven inert before anything estimates the
   * normal.
   */
  void updateTerrainNormal();
  /** Least-squares chassis planar twist [vx, vy, omega] of @p measured's wheels.
   *
   * Reads only @p measured's drive rates and steering angles - the MEASURED
   * state - and runs them forwards through mc_rbdyn::steeringRollingMatrix(),
   * the same rows the QP's rolling constraint is built from. Each wheel's two
   * rows and their right-hand side are scaled by sqrt(activation), so a
   * Detached wheel drops out of the estimate and a Sliding one fades.
   *
   * The solve is a complete-orthogonal-decomposition least squares, which
   * returns the minimum-norm solution for a rank-deficient block instead of
   * throwing, so a single attached wheel (or none at all) is not a failure.
   *
   * Deliberately NOT a function of the solved alphaD, of the QP's rolling-row
   * output, or of any reference: an estimate derived from the QP would confirm
   * the model instead of measuring against it, and every validation built on
   * this quantity would be vacuous. The caller passes realRobot(), which the
   * Encoder observer refreshes from the encoder packet before run().
   *
   * @param residual receives ||A v - b||, the weighted inconsistency of the
   * measured wheel state with a single rigid planar twist. Small while rolling;
   * it is the quantity that grows when the wheels skid.
   * @returns the twist, or a non-finite vector when the measured state is not
   * finite - consumers must check.
   */
  Eigen::Vector3d wheelOdometryTwist(const mc_rbdyn::Robot & measured, double & residual);
  /** Refresh odometryTwist_/odometryResidual_ from realRobot(). */
  void updateOdometry();
  /** Compare the tilt observer's roll and pitch against the reference attitude.
   *
   * Pure instrumentation: it writes only the attitudeMonitor* members, which
   * nothing but the log and the GUI reads. No estimate reaches the QP here -
   * the terrain normal is still the configured constant.
   *
   * The reference is realRobot().posW(), i.e. whatever the "FloatingBase" body
   * sensor reports. Under mc_mujoco that is the simulator's own qpos and the
   * comparison is against ground truth; under mc_rtc_ticker it is the control
   * robot's integrated pose, which is also what the synthesised IMU is built
   * from, and the comparison is a consistency check on the estimator's signal
   * path rather than independent validation. Neither is affected by this
   * function, which is why it is safe in both.
   *
   * Silently inert - attitudeMonitorValid_ stays false - when no observer
   * published "VelocityAidedTilt::Tilt::<robot>", so the RollingContact
   * pipelines that carry no tilt observer are unaffected.
   */
  void updateAttitudeMonitor();
  void updateModes();
  void updateDiagnostics(bool solverSuccess);
  void syncControlRobotFromSensors();
  void safeStop(const std::string & reason);
  std::vector<mc_rbdyn::RollingContactDescription> makeWheels() const;
  /** Index of the wheel named @p name, for the datastore lookups.
   *
   * @param query names the query in the exception message: "drive target"
   * produces "Unknown rolling-contact drive target wheel: <name>".
   * @throws std::invalid_argument when no wheel carries that name.
   */
  size_t wheelIndex(const std::string & name, const char * query) const;
  /** Publish a datastore call returning one per-wheel entry of @p values.
   *
   * @p values is held by reference: every caller passes one of this
   * controller's own diagnostic vectors, which stays index-aligned with
   * wheels_ and outlives the datastore.
   */
  void makeWheelValueCall(const std::string & key, const char * query, const std::vector<double> & values);

  // Constructor-time publication of this controller's external interfaces.
  // Split out of the constructor only for length; each is called exactly once,
  // in this order, and must keep that order: the datastore and log callbacks
  // capture members that the constraint and task construction between them
  // sets up.
  /** The operator's twist entry point, under "Rolling Contact"/"Command". */
  void registerCommandGUI();
  /** Every RollingContact::* datastore call. */
  void registerDatastoreCalls();
  /** Every RollingContact_* log entry, including the per-wheel ones. */
  void registerLogEntries();
  /** The read-only status labels, under "Rolling Contact". */
  void registerStatusGUI();
  /** updateAttitudeMonitor()'s output, under "Rolling Contact"/"Attitude". */
  void registerAttitudeGUI();

  std::vector<mc_rbdyn::RollingContactDescription> wheels_;
  std::vector<Eigen::Vector2d> wheelOffsets_;
  std::unique_ptr<KeyboardInput> keyboard_;
  std::unique_ptr<mc_solver::RollingContactDynamicsConstraint> dynamics_;
  std::unique_ptr<mc_solver::RollingContactConstraint> rolling_;
  std::shared_ptr<mc_tasks::OrientationTask> baseOrientationTask_;
  std::shared_ptr<mc_tasks::PositionTask> basePositionTask_;
  std::vector<mc_rbdyn::RollingContactModeManager> modeManagers_;
  std::string scenario_ = "hold";
  double linearSpeed_ = 0.0;
  double yawRate_ = 0.0;
  double steeringAngle_ = 0.0;
  // Commanded planar chassis twist [vx, vy, omega] in the chassis frame. Every
  // four-steering wheel reference is generated from this single command.
  Eigen::Vector3d commandedTwist_ = Eigen::Vector3d::Zero();
  // Per-axis weight [w_vx, w_vy, w_omega] on the planar twist error
  // (eq:planar-twist-weight), applied via dimWeight() to basePositionTask_'s
  // x/y rows and baseOrientationTask_'s z row respectively. This - not the
  // task-level weight() - is where unit normalisation belongs: w_vx and
  // w_vy carry (m/s)^-2, w_omega carries (rad/s)^-2, and a reader must not
  // assume the three are comparable in magnitude just because they are all
  // called "weight". Defaults to Ones(), which reproduces the previous
  // uniform/scalar behaviour exactly.
  Eigen::Vector3d twistWeight_ = Eigen::Vector3d::Ones();
  // First-order convergence time of a steering hinge towards its reference
  // heading, saturated by maxSteeringRate_. Both are configurable and
  // maxSteeringRate_ defaults to the steering joints' model velocity limit;
  // see the constructor for the rationale behind 0.15 s.
  double steeringTimeConstant_ = 0.15;
  double maxSteeringRate_ = 8.0;
  double keyboardYawFeedbackGain_ = 0.5;
  /** Largest heading error the accumulated yaw target may hold, in rad.
   *
   * Must stay strictly below pi: sva::rotationError() is singular there and
   * returns NaN a hundredth of a radian short of it, which fails the QP.
   */
  double maxYawTargetError_ = 0.5 * 3.14159265358979323846;
  double driveAcceleration_ = 20.0;
  double positionFeedbackGain_ = 5.0;
  double guiForwardCommand_ = 0.0;
  double guiLateralCommand_ = 0.0;
  double guiYawCommand_ = 0.0;
  double commandPeriod_ = 4.0;
  double recoverySpeed_ = 0.05;
  double recoveryResidual_ = 0.2;
  double recoveryTransverseResidual_ = 0.5;
  double rollingWeight_ = 1000.0;
  double recoveryRollingWeight_ = 1e6;
  double elapsed_ = 0.0;
  bool fourSteering_ = false;
  bool closedLoopFeedback_ = false;
  bool keyboardCaptureWasRunning_ = false;
  bool contactFallback_ = false;
  bool lastSolverSuccess_ = false;
  double updateTimeMs_ = 0.0;
  double solveAndBuildTimeMs_ = 0.0;
  double totalTimeMs_ = 0.0;
  double maxRollingResidual_ = 0.0;
  double maxLateralResidual_ = 0.0;
  /** ||sigma^lat||, the norm of the realised lateral slack of the current cycle.
   *
   * Zero while the lateral rows are hard. With softLateralRows it is the single
   * scalar that says how much skid the QP had to accept, and it is what
   * lateralSlackWeight is tuned against.
   */
  double lateralSlackNorm_ = 0.0;
  double minFrictionMargin_ = 0.0;
  double referenceLinearSpeed_ = 0.0;
  double referenceLateralSpeed_ = 0.0;
  double referenceYawRate_ = 0.0;
  Eigen::Vector3d terrainNormal_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d terrainTangentX_ = Eigen::Vector3d::UnitX();
  Eigen::Vector3d terrainTangentY_ = Eigen::Vector3d::UnitY();
  Eigen::Vector3d basePositionTarget_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d baseReferenceVelocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d baseTrackingVelocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d baseReferenceAngularVelocity_ = Eigen::Vector3d::Zero();
  double baseYawTarget_ = 0.0;
  /** Operator-commanded heading for the closed-loop keyboard's pure-yaw
   * feedback, in measuredWorldYaw()'s +psi convention.
   *
   * baseYawTarget_ cannot serve this role in the closed-loop keyboard branch
   * of updateChassisReference(): it is deliberately re-snapped to the
   * measured heading every cycle there (the sensor is authoritative in
   * closed loop, see the comment on that branch), so comparing it against
   * measuredWorldYaw() in the same cycle - which is what
   * updateWheelReferences() needs for its pure-yaw feedback - is always ~0.
   * This member is the accumulator that instead integrates the commanded yaw
   * rate every cycle the closed-loop keyboard branch runs, independent of
   * updateWheelReferences()'s pure-yaw gate on whether the resulting
   * correction gets applied, so it stays current through mixed commands
   * instead of jumping when pure yaw resumes. Seeded from measuredWorldYaw()
   * in reset(), same as baseYawTarget_; never used outside the closed-loop
   * keyboard scenario, which is fixed for the controller's whole lifetime
   * (see scenario_/closedLoopFeedback_).
   */
  double keyboardYawTarget_ = 0.0;
  double keyboardYawCorrection_ = 0.0;
  double keyboardYawError_ = 0.0;
  bool diagnosticsValid_ = false;
  std::string invalidReason_ = "not-run";
  std::vector<double> driveTargets_;
  std::vector<double> wheelReferenceRates_;
  std::vector<double> steeringTargets_;
  std::vector<double> steeringRateReferences_;
  std::vector<double> rollingResiduals_;
  std::vector<double> lateralResiduals_;
  std::vector<double> normalResiduals_;
  std::vector<double> accelerationResiduals_;
  std::vector<double> lateralAccelerationResiduals_;
  std::vector<double> normalAccelerationResiduals_;
  std::vector<double> hardPromotionResiduals_;
  std::vector<double> normalForces_;
  std::vector<double> tangentialForces_;
  std::vector<double> frictionMargins_;
  std::vector<double> driveTorques_;
  std::vector<double> driveTorqueMargins_;
  /** Last wheel-odometry twist [vx, vy, omega] in the chassis frame, and the
   * weighted inconsistency of the measured wheel state that produced it.
   *
   * Published on the datastore for the tilt observer and logged as
   * RollingContact_odometry_twist / _residual. Nothing in the QP reads them.
   */
  Eigen::Vector3d odometryTwist_ = Eigen::Vector3d::Zero();
  double odometryResidual_ = 0.0;
  /** updateAttitudeMonitor()'s output: estimated and reference attitude.
   *
   * The two roll/pitch pairs are in RADIANS and share mc_rbdyn::rpyFromMat's
   * convention; the GUI converts to degrees where it shows them. Both are
   * functions of a tilt vector alone - see updateAttitudeMonitor() - so no yaw
   * enters either side and the difference is exactly the estimator's error.
   */
  Eigen::Vector2d attitudeMonitorEstimatedRP_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d attitudeMonitorReferenceRP_ = Eigen::Vector2d::Zero();
  /** The same two attitudes as chassis +z in world, for the 3D arrows only.
   *
   * The estimated one is copied from the reference while no observer publishes
   * it, so that the two arrows sit on top of each other rather than one of them
   * pointing at the origin.
   */
  Eigen::Vector3d attitudeMonitorEstimatedNormal_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d attitudeMonitorReferenceNormal_ = Eigen::Vector3d::UnitZ();
  /** Angle between the estimated and the reference world normal, in radians. */
  double attitudeMonitorError_ = 0.0;
  /** False until an observer publishes a tilt; the monitor holds zeros then. */
  bool attitudeMonitorValid_ = false;
  double dynamicsResidual_ = 0.0;
  double floatingBaseEffortNorm_ = 0.0;
  std::vector<double> appliedActivations_;
  std::vector<bool> hardPromotionPending_;
  std::vector<ExternalContactMeasurement> externalMeasurements_;

  // Scratch buffers owned by the controller so that run() allocates nothing.
  // Each is sized once, in the constructor or by its in-class initialiser, and
  // only overwritten afterwards; none carries state across cycles, so reset()
  // deliberately leaves them alone.
  //
  // postureTargets_ holds one entry per drive joint, plus one per steering
  // joint on a four-steering chassis - exactly the key set updateReference()
  // and safeStop() used to rebuild from scratch every cycle. The keys never
  // change after construction, so only the stored values are rewritten.
  std::map<std::string, std::vector<double>> postureTargets_;
  std::vector<double> measuredDrivePositions_;
  /** Three-element scratch for the chassis tasks' refVel(), which copies. */
  Eigen::VectorXd taskRefVel_ = Eigen::VectorXd::Zero(3);
  /** nrDof scratch for the keyboard posture feed-forward. */
  Eigen::VectorXd keyboardRefVel_;
  /** nrDof scratch for the solved accelerations read back by updateDiagnostics(). */
  Eigen::VectorXd alphaDBuffer_;
  /** Scratch for wheelOdometryTwist(): the planar wheel descriptions, the
   * measured drive rates, and the weighted (2n)x3 block and right-hand side.
   *
   * steeringRollingMatrix() itself returns a freshly allocated (2n)x(3+n)
   * matrix, so this path is not allocation-free the way the rest of run() is.
   * Rebuilding those rows here to avoid it would duplicate the one piece of
   * kinematics the whole estimate is supposed to share with the constraint,
   * which is a worse trade than one small allocation per cycle.
   */
  std::vector<mc_rbdyn::PlanarWheel> odometryWheels_;
  Eigen::VectorXd odometryRates_;
  Eigen::MatrixXd odometryMatrix_;
  Eigen::VectorXd odometryRhs_;
};

} // namespace mc_control
