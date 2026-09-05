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

#include <chrono>
#include <limits>
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

  void updateReference();
  void updateModes();
  void updateDiagnostics(bool solverSuccess);
  void synchronizeMeasuredState();
  void safeStop(const std::string & reason);
  std::vector<mc_rbdyn::RollingContactDescription> makeWheels() const;

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
  // First-order convergence time of a steering hinge towards its reference
  // heading, saturated by the URDF hinge velocity limit below.
  double steeringTimeConstant_ = 0.15;
  double maxSteeringRate_ = 8.0;
  double keyboardYawFeedbackGain_ = 0.5;
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
  bool keyboardStopLatched_ = false;
  bool keyboardCaptureWasRunning_ = false;
  bool contactFallback_ = false;
  bool lastSolverSuccess_ = false;
  double updateTimeMs_ = 0.0;
  double solveAndBuildTimeMs_ = 0.0;
  double totalTimeMs_ = 0.0;
  double maxRollingResidual_ = 0.0;
  double maxLateralResidual_ = 0.0;
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
  double dynamicsResidual_ = 0.0;
  double floatingBaseEffortNorm_ = 0.0;
  std::vector<double> appliedActivations_;
  std::vector<bool> hardPromotionPending_;
  std::vector<ExternalContactMeasurement> externalMeasurements_;
};

} // namespace mc_control
