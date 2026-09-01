/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "mc_rolling_contact_controller.h"

#include <mc_rtc/gui/Label.h>
#include <mc_rtc/logging.h>
#include <mc_solver/TasksQPSolver.h>

#include <mc_tvm/Robot.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace mc_control
{

namespace
{

mc_solver::RollingContactLongitudinal longitudinalMode(const mc_rtc::Configuration & config)
{
  const std::string value = config("longitudinal", std::string{"hard"});
  if(value == "hard") { return mc_solver::RollingContactLongitudinal::Hard; }
  if(value == "soft") { return mc_solver::RollingContactLongitudinal::Soft; }
  mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact longitudinal must be hard or soft, got {}",
                                                      value);
}

} // namespace

MCRollingContactController::MCRollingContactController(mc_rbdyn::RobotModulePtr robotModule,
                                                       double dt,
                                                       const mc_rtc::Configuration & config)
: MCRollingContactController(robotModule, dt, config, Backend::Tasks)
{
}

MCRollingContactController::MCRollingContactController(mc_rbdyn::RobotModulePtr robotModule,
                                                       double dt,
                                                       const mc_rtc::Configuration & config,
                                                       Backend backend)
: MCController(robotModule, dt, config, backend)
{
  if(robot().name() != "rolling_diff" && robot().name() != "rolling_4s")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact controller requires rolling_diff or rolling_4s, got {}", robot().name());
  }
  fourSteering_ = robot().name() == "rolling_4s";
  const auto settings = config.has("RollingContact") ? config("RollingContact") : mc_rtc::Configuration{};
  scenario_ = settings("scenario", std::string{"hold"});
  closedLoopFeedback_ = settings("closedLoopFeedback", false);
  linearSpeed_ = settings("linearSpeed", 0.2);
  yawRate_ = settings("yawRate", 0.35);
  steeringAngle_ = settings("steeringAngle", 0.3);
  commandPeriod_ = settings("commandPeriod", 4.0);
  recoverySpeed_ = settings("recoverySpeed", 0.05);
  recoveryResidual_ = settings("recoveryResidual", 0.2);
  recoveryTransverseResidual_ = settings("recoveryTransverseResidual", 0.5);
  rollingWeight_ = settings("rollingWeight", 1000.0);
  recoveryRollingWeight_ = settings("recoveryRollingWeight", 1e6);
  if(!std::isfinite(commandPeriod_) || commandPeriod_ <= 0.0 || !std::isfinite(linearSpeed_)
     || !std::isfinite(yawRate_) || !std::isfinite(steeringAngle_) || !std::isfinite(recoverySpeed_)
     || recoverySpeed_ < 0.0 || !std::isfinite(recoveryResidual_) || recoveryResidual_ < 0.0
     || !std::isfinite(recoveryTransverseResidual_) || recoveryTransverseResidual_ < 0.0
     || !std::isfinite(rollingWeight_) || rollingWeight_ <= 0.0 || !std::isfinite(recoveryRollingWeight_)
     || recoveryRollingWeight_ < rollingWeight_)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact commandPeriod and rolling weights must be positive, recovery thresholds non-negative, "
        "recoveryRollingWeight must be at least rollingWeight, and command scalars must be finite");
  }
  const std::vector<std::string> differentialScenarios = {"hold",       "reverse",      "forward",
                                                           "turn_left",  "turn_right",   "circle_left",
                                                           "circle_right", "sinusoid",   "unequal_radii",
                                                           "infeasible_soft", "mode_cycle"};
  const std::vector<std::string> steeringScenarios = {"hold",       "forward",       "reverse",
                                                       "crab",       "ackermann_left", "ackermann_right",
                                                       "pure_yaw",   "steering_rate", "incompatible",
                                                       "mode_cycle"};
  const auto & supported = fourSteering_ ? steeringScenarios : differentialScenarios;
  if(std::find(supported.begin(), supported.end(), scenario_) == supported.end())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario '{}' for {}", scenario_,
                                                        robot().name());
  }

  wheels_ = makeWheels();
  mc_solver::RollingContactConstraintOptions options;
  options.terrainNormal = settings("terrainNormal", Eigen::Vector3d{0.0, 0.0, 1.0});
  if(!options.terrainNormal.allFinite() || options.terrainNormal.norm() < 1e-8)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact terrainNormal must be finite and nonzero");
  }
  terrainNormal_ = options.terrainNormal.normalized();
  terrainTangentX_ = Eigen::Vector3d::UnitX() - terrainNormal_ * terrainNormal_.x();
  if(terrainTangentX_.norm() < 1e-8)
  {
    terrainTangentX_ = Eigen::Vector3d::UnitY() - terrainNormal_ * terrainNormal_.y();
  }
  terrainTangentX_.normalize();
  terrainTangentY_ = terrainNormal_.cross(terrainTangentX_).normalized();
  options.longitudinal = longitudinalMode(settings);
  options.velocityGain = settings("velocityGain", 20.0);
  options.rollingWeight = rollingWeight_;
  options.constrainNormal = true;
  options.differentialPlanar = !fourSteering_;
  options.steeringPlanar = fourSteering_;
  if(fourSteering_) { options.steeringPlanarWheels = {"front_left", "rear_left"}; }
  dynamics_ = std::make_unique<mc_solver::RollingContactDynamicsConstraint>(robots(), 0, dt, wheels_);
  dynamics_->terrainNormal(options.terrainNormal);
  rolling_ = std::make_unique<mc_solver::RollingContactConstraint>(robots(), 0, wheels_, options);

  mc_rbdyn::RollingContactModeThresholds thresholds;
  thresholds.slipEnter = settings("slipEnter", 0.05);
  thresholds.slipExit = settings("slipExit", 0.02);
  thresholds.residualEnter = settings("residualEnter", 0.05);
  thresholds.residualExit = settings("residualExit", 0.02);
  thresholds.normalForceEnter = settings("normalForceEnter", 5.0);
  thresholds.normalForceExit = settings("normalForceExit", 1.0);
  thresholds.frictionMarginEnter = settings("frictionMarginEnter", 1e-3);
  thresholds.frictionMarginExit = settings("frictionMarginExit", -1e-5);
  thresholds.torqueMarginEnter = settings("torqueMarginEnter", 0.1);
  thresholds.torqueMarginExit = settings("torqueMarginExit", 0.0);
  thresholds.minimumDwell = settings("minimumDwell", 0.1);
  thresholds.transitionTime = settings("transitionTime", 0.2);
  thresholds.filterTimeConstant = settings("filterTimeConstant", 0.02);
  thresholds.validate();
  modeManagers_.reserve(wheels_.size());
  for(const auto & wheel : wheels_)
  {
    modeManagers_.emplace_back(thresholds);
    modeManagers_.back().reset(wheel.mode, wheel.mode,
                               wheel.mode == mc_rbdyn::RollingContactMode::Detached ? 0.0 : 1.0);
  }

  solver().addConstraintSet(*dynamics_);
  solver().addConstraintSet(*rolling_);
  postureTask->stiffness(settings("postureStiffness", 5.0));
  postureTask->weight(settings("postureWeight", 100.0));
  if(fourSteering_)
  {
    const double steeringStiffness = settings("steeringStiffness", 1000.0);
    const double steeringWeight = settings("steeringWeight", 10.0);
    if(!std::isfinite(steeringStiffness) || steeringStiffness <= 0.0 || !std::isfinite(steeringWeight)
       || steeringWeight <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact steeringStiffness and steeringWeight must be finite and positive");
    }
    std::vector<tasks::qp::JointStiffness> steeringGains;
    std::map<std::string, double> steeringWeights;
    steeringGains.reserve(wheels_.size());
    for(const auto & wheel : wheels_)
    {
      steeringGains.emplace_back(wheel.steeringJoint, steeringStiffness);
      steeringWeights.emplace(wheel.steeringJoint, steeringWeight);
    }
    postureTask->jointStiffness(solver(), steeringGains);
    postureTask->jointWeights(steeringWeights);
  }
  solver().addTask(postureTask);
  baseOrientationTask_ = std::make_shared<mc_tasks::OrientationTask>(
      "chassis", robots(), 0, settings("baseOrientationStiffness", 10.0),
      settings("baseOrientationWeight", 500.0));
  baseOrientationTask_->dimWeight(Eigen::Vector3d::Ones());
  solver().addTask(baseOrientationTask_);
  basePositionTask_ = std::make_shared<mc_tasks::PositionTask>(
      "chassis", robots(), 0, settings("basePositionStiffness", 5.0), settings("basePositionWeight", 500.0));
  solver().addTask(basePositionTask_);
  solver().setContacts({});
  driveTargets_.resize(wheels_.size(), 0.0);
  rollingResiduals_.resize(wheels_.size(), 0.0);
  lateralResiduals_.resize(wheels_.size(), 0.0);
  normalResiduals_.resize(wheels_.size(), 0.0);
  accelerationResiduals_.resize(wheels_.size(), 0.0);
  lateralAccelerationResiduals_.resize(wheels_.size(), 0.0);
  normalAccelerationResiduals_.resize(wheels_.size(), 0.0);
  hardPromotionResiduals_.resize(wheels_.size(), 0.0);
  normalForces_.resize(wheels_.size(), 0.0);
  tangentialForces_.resize(wheels_.size(), 0.0);
  frictionMargins_.resize(wheels_.size(), 0.0);
  driveTorques_.resize(wheels_.size(), 0.0);
  driveTorqueMargins_.resize(wheels_.size(), 0.0);
  appliedActivations_.resize(wheels_.size(), 1.0);
  hardPromotionPending_.resize(wheels_.size(), false);
  externalMeasurements_.resize(wheels_.size());

  datastore().make_call(
      "RollingContact::SetMeasuredContact",
      [this](const std::string & name,
             double rollingSlip,
             double lateralSlip,
             double normalForce,
             double tangentialForce,
             bool valid)
      {
        const auto wheel = std::find_if(wheels_.begin(), wheels_.end(),
                                        [&name](const auto & candidate) { return candidate.name == name; });
        if(wheel == wheels_.end())
        {
          throw std::invalid_argument("Unknown rolling-contact measurement wheel: " + name);
        }
        const size_t index = static_cast<size_t>(std::distance(wheels_.begin(), wheel));
        auto & measurement = externalMeasurements_[index];
        measurement.rollingSlip = rollingSlip;
        measurement.lateralSlip = lateralSlip;
        measurement.normalForce = normalForce;
        measurement.tangentialForce = tangentialForce;
        measurement.age = 0.0;
        measurement.valid = valid && std::isfinite(rollingSlip) && std::isfinite(lateralSlip)
                            && std::isfinite(normalForce) && normalForce >= 0.0 && std::isfinite(tangentialForce)
                            && tangentialForce >= 0.0;
      });
  datastore().make_call("RollingContact::GetEstimatedMode",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact mode wheel: " + name);
                          }
                          const size_t index = static_cast<size_t>(std::distance(wheels_.begin(), wheel));
                          return std::string{mc_rbdyn::to_string(modeManagers_[index].state().estimated)};
                        });
  datastore().make_call("RollingContact::GetActivation",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact activation wheel: " + name);
                          }
                          const size_t index = static_cast<size_t>(std::distance(wheels_.begin(), wheel));
                          return modeManagers_[index].state().activation;
                        });
  datastore().make_call("RollingContact::GetSolverActivation",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact solver activation wheel: " + name);
                          }
                          return appliedActivations_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetAccelerationResidual",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact acceleration residual wheel: " + name);
                          }
                          return accelerationResiduals_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetHardPromotionResidual",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact hard promotion residual wheel: "
                                                        + name);
                          }
                          return hardPromotionResiduals_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetLateralAccelerationResidual",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact lateral acceleration wheel: " + name);
                          }
                          return lateralAccelerationResiduals_[static_cast<size_t>(
                              std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetNormalAccelerationResidual",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact normal acceleration wheel: " + name);
                          }
                          return normalAccelerationResiduals_[static_cast<size_t>(
                              std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetBasePositionTarget", [this]() { return basePositionTarget_; });
  datastore().make_call("RollingContact::GetBaseYawTarget", [this]() { return baseYawTarget_; });
  datastore().make_call("RollingContact::GetContactFallback", [this]() { return contactFallback_; });
  datastore().make_call("RollingContact::GetRollingWeight", [this]() { return rolling_->rollingWeight(); });
  datastore().make_call("RollingContact::GetDynamicsResidual", [this]() { return dynamicsResidual_; });
  datastore().make_call("RollingContact::GetFloatingBaseEffortNorm", [this]() { return floatingBaseEffortNorm_; });
  datastore().make_call(
      "RollingContact::GetQPNormalForce",
      [this](const std::string & name)
      {
        const auto wheel = std::find_if(wheels_.begin(), wheels_.end(), [&name](const auto & candidate)
                                        { return candidate.name == name; });
        if(wheel == wheels_.end()) { throw std::invalid_argument("Unknown rolling-contact QP-force wheel: " + name); }
        return normalForces_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
      });
  datastore().make_call(
      "RollingContact::GetQPTangentialForce",
      [this](const std::string & name)
      {
        const auto wheel = std::find_if(wheels_.begin(), wheels_.end(), [&name](const auto & candidate)
                                        { return candidate.name == name; });
        if(wheel == wheels_.end()) { throw std::invalid_argument("Unknown rolling-contact QP-force wheel: " + name); }
        return tangentialForces_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
      });
  datastore().make_call("RollingContact::GetDriveTarget",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact drive target wheel: " + name);
                          }
                          return driveTargets_[static_cast<size_t>(std::distance(wheels_.begin(), wheel))];
                        });
  datastore().make_call("RollingContact::GetDriveAcceleration",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact drive acceleration wheel: " + name);
                          }
                          const auto joint = robot().jointIndexByName(wheel->driveJoint);
                          return robot().mbc().alphaD[joint][0];
                        });
  datastore().make_call("RollingContact::GetDrivePosition",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact drive position wheel: " + name);
                          }
                          const auto joint = robot().jointIndexByName(wheel->driveJoint);
                          return robot().mbc().q[joint][0];
                        });
  datastore().make_call("RollingContact::GetDriveVelocity",
                        [this](const std::string & name)
                        {
                          const auto wheel = std::find_if(
                              wheels_.begin(), wheels_.end(),
                              [&name](const auto & candidate) { return candidate.name == name; });
                          if(wheel == wheels_.end())
                          {
                            throw std::invalid_argument("Unknown rolling-contact drive velocity wheel: " + name);
                          }
                          const auto joint = robot().jointIndexByName(wheel->driveJoint);
                          return robot().mbc().alpha[joint][0];
                        });
  datastore().make_call("RollingContact::GetBackend", [this]()
                        { return solver().backend() == Backend::Tasks ? std::string{"Tasks"} : std::string{"TVM"}; });
  datastore().make_call("RollingContact::GetPostureEvalNorm", [this]() { return postureTask->eval().norm(); });
  datastore().make_call("RollingContact::GetPositionEvalNorm", [this]() { return basePositionTask_->eval().norm(); });
  datastore().make_call("RollingContact::GetOrientationEvalNorm",
                        [this]() { return baseOrientationTask_->eval().norm(); });
  datastore().make_call("RollingContact::GetHardRhsNorm", [this]() { return rolling_->hardRhs().norm(); });
  datastore().make_call("RollingContact::GetSlidingGenerator",
                        [this](const std::string & name) { return dynamics_->slidingGenerator(name); });

  logger().addLogEntry("RollingContact_scenario", [this]() { return scenario_; });
  logger().addLogEntry("RollingContact_backend", [this]()
                       { return solver().backend() == Backend::Tasks ? std::string{"Tasks"} : std::string{"TVM"}; });
  logger().addLogEntry("RollingContact_solver_success", [this]() { return lastSolverSuccess_; });
  logger().addLogEntry("RollingContact_closed_loop_feedback", [this]() { return closedLoopFeedback_; });
  logger().addLogEntry("RollingContact_contact_fallback", [this]() { return contactFallback_; });
  logger().addLogEntry("RollingContact_rolling_weight", [this]() { return rolling_->rollingWeight(); });
  logger().addLogEntry("RollingContact_dynamics_residual", [this]() { return dynamicsResidual_; });
  logger().addLogEntry("RollingContact_floating_base_effort_norm", [this]() { return floatingBaseEffortNorm_; });
  logger().addLogEntry("RollingContact_update_ms", [this]() { return updateTimeMs_; });
  logger().addLogEntry("RollingContact_solve_and_build_ms", [this]() { return solveAndBuildTimeMs_; });
  logger().addLogEntry("RollingContact_total_ms", [this]() { return totalTimeMs_; });
  logger().addLogEntry("RollingContact_max_longitudinal_residual", [this]() { return maxRollingResidual_; });
  logger().addLogEntry("RollingContact_max_lateral_residual", [this]() { return maxLateralResidual_; });
  logger().addLogEntry("RollingContact_min_friction_margin", [this]() { return minFrictionMargin_; });
  logger().addLogEntry("RollingContact_diagnostics_valid", [this]() { return diagnosticsValid_; });
  logger().addLogEntry("RollingContact_invalid_reason", [this]() { return invalidReason_; });
  logger().addLogEntry("RollingContact_reference_linear_speed", [this]() { return referenceLinearSpeed_; });
  logger().addLogEntry("RollingContact_reference_yaw_rate", [this]() { return referenceYawRate_; });
  logger().addLogEntry("RollingContact_base_pose", [this]() { return robot().posW(); });
  logger().addLogEntry("RollingContact_base_twist", [this]() { return robot().velW(); });
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto & wheel = wheels_[i];
    const auto joint = robot().jointIndexByName(wheel.driveJoint);
    logger().addLogEntry("RollingContact_" + wheel.name + "_position",
                         [this, joint]() { return robot().mbc().q[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rate",
                         [this, joint]() { return robot().mbc().alpha[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_target", [this, i]() { return driveTargets_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_requested_mode", [this, i]()
                         { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().requested)}; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_mode", [this, i]()
                         { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().estimated)}; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_activation",
                         [this, i]() { return modeManagers_[i].state().activation; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_solver_activation",
                         [this, i]() { return appliedActivations_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_filtered_slip",
                         [this, i]() { return modeManagers_[i].state().filteredSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_dwell_time",
                         [this, i]() { return modeManagers_[i].state().dwellTime; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measurement_valid",
                         [this, i]() { return modeManagers_[i].state().measurementValid; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_invalid_reason",
                         [this, i]() { return modeManagers_[i].state().invalidReason; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rolling_residual",
                         [this, i]() { return rollingResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_lateral_residual",
                         [this, i]() { return lateralResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_residual",
                         [this, i]() { return normalResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_acceleration_residual",
                         [this, i]() { return accelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_lateral_acceleration_residual",
                         [this, i]() { return lateralAccelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_acceleration_residual",
                         [this, i]() { return normalAccelerationResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_hard_promotion_residual",
                         [this, i]() { return hardPromotionResiduals_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_normal_force",
                         [this, i]() { return normalForces_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_tangential_force",
                         [this, i]() { return tangentialForces_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_friction_margin",
                         [this, i]() { return frictionMargins_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_drive_torque",
                         [this, i]() { return driveTorques_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_drive_torque_margin",
                         [this, i]() { return driveTorqueMargins_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_sliding_generator",
                         [this, i]() { return dynamics_->slidingGenerator(wheels_[i].name); });
    logger().addLogEntry("RollingContact_" + wheel.name + "_simulator_measurement_valid",
                         [this, i]()
                         {
                           return externalMeasurements_[i].valid
                                  && externalMeasurements_[i].age <= 2.0 * solver().dt();
                         });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_rolling_slip",
                         [this, i]() { return externalMeasurements_[i].rollingSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_lateral_slip",
                         [this, i]() { return externalMeasurements_[i].lateralSlip; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_normal_force",
                         [this, i]() { return externalMeasurements_[i].normalForce; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_tangential_force",
                         [this, i]() { return externalMeasurements_[i].tangentialForce; });
    if(!wheel.steeringJoint.empty())
    {
      const auto steeringJoint = robot().jointIndexByName(wheel.steeringJoint);
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_position",
                           [this, steeringJoint]() { return robot().mbc().q[steeringJoint][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_rate",
                           [this, steeringJoint]() { return robot().mbc().alpha[steeringJoint][0]; });
    }
  }
  gui()->addElement({"Rolling Contact"},
                    mc_rtc::gui::Label("Scenario", [this]() { return scenario_; }),
                    mc_rtc::gui::Label("Backend", [this]()
                                       { return solver().backend() == Backend::Tasks ? "Tasks" : "TVM"; }),
                    mc_rtc::gui::Label("Diagnostics valid", [this]() { return diagnosticsValid_; }),
                    mc_rtc::gui::Label("Invalid reason", [this]() { return invalidReason_; }));
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    gui()->addElement(
        {"Rolling Contact", wheels_[i].name},
        mc_rtc::gui::Label("Requested mode", [this, i]()
                           { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().requested)}; }),
        mc_rtc::gui::Label("Estimated mode", [this, i]()
                           { return std::string{mc_rbdyn::to_string(modeManagers_[i].state().estimated)}; }),
        mc_rtc::gui::Label("Activation", [this, i]() { return modeManagers_[i].state().activation; }),
        mc_rtc::gui::Label("Solver activation", [this, i]() { return appliedActivations_[i]; }),
        mc_rtc::gui::Label("Slip", [this, i]() { return modeManagers_[i].state().filteredSlip; }),
        mc_rtc::gui::Label("Normal force", [this, i]() { return normalForces_[i]; }),
        mc_rtc::gui::Label("Friction margin", [this, i]() { return frictionMargins_[i]; }),
        mc_rtc::gui::Label("Drive torque margin", [this, i]() { return driveTorqueMargins_[i]; }));
  }
  mc_rtc::log::success("RollingContact CPU controller initialized for {} ({})", robot().name(), scenario_);
}

MCRollingContactController::~MCRollingContactController()
{
  // TVM keeps graph nodes that refer to tasks and constraint functions. Remove
  // every derived-class object while it is still alive, before C++ destroys
  // these members and eventually the base-class solver.
  solver().removeTask(basePositionTask_);
  solver().removeTask(baseOrientationTask_);
  solver().removeTask(postureTask);
  solver().removeConstraintSet(*rolling_);
  solver().removeConstraintSet(*dynamics_);
}

std::vector<mc_rbdyn::RollingContactDescription> MCRollingContactController::makeWheels() const
{
  std::vector<mc_rbdyn::RollingContactDescription> wheels;
  const std::vector<std::string> names = fourSteering_
                                             ? std::vector<std::string>{"front_left", "front_right", "rear_left",
                                                                        "rear_right"}
                                             : std::vector<std::string>{"left", "right"};
  for(const auto & name : names)
  {
    mc_rbdyn::RollingContactDescription wheel;
    wheel.name = name;
    wheel.carrierFrame = name + "_carrier";
    wheel.wheelBody = name + "_wheel";
    wheel.driveJoint = name + "_drive";
    if(fourSteering_) { wheel.steeringJoint = name + "_steer"; }
    wheel.radius = 0.2;
    if(!fourSteering_ && scenario_ == "unequal_radii") { wheel.radius = name == "left" ? 0.18 : 0.22; }
    wheel.width = 0.08;
    wheel.friction = 0.8;
    wheel.validate();
    wheels.push_back(std::move(wheel));
  }
  return wheels;
}

void MCRollingContactController::reset(const ControllerResetData & data)
{
  MCController::reset(data);
  elapsed_ = 0.0;
  lastSolverSuccess_ = false;
  contactFallback_ = false;
  updateTimeMs_ = 0.0;
  solveAndBuildTimeMs_ = 0.0;
  totalTimeMs_ = 0.0;
  maxRollingResidual_ = 0.0;
  maxLateralResidual_ = 0.0;
  minFrictionMargin_ = 0.0;
  referenceLinearSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  dynamicsResidual_ = 0.0;
  floatingBaseEffortNorm_ = 0.0;
  basePositionTarget_ = robot().frame("chassis").position().translation();
  baseYawTarget_ = 0.0;
  diagnosticsValid_ = false;
  invalidReason_ = "not-run";
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto joint = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[joint][0];
    modeManagers_[i].reset(wheels_[i].mode, wheels_[i].mode,
                           wheels_[i].mode == mc_rbdyn::RollingContactMode::Detached ? 0.0 : 1.0);
    appliedActivations_[i] = modeManagers_[i].state().activation;
    hardPromotionPending_[i] = wheels_[i].mode == mc_rbdyn::RollingContactMode::Detached;
    rolling_->mode(wheels_[i].name, wheels_[i].mode, appliedActivations_[i]);
    dynamics_->mode(wheels_[i].name, wheels_[i].mode);
  }
  std::fill(rollingResiduals_.begin(), rollingResiduals_.end(), 0.0);
  std::fill(lateralResiduals_.begin(), lateralResiduals_.end(), 0.0);
  std::fill(normalResiduals_.begin(), normalResiduals_.end(), 0.0);
  std::fill(accelerationResiduals_.begin(), accelerationResiduals_.end(), 0.0);
  std::fill(lateralAccelerationResiduals_.begin(), lateralAccelerationResiduals_.end(), 0.0);
  std::fill(normalAccelerationResiduals_.begin(), normalAccelerationResiduals_.end(), 0.0);
  std::fill(hardPromotionResiduals_.begin(), hardPromotionResiduals_.end(), 0.0);
  std::fill(normalForces_.begin(), normalForces_.end(), 0.0);
  std::fill(tangentialForces_.begin(), tangentialForces_.end(), 0.0);
  std::fill(frictionMargins_.begin(), frictionMargins_.end(), 0.0);
  std::fill(driveTorques_.begin(), driveTorques_.end(), 0.0);
  std::fill(driveTorqueMargins_.begin(), driveTorqueMargins_.end(), 0.0);
  for(auto & measurement : externalMeasurements_) { measurement = {}; }
  postureTask->reset();
  baseOrientationTask_->reset();
  basePositionTask_->reset();
}

void MCRollingContactController::updateReference()
{
  const double phase = 2.0 * 3.14159265358979323846 * elapsed_ / commandPeriod_;
  double v = linearSpeed_;
  double yaw = yawRate_;
  if(scenario_ == "hold") { v = yaw = 0.0; }
  else if(scenario_ == "reverse") { v = -std::abs(linearSpeed_); yaw = 0.0; }
  else if(scenario_ == "forward" || scenario_ == "unequal_radii" || scenario_ == "infeasible_soft")
  {
    yaw = 0.0;
  }
  else if(scenario_ == "mode_cycle") { v = yaw = 0.0; }
  else if(scenario_ == "turn_left") { v = 0.0; yaw = std::abs(yawRate_); }
  else if(scenario_ == "turn_right") { v = 0.0; yaw = -std::abs(yawRate_); }
  else if(scenario_ == "circle_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "circle_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "sinusoid") { yaw *= std::sin(phase); }
  else if(scenario_ == "crab" || scenario_ == "steering_rate" || scenario_ == "incompatible") { yaw = 0.0; }
  else if(scenario_ == "ackermann_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "ackermann_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "pure_yaw") { v = 0.0; }
  else if(scenario_ != "hold")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario: {}", scenario_);
  }
  referenceLinearSpeed_ = v;
  referenceYawRate_ = yaw;
  if(contactFallback_)
  {
    v = 0.0;
    yaw = 0.0;
    referenceLinearSpeed_ = 0.0;
    referenceYawRate_ = 0.0;
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      driveTargets_[i] = robot().mbc().q[drive][0];
    }
  }

  baseYawTarget_ = std::remainder(baseYawTarget_ + yaw * solver().dt(), 2.0 * 3.14159265358979323846);
  const Eigen::Vector3d heading = std::cos(baseYawTarget_) * terrainTangentX_
                                  + std::sin(baseYawTarget_) * terrainTangentY_;
  const Eigen::Vector3d side = -std::sin(baseYawTarget_) * terrainTangentX_
                               + std::cos(baseYawTarget_) * terrainTangentY_;
  double velocityAngle = 0.0;
  if(scenario_ == "crab") { velocityAngle = steeringAngle_; }
  else if(scenario_ == "steering_rate") { velocityAngle = steeringAngle_ * std::sin(phase); }
  basePositionTarget_ += v * solver().dt() * (std::cos(velocityAngle) * heading + std::sin(velocityAngle) * side);
  basePositionTask_->position(basePositionTarget_);
  Eigen::Matrix3d baseRotation;
  baseRotation.col(0) = heading;
  baseRotation.col(1) = side;
  baseRotation.col(2) = terrainNormal_;
  baseOrientationTask_->orientation(baseRotation.transpose());

  std::map<std::string, std::vector<double>> targets;
  if(!fourSteering_)
  {
    constexpr double halfTrack = 0.3;
    std::array<double, 2> rates = {(v - halfTrack * yaw) / wheels_[0].radius,
                                   (v + halfTrack * yaw) / wheels_[1].radius};
    if(scenario_ == "infeasible_soft") { rates[0] += 4.0; }
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      driveTargets_[i] += rates[i] * solver().dt();
      targets[wheels_[i].driveJoint] = {driveTargets_[i]};
    }
  }
  else
  {
    const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3},
                                                    Eigen::Vector2d{0.45, -0.3},
                                                    Eigen::Vector2d{-0.45, 0.3},
                                                    Eigen::Vector2d{-0.45, -0.3}};
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      double steering = 0.0;
      double rate = v / wheels_[i].radius;
      if(scenario_ == "crab") { steering = steeringAngle_; }
      else if(scenario_ == "steering_rate") { steering = steeringAngle_ * std::sin(phase); }
      else if(scenario_ == "ackermann_left" || scenario_ == "ackermann_right" || scenario_ == "pure_yaw")
      {
        const double pointX = v - yaw * offsets[i].y();
        const double pointY = yaw * offsets[i].x();
        steering = std::atan2(pointY, pointX);
        rate = std::hypot(pointX, pointY) / wheels_[i].radius;
      }
      else if(scenario_ == "incompatible")
      {
        steering = i == 0 ? steeringAngle_ : 0.0;
        if(i == 0) { rate += 4.0; }
      }
      if(contactFallback_)
      {
        const auto steeringJoint = robot().jointIndexByName(wheels_[i].steeringJoint);
        steering = robot().mbc().q[steeringJoint][0];
      }
      driveTargets_[i] += rate * solver().dt();
      targets[wheels_[i].driveJoint] = {driveTargets_[i]};
      targets[wheels_[i].steeringJoint] = {steering};
    }
  }
  postureTask->target(targets);
}

void MCRollingContactController::updateModes()
{
  const bool wasContactFallback = contactFallback_;
  contactFallback_ = false;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    externalMeasurements_[i].age += solver().dt();
    auto requested = mc_rbdyn::RollingContactMode::Rolling;
    if(scenario_ == "mode_cycle")
    {
      const double cycle = std::fmod(elapsed_, commandPeriod_) / commandPeriod_;
      if(cycle < 0.25) { requested = mc_rbdyn::RollingContactMode::Rolling; }
      else if(cycle < 0.5) { requested = mc_rbdyn::RollingContactMode::Fixed; }
      else if(cycle < 0.75) { requested = mc_rbdyn::RollingContactMode::Sliding; }
      else { requested = mc_rbdyn::RollingContactMode::Detached; }
    }
    modeManagers_[i].requestedMode(requested);
    if(lastSolverSuccess_)
    {
      mc_rbdyn::RollingContactModeObservation observation;
      observation.valid = diagnosticsValid_;
      observation.slipSpeed = std::hypot(rollingResiduals_[i], lateralResiduals_[i]);
      observation.rollingResidual = accelerationResiduals_[i];
      observation.normalForce = normalForces_[i];
      observation.frictionMargin = frictionMargins_[i];
      observation.torqueMargin = driveTorqueMargins_[i];
      const auto & measured = externalMeasurements_[i];
      if(measured.valid && measured.age <= 2.0 * solver().dt())
      {
        observation.slipSpeed = std::hypot(measured.rollingSlip, measured.lateralSlip);
        // A physics adapter supplies the authoritative contact velocity. The
        // inactive rolling-row acceleration residual can be intentionally
        // large in sliding/detached modes and must not veto physical recovery.
        // Hard-row promotion is gated separately below using that residual.
        observation.rollingResidual = measured.rollingSlip;
        observation.normalForce = measured.normalForce;
        observation.frictionMargin = wheels_[i].friction * measured.normalForce - measured.tangentialForce;
      }
      const bool hasFreshExternalMeasurement = measured.valid && measured.age <= 2.0 * solver().dt();
      if(modeManagers_[i].state().estimated == mc_rbdyn::RollingContactMode::Detached
         && requested != mc_rbdyn::RollingContactMode::Detached && !hasFreshExternalMeasurement)
      {
        // The headless ticker has no terrain contact sensor. A scheduled
        // re-attachment therefore uses the still-valid terrain geometry as a
        // deterministic contact-presence probe; physics adapters should feed
        // their measured normal force instead.
        const auto & thresholds = modeManagers_[i].thresholds();
        observation.normalForce = thresholds.normalForceEnter + 1.0;
        observation.frictionMargin = thresholds.frictionMarginEnter + 1.0;
        observation.torqueMargin = thresholds.torqueMarginEnter + 1.0;
      }
      if(modeManagers_[i].state().estimated == mc_rbdyn::RollingContactMode::Detached
         && requested != mc_rbdyn::RollingContactMode::Detached)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        if(std::abs(robot().mbc().alpha[drive][0]) * wheels_[i].radius > recoverySpeed_)
        {
          // A geometrically re-established contact is not yet safe to load
          // while the rim is moving quickly. Keep its QP force at zero until
          // the bounded safe-stop target has reduced wheel speed.
          observation.normalForce = 0.0;
        }
      }
      modeManagers_[i].update(observation, solver().dt());
    }
    const auto & state = modeManagers_[i].state();
    if(state.estimated == mc_rbdyn::RollingContactMode::Detached) { hardPromotionPending_[i] = true; }
    double appliedActivation = state.activation;
    if(hardPromotionPending_[i] && state.estimated == mc_rbdyn::RollingContactMode::Rolling
       && appliedActivation >= 1.0 - 1e-12)
    {
      // An activation below one keeps all would-be hard rows as a weighted
      // transition objective. Promote them only after the previous solution
      // demonstrates acceleration-level feasibility.
      appliedActivation = 1.0 - 1e-6;
    }
    appliedActivations_[i] = appliedActivation;
    wheels_[i].mode = state.estimated;
    wheels_[i].activation = state.activation;
    rolling_->mode(wheels_[i].name, state.estimated, appliedActivation);
    dynamics_->mode(wheels_[i].name, state.estimated);
    contactFallback_ = contactFallback_
                       || (requested == mc_rbdyn::RollingContactMode::Rolling
                           && (state.estimated != mc_rbdyn::RollingContactMode::Rolling
                               || hardPromotionPending_[i]));
  }
  bool anyPromotionPending = false;
  bool allReadyForHardPromotion = diagnosticsValid_;
  const double promotionResidualLimit =
      rolling_->options().longitudinal == mc_solver::RollingContactLongitudinal::Hard ? recoveryResidual_
                                                                                     : recoveryTransverseResidual_;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    if(!hardPromotionPending_[i]) { continue; }
    anyPromotionPending = true;
    const auto & state = modeManagers_[i].state();
    const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
    allReadyForHardPromotion =
        allReadyForHardPromotion && state.estimated == mc_rbdyn::RollingContactMode::Rolling
        && state.activation >= 1.0 - 1e-12 && hardPromotionResiduals_[i] <= promotionResidualLimit
        && std::abs(robot().mbc().alpha[drive][0]) * wheels_[i].radius <= recoverySpeed_;
  }
  if(anyPromotionPending && allReadyForHardPromotion)
  {
    std::fill(hardPromotionPending_.begin(), hardPromotionPending_.end(), false);
  }
  if(contactFallback_ && !wasContactFallback)
  {
    const auto & chassis = robot().frame("chassis").position();
    basePositionTarget_ = chassis.translation();
    const Eigen::Vector3d forward = chassis.rotation().transpose() * Eigen::Vector3d::UnitX();
    baseYawTarget_ = std::atan2(forward.dot(terrainTangentY_), forward.dot(terrainTangentX_));
  }
  if(contactFallback_ != wasContactFallback)
  {
    // Transitional rows remain soft until their previous-cycle residuals are
    // safe to promote. Give those rows enough authority to demonstrate
    // feasibility against ordinary tracking tasks on both QP backends.
    rolling_->rollingWeight(contactFallback_ ? recoveryRollingWeight_ : rollingWeight_);
  }
}

void MCRollingContactController::safeStop(const std::string & reason)
{
  diagnosticsValid_ = false;
  invalidReason_ = reason;
  lastSolverSuccess_ = false;
  referenceLinearSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  std::map<std::string, std::vector<double>> targets;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[drive][0];
    targets[wheels_[i].driveJoint] = {driveTargets_[i]};
    if(!wheels_[i].steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
      targets[wheels_[i].steeringJoint] = {robot().mbc().q[steering][0]};
    }
    modeManagers_[i].reset(mc_rbdyn::RollingContactMode::Detached, mc_rbdyn::RollingContactMode::Detached, 0.0);
    hardPromotionPending_[i] = true;
    appliedActivations_[i] = 0.0;
    rolling_->mode(wheels_[i].name, mc_rbdyn::RollingContactMode::Detached);
    dynamics_->mode(wheels_[i].name, mc_rbdyn::RollingContactMode::Detached);
    wheels_[i].mode = mc_rbdyn::RollingContactMode::Detached;
    wheels_[i].activation = 0.0;
  }
  std::fill(normalForces_.begin(), normalForces_.end(), 0.0);
  std::fill(tangentialForces_.begin(), tangentialForces_.end(), 0.0);
  std::fill(frictionMargins_.begin(), frictionMargins_.end(), 0.0);
  std::fill(driveTorques_.begin(), driveTorques_.end(), 0.0);
  std::fill(driveTorqueMargins_.begin(), driveTorqueMargins_.end(), 0.0);
  postureTask->target(targets);
}

void MCRollingContactController::updateDiagnostics(bool solverSuccess)
{
  lastSolverSuccess_ = solverSuccess;
  diagnosticsValid_ = solverSuccess;
  invalidReason_ = solverSuccess ? std::string{} : std::string{"solver-failure"};
  maxRollingResidual_ = 0.0;
  maxLateralResidual_ = 0.0;
  minFrictionMargin_ = std::numeric_limits<double>::infinity();
  const auto & geometry = rolling_->geometryResults();
  Eigen::VectorXd alphaD;
  if(solver().backend() == Backend::TVM)
  {
    alphaD = robot().tvmRobot().alphaD()->value();
    dynamics_->dynamicFunction().updateValue();
    dynamicsResidual_ = dynamics_->dynamicFunction().value().lpNorm<Eigen::Infinity>();
    const auto & effort = robot().tvmRobot().tau()->value();
    const Eigen::Index floatingDof = robot().mb().joint(0).type() == rbd::Joint::Free ? 6 : 0;
    floatingBaseEffortNorm_ = floatingDof == 0 ? 0.0 : effort.head(floatingDof).norm();
  }
  else { alphaD = rbd::dofToVector(robot().mb(), robot().mbc().alphaD); }
  for(size_t i = 0; i < geometry.size(); ++i)
  {
    rollingResiduals_[i] = geometry[i].rollingDirection.dot(geometry[i].slipVelocity);
    lateralResiduals_[i] = geometry[i].lateralDirection.dot(geometry[i].slipVelocity);
    normalResiduals_[i] = geometry[i].normalDirection.dot(geometry[i].velocityResidual);
    const Eigen::Vector3d accelerationResidual = geometry[i].rollingMatrix * alphaD - geometry[i].rhs;
    accelerationResiduals_[i] = std::abs(accelerationResidual.x());
    lateralAccelerationResiduals_[i] = std::abs(accelerationResidual.y());
    normalAccelerationResiduals_[i] = std::abs(accelerationResidual.z());
    // Normal reattachment is certified by the fresh simulator/contact-sensor
    // normal-force hysteresis. A soft normal acceleration row can legitimately
    // retain the gravity term while the contact is unloaded, so using it as a
    // hard-promotion gate would deadlock recovery. Lateral compatibility and,
    // for hard rolling, longitudinal compatibility are checked algebraically.
    hardPromotionResiduals_[i] = lateralAccelerationResiduals_[i];
    if(rolling_->options().longitudinal == mc_solver::RollingContactLongitudinal::Hard)
    {
      hardPromotionResiduals_[i] = std::max(hardPromotionResiduals_[i], accelerationResiduals_[i]);
    }
    maxRollingResidual_ = std::max(maxRollingResidual_, std::abs(rollingResiduals_[i]));
    maxLateralResidual_ = std::max(maxLateralResidual_, std::abs(lateralResiduals_[i]));
    normalForces_[i] = 0.0;
    tangentialForces_[i] = 0.0;
    frictionMargins_[i] = 0.0;
    const auto joint = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTorques_[i] = robot().jointTorque()[joint][0];
    driveTorqueMargins_[i] = std::min(robot().tu()[joint][0] - driveTorques_[i],
                                      driveTorques_[i] - robot().tl()[joint][0]);
  }
  if(solverSuccess)
  {
    if(solver().backend() == Backend::Tasks)
    {
      const auto & tasksSolver = static_cast<mc_solver::TasksQPSolver &>(solver());
      const auto & lambda = tasksSolver.solver().lambdaVec();
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        const auto & wheel = wheels_[i];
        const int begin = dynamics_->lambdaBegin(wheel.name) - tasksSolver.data().lambdaBegin();
        if(begin >= 0 && begin + dynamics_->lambdaCount(wheel.name) <= lambda.size())
        {
          normalForces_[i] = dynamics_->normalForce(wheel.name, lambda.segment(begin, 8));
          tangentialForces_[i] = dynamics_->tangentialForce(wheel.name, lambda.segment(begin, 8));
          frictionMargins_[i] = dynamics_->frictionMargin(wheel.name, lambda.segment(begin, 8));
          minFrictionMargin_ = std::min(minFrictionMargin_, frictionMargins_[i]);
        }
      }
    }
    else
    {
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        normalForces_[i] = dynamics_->normalForce(wheels_[i].name);
        tangentialForces_[i] = dynamics_->tangentialForce(wheels_[i].name);
        frictionMargins_[i] = dynamics_->frictionMargin(wheels_[i].name);
        minFrictionMargin_ = std::min(minFrictionMargin_, frictionMargins_[i]);
      }
    }
  }
  if(!std::isfinite(minFrictionMargin_)) { minFrictionMargin_ = 0.0; }
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    diagnosticsValid_ = diagnosticsValid_ && std::isfinite(rollingResiduals_[i])
                        && std::isfinite(lateralResiduals_[i]) && std::isfinite(normalResiduals_[i])
                        && std::isfinite(accelerationResiduals_[i])
                        && std::isfinite(lateralAccelerationResiduals_[i])
                        && std::isfinite(normalAccelerationResiduals_[i]) && std::isfinite(hardPromotionResiduals_[i])
                        && std::isfinite(normalForces_[i])
                        && std::isfinite(tangentialForces_[i]) && std::isfinite(frictionMargins_[i])
                        && std::isfinite(driveTorques_[i]) && std::isfinite(driveTorqueMargins_[i]);
  }
  if(!diagnosticsValid_ && invalidReason_.empty()) { invalidReason_ = "non-finite-diagnostic"; }
}

bool MCRollingContactController::run()
{
  const auto begin = std::chrono::steady_clock::now();
  bool success = false;
  try
  {
    if(closedLoopFeedback_)
    {
      robot().mbc().q = realRobot().mbc().q;
      robot().mbc().alpha = realRobot().mbc().alpha;
      robot().forwardKinematics();
      robot().forwardVelocity();
    }
    updateModes();
    updateReference();
    success = MCController::run();
    updateDiagnostics(success);
    if(!success) { safeStop("solver-failure"); }
  }
  catch(const std::exception & error)
  {
    safeStop(std::string{"runtime-exception: "} + error.what());
    mc_rtc::log::error("RollingContact controller contained runtime failure: {}", error.what());
  }
  elapsed_ += solver().dt();
  totalTimeMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  solveAndBuildTimeMs_ = solver().solveAndBuildTime();
  updateTimeMs_ = std::max(0.0, totalTimeMs_ - solveAndBuildTimeMs_);
  return success;
}

} // namespace mc_control

MULTI_CONTROLLERS_CONSTRUCTOR(
    "RollingContact",
    mc_control::MCRollingContactController(rm, dt, config, mc_control::MCController::Backend::Tasks),
    "RollingContact_TVM",
    mc_control::MCRollingContactController(rm, dt, config, mc_control::MCController::Backend::TVM))
