/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "mc_rolling_contact_controller.h"

#include <mc_rbdyn/CylindricalSurface.h>
#include <mc_rtc/gui/Button.h>
#include <mc_rtc/gui/Label.h>
#include <mc_rtc/gui/NumberInput.h>
#include <mc_rtc/logging.h>
#include <mc_solver/TasksQPSolver.h>

#include <mc_tvm/Robot.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <sstream>

#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
#  include <RoboticsUtils/KeyboardCapture.h>
#endif

namespace mc_control
{

namespace
{

// Keep a small additional margin below the +/-90 degree mechanical limit in
// the keyboard crab mode. The exact limit is a singular configuration for
// the contact QP (and for the MuJoCo hinge servo); 1.55 rad still produces
// essentially lateral motion while leaving the solver and actuator room.
constexpr double keyboardSteeringLimit = 1.55;

mc_solver::RollingContactLongitudinal longitudinalMode(const mc_rtc::Configuration & config)
{
  const std::string value = config("longitudinal", std::string{"hard"});
  if(value == "hard") { return mc_solver::RollingContactLongitudinal::Hard; }
  if(value == "soft") { return mc_solver::RollingContactLongitudinal::Soft; }
  mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact longitudinal must be hard or soft, got {}",
                                                      value);
}

} // namespace

struct MCRollingContactController::KeyboardInput
{
  bool start(double linearVelocity, double angularVelocity, int pollIntervalMs, char exitKey)
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    const char normalizedExitKey = static_cast<char>(std::tolower(static_cast<unsigned char>(exitKey)));
    // KeyboardCapture reports key events, but its internal command is derived
    // from only the key read during the current poll. A normal terminal sends
    // one byte per press (and does not report releases), so preserve each
    // command here until another command or the operator stops the command.
    capture.setKeyPressCallback([this, linearVelocity, angularVelocity, normalizedExitKey](char key)
                                 { handleKey(key, linearVelocity, angularVelocity, normalizedExitKey); });
    // Handle the exit key in the controller thread. RoboticsUtils invokes the
    // key callback from its polling thread, so enabling its built-in exit
    // path here would make that thread attempt to join itself during stop().
    RoboticsUtils::KeyboardCapture::Config config(
        linearVelocity, angularVelocity, pollIntervalMs, false, normalizedExitKey);
    return capture.start(config);
#else
    static_cast<void>(linearVelocity);
    static_cast<void>(angularVelocity);
    static_cast<void>(pollIntervalMs);
    static_cast<void>(exitKey);
    return false;
#endif
  }

  std::array<double, 3> command()
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    // The key callback runs on KeyboardCapture's worker. Defer stop() to this
    // controller-thread poll so the worker can exit and be joined safely.
    if(stopRequested.exchange(false)) { capture.stop(); }
    // Treat a stopped capture as an explicit zero command so the robot cannot
    // continue driving after the controller-side exit key has been processed.
    if(!capture.isRunning()) { return {0.0, 0.0, 0.0}; }
    return {vx.load(), vy.load(), wz.load()};
#else
    return {0.0, 0.0, 0.0};
#endif
  }

  bool running() const
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    return capture.isRunning();
#else
    return false;
#endif
  }

  void clear()
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    vx.store(0.0);
    vy.store(0.0);
    wz.store(0.0);
#endif
  }

  std::string status() const
  {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
    // KeyboardCapture's status describes only the byte seen during its latest
    // poll. The controller deliberately latches commands until another
    // command, space, or x arrives, so expose that authoritative latched state
    // to the GUI/log instead of the transient helper state.
    const double forward = vx.load();
    const double lateral = vy.load();
    const double yaw = wz.load();
    std::ostringstream status;
    status << "W:" << (forward > 0.0 ? "1" : "0") << " S:" << (forward < 0.0 ? "1" : "0")
           << " A:" << (lateral < 0.0 ? "1" : "0") << " D:" << (lateral > 0.0 ? "1" : "0")
           << " Q:" << (yaw < 0.0 ? "1" : "0") << " E:" << (yaw > 0.0 ? "1" : "0");
    return status.str();
#else
    return "unavailable";
#endif
  }

#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
  void handleKey(char key, double linearVelocity, double angularVelocity, char exitKey)
  {
    key = static_cast<char>(std::tolower(static_cast<unsigned char>(key)));
    if(key == exitKey || key == ' ')
    {
      vx.store(0.0);
      vy.store(0.0);
      wz.store(0.0);
      if(key == exitKey) { stopRequested.store(true); }
      return;
    }
    switch(key)
    {
      case 'w':
        vx.store(linearVelocity);
        // Translation keys are an explicit change away from a pure yaw
        // command.  Terminal input has no key-release event, therefore a
        // previously latched Q/E would otherwise remain active and turn a
        // requested straight/crab motion into an unintended circle.
        wz.store(0.0);
        break;
      case 's':
        vx.store(-linearVelocity);
        wz.store(0.0);
        break;
      case 'a':
        vy.store(-linearVelocity);
        wz.store(0.0);
        break;
      case 'd':
        vy.store(linearVelocity);
        wz.store(0.0);
        break;
      case 'q':
        wz.store(-angularVelocity);
        break;
      case 'e':
        wz.store(angularVelocity);
        break;
      default:
        break;
    }
  }

  RoboticsUtils::KeyboardCapture capture;
  std::atomic<double> vx{0.0};
  std::atomic<double> vy{0.0};
  std::atomic<double> wz{0.0};
  std::atomic<bool> stopRequested{false};
#endif
};

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
  if(robot().name() != "rolling_diff" && robot().name() != "rolling_4s" && robot().name() != "ranger_mini_v3")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact controller requires rolling_diff, rolling_4s, or ranger_mini_v3, got {}", robot().name());
  }
  fourSteering_ = robot().name() != "rolling_diff";
  const auto settings = config.has("RollingContact") ? config("RollingContact") : mc_rtc::Configuration{};
  scenario_ = settings("scenario", std::string{"hold"});
  closedLoopFeedback_ = settings("closedLoopFeedback", false);
  positionFeedbackGain_ = settings("positionFeedbackGain", 5.0);
  linearSpeed_ = settings("linearSpeed", 0.2);
  yawRate_ = settings("yawRate", 0.35);
  steeringAngle_ = settings("steeringAngle", 0.3);
  commandPeriod_ = settings("commandPeriod", 4.0);
  recoverySpeed_ = settings("recoverySpeed", 0.05);
  recoveryResidual_ = settings("recoveryResidual", 0.2);
  recoveryTransverseResidual_ = settings("recoveryTransverseResidual", 0.5);
  rollingWeight_ = settings("rollingWeight", 1000.0);
  recoveryRollingWeight_ = settings("recoveryRollingWeight", 1e6);
  if(!std::isfinite(commandPeriod_) || commandPeriod_ <= 0.0 || !std::isfinite(positionFeedbackGain_)
     || positionFeedbackGain_ < 0.0 || !std::isfinite(linearSpeed_)
     || !std::isfinite(yawRate_) || !std::isfinite(steeringAngle_) || !std::isfinite(recoverySpeed_)
     || recoverySpeed_ < 0.0 || !std::isfinite(recoveryResidual_) || recoveryResidual_ < 0.0
     || !std::isfinite(recoveryTransverseResidual_) || recoveryTransverseResidual_ < 0.0
     || !std::isfinite(rollingWeight_) || rollingWeight_ <= 0.0 || !std::isfinite(recoveryRollingWeight_)
     || recoveryRollingWeight_ < rollingWeight_)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact commandPeriod and rolling weights must be positive, positionFeedbackGain and recovery "
        "thresholds non-negative, "
        "recoveryRollingWeight must be at least rollingWeight, and command scalars must be finite");
  }
  const std::vector<std::string> differentialScenarios = {"hold",       "reverse",      "forward",
                                                           "turn_left",  "turn_right",   "circle_left",
                                                           "circle_right", "sinusoid",   "unequal_radii",
                                                           "infeasible_soft", "mode_cycle"};
  const std::vector<std::string> steeringScenarios = {"hold",       "forward",       "reverse",
                                                       "crab",       "ackermann_left", "ackermann_right",
                                                       "pure_yaw",   "steering_rate", "incompatible",
                                                       "mode_cycle", "keyboard"};
  const auto & supported = fourSteering_ ? steeringScenarios : differentialScenarios;
  if(std::find(supported.begin(), supported.end(), scenario_) == supported.end())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario '{}' for {}", scenario_,
                                                        robot().name());
  }

  wheels_ = makeWheels();
  wheelOffsets_.reserve(wheels_.size());
  const auto & chassis = robot().frame("chassis").position();
  for(const auto & wheel : wheels_)
  {
    const Eigen::Vector3d worldOffset =
        robot().frame(wheel.carrierFrame).position().translation() - chassis.translation();
    wheelOffsets_.push_back((chassis.rotation() * worldOffset).head<2>());
  }
  if(scenario_ == "keyboard")
  {
    const double keyboardLinearSpeed = settings("keyboardLinearSpeed", 0.3);
    const double keyboardAngularSpeed = settings("keyboardAngularSpeed", 0.5);
    const int keyboardPollIntervalMs = settings("keyboardPollIntervalMs", 10);
    const std::string keyboardExitKey = settings("keyboardExitKey", std::string{"x"});
    keyboardWheelPositionLookahead_ = settings("keyboardWheelPositionLookahead", 0.0);
    keyboardYawScale_ = settings("keyboardYawScale", 1.7);
    keyboardMixedDriveScale_ = settings("keyboardMixedDriveScale", 1.15);
    keyboardYawFeedbackGain_ = settings("keyboardYawFeedbackGain", 0.5);
    keyboardSteeringRate_ = settings("keyboardSteeringRate", 4.0);
    keyboardDriveAcceleration_ = settings("keyboardDriveAcceleration", 20.0);
    if(!std::isfinite(keyboardLinearSpeed) || keyboardLinearSpeed <= 0.0 || !std::isfinite(keyboardAngularSpeed)
       || keyboardAngularSpeed <= 0.0 || keyboardPollIntervalMs <= 0 || keyboardExitKey.size() != 1
       || !std::isfinite(keyboardWheelPositionLookahead_) || keyboardWheelPositionLookahead_ < 0.0
       || !std::isfinite(keyboardYawScale_) || keyboardYawScale_ <= 0.0
       || !std::isfinite(keyboardMixedDriveScale_) || keyboardMixedDriveScale_ <= 0.0
       || !std::isfinite(keyboardYawFeedbackGain_) || keyboardYawFeedbackGain_ < 0.0
       || !std::isfinite(keyboardSteeringRate_) || keyboardSteeringRate_ <= 0.0
       || !std::isfinite(keyboardDriveAcceleration_) || keyboardDriveAcceleration_ <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact keyboard speeds and poll interval must be positive, keyboardExitKey one character, "
          "keyboardWheelPositionLookahead non-negative, keyboardYawScale and keyboardMixedDriveScale positive, "
          "and keyboardYawFeedbackGain non-negative");
    }
    keyboard_ = std::make_unique<KeyboardInput>();
    if(!keyboard_->start(keyboardLinearSpeed, keyboardAngularSpeed, keyboardPollIntervalMs, keyboardExitKey[0]))
    {
#ifdef MC_ROLLING_CONTACT_HAS_KEYBOARD
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "RollingContact keyboard capture requires an interactive terminal on standard input");
#else
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "RollingContact was built without RoboticsUtils::Keyboard; install RoboticsUtils and rebuild mc_rtc");
#endif
    }
    mc_rtc::log::info("RollingContact keyboard: W/S/A/D clear a latched Q/E yaw; Q/E add yaw to translation; "
                      "press '{}' to stop capture and zero velocity references, space to clear the axes",
                      keyboardExitKey[0]);
  }
  gui()->addElement(
      {"Rolling Contact", "Command"},
      mc_rtc::gui::NumberInput(
          "Forward velocity", [this]() { return guiForwardCommand_; },
          [this](double value)
          {
            if(std::isfinite(value)) { guiForwardCommand_ = value; }
          }),
      mc_rtc::gui::NumberInput(
          "Lateral velocity", [this]() { return guiLateralCommand_; },
          [this](double value)
          {
            if(std::isfinite(value)) { guiLateralCommand_ = value; }
          }),
      mc_rtc::gui::NumberInput(
          "Yaw velocity", [this]() { return guiYawCommand_; },
          [this](double value)
          {
            if(std::isfinite(value)) { guiYawCommand_ = value; }
          }),
      mc_rtc::gui::Button(
          "Clear command", [this]()
          {
            guiForwardCommand_ = 0.0;
            guiLateralCommand_ = 0.0;
            guiYawCommand_ = 0.0;
            if(keyboard_) { keyboard_->clear(); }
          }));
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
  wheelReferenceRates_.resize(wheels_.size(), 0.0);
  steeringTargets_.resize(wheels_.size(), 0.0);
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
  datastore().make_call("RollingContact::GetKeyboardStatus", [this]()
                        { return keyboard_ ? keyboard_->status() : std::string{"disabled"}; });
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
  logger().addLogEntry("RollingContact_reference_lateral_speed", [this]() { return referenceLateralSpeed_; });
  logger().addLogEntry("RollingContact_reference_yaw_rate", [this]() { return referenceYawRate_; });
  logger().addLogEntry("RollingContact_gui_forward_command", [this]() { return guiForwardCommand_; });
  logger().addLogEntry("RollingContact_gui_lateral_command", [this]() { return guiLateralCommand_; });
  logger().addLogEntry("RollingContact_gui_yaw_command", [this]() { return guiYawCommand_; });
  logger().addLogEntry("RollingContact_keyboard_status", [this]()
                       { return keyboard_ ? keyboard_->status() : std::string{"disabled"}; });
  logger().addLogEntry("RollingContact_keyboard_running", [this]() { return keyboard_ && keyboard_->running(); });
  logger().addLogEntry("RollingContact_keyboard_transition_grace",
                       [this]() { return keyboardCommandTransitionGrace_; });
  logger().addLogEntry("RollingContact_keyboard_steering_ready", [this]() { return keyboardSteeringReady_; });
  logger().addLogEntry("RollingContact_keyboard_yaw_error", [this]() { return keyboardYawError_; });
  logger().addLogEntry("RollingContact_keyboard_yaw_correction", [this]() { return keyboardYawCorrection_; });
  logger().addLogEntry("RollingContact_base_position_target", [this]() { return basePositionTarget_; });
  logger().addLogEntry("RollingContact_base_yaw_target", [this]() { return baseYawTarget_; });
  logger().addLogEntry("RollingContact_base_position_reference_velocity", [this]()
                       { return baseReferenceVelocity_; });
  logger().addLogEntry("RollingContact_base_tracking_velocity", [this]() { return baseTrackingVelocity_; });
  logger().addLogEntry("RollingContact_base_angular_reference_velocity", [this]()
                       { return baseReferenceAngularVelocity_; });
  logger().addLogEntry("RollingContact_base_position_tracking_error", [this]()
                       {
                         const auto & state = closedLoopFeedback_ ? realRobot() : robot();
                         return (basePositionTarget_ - state.posW().translation()).norm();
                       });
  // In closed-loop operation robot() is the controller/output state and may
  // contain the one-step prediction produced by the QP. realRobot() is the
  // measured state and is what an interface/diagnostic consumer expects for a
  // floating-base pose/twist. Keep the open-loop behavior unchanged.
  logger().addLogEntry("RollingContact_base_pose", [this]() -> const sva::PTransformd &
                       { return closedLoopFeedback_ ? realRobot().posW() : robot().posW(); });
  logger().addLogEntry("RollingContact_base_twist", [this]() -> const sva::MotionVecd &
                       { return closedLoopFeedback_ ? realRobot().velW() : robot().velW(); });
  // The FloatingBase packet and the chassis frame must describe the same
  // physical origin. Compare the measured robot (not the one-step predicted
  // control state) so this remains a meaningful runtime model check.
  logger().addLogEntry("RollingContact_floating_base_chassis_error", [this]()
                       {
                         const auto & measured = closedLoopFeedback_ ? realRobot() : robot();
                         if(!measured.hasBodySensor("FloatingBase"))
                         {
                           return std::numeric_limits<double>::infinity();
                         }
                         const auto & sensor = measured.bodySensor("FloatingBase");
                         const sva::PTransformd sensorPose(sensor.orientation(), sensor.position());
                         return (sensorPose.matrix() - measured.frame("chassis").position().matrix()).norm();
                       });
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto & wheel = wheels_[i];
    const auto joint = robot().jointIndexByName(wheel.driveJoint);
    logger().addLogEntry("RollingContact_" + wheel.name + "_position",
                         [this, joint]() { return robot().mbc().q[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rate",
                         [this, joint]() { return robot().mbc().alpha[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_measured_rate",
                         [this, joint]()
                         {
                           const auto & refJointOrder = robot().refJointOrder();
                           const auto it = std::find(refJointOrder.begin(), refJointOrder.end(),
                                                     robot().mb().joint(joint).name());
                           const auto & velocities = robot().encoderVelocities();
                           if(it != refJointOrder.end())
                           {
                             const auto index = static_cast<size_t>(std::distance(refJointOrder.begin(), it));
                             if(index < velocities.size() && std::isfinite(velocities[index]))
                             {
                               return velocities[index];
                             }
                           }
                           return robot().mbc().alpha[joint][0];
                         });
    logger().addLogEntry("RollingContact_" + wheel.name + "_acceleration",
                         [this, joint]() { return robot().mbc().alphaD[joint][0]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_target", [this, i]() { return driveTargets_[i]; });
    if(!wheel.steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheel.steeringJoint);
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_position",
                           [this, steering]() { return robot().mbc().q[steering][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_target",
                           [this, i]() { return steeringTargets_[i]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_rate",
                           [this, steering]() { return robot().mbc().alpha[steering][0]; });
    }
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
      logger().addLogEntry("RollingContact_" + wheel.name + "_measured_steering_position",
                           [this, steeringJoint]()
                           {
                             // The MuJoCo adapter publishes encoder values on the
                             // controlled robot. Use that stream for the measured
                             // steering angle instead of realRobot(), whose MBC is
                             // not updated when no observer pipeline is configured.
                             const auto & refJointOrder = robot().refJointOrder();
                             const auto it = std::find(refJointOrder.begin(), refJointOrder.end(),
                                                       robot().mb().joint(steeringJoint).name());
                             const auto & encoders = robot().encoderValues();
                             if(it != refJointOrder.end())
                             {
                               const auto index = static_cast<size_t>(std::distance(refJointOrder.begin(), it));
                               if(index < encoders.size() && std::isfinite(encoders[index])) { return encoders[index]; }
                             }
                             return robot().mbc().q[steeringJoint][0];
                           });
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
  // Restore terminal settings and join the keyboard thread before tearing down
  // any controller state that the interactive loop can indirectly observe.
  keyboard_.reset();
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
  const std::vector<std::string> surfaces = fourSteering_
                                                ? std::vector<std::string>{"FrontLeftWheel", "FrontRightWheel",
                                                                           "RearLeftWheel", "RearRightWheel"}
                                                : std::vector<std::string>{"LeftWheel", "RightWheel"};
  for(size_t i = 0; i < names.size(); ++i)
  {
    const auto & name = names[i];
    mc_rbdyn::RollingContactDescription wheel;
    wheel.name = name;
    wheel.carrierFrame = name + "_carrier";
    wheel.wheelBody = name + "_wheel";
    wheel.driveJoint = name + "_drive";
    if(fourSteering_) { wheel.steeringJoint = name + "_steer"; }
    const auto * surface = dynamic_cast<const mc_rbdyn::CylindricalSurface *>(&robot().surface(surfaces[i]));
    if(!surface)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact surface {} must be cylindrical", surfaces[i]);
    }
    wheel.radius = surface->radius();
    if(!fourSteering_ && scenario_ == "unequal_radii") { wheel.radius = name == "left" ? 0.18 : 0.22; }
    wheel.width = surface->width();
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
  keyboardStopLatched_ = false;
  keyboardCaptureWasRunning_ = false;
  keyboardSteeringReady_ = true;
  keyboardCommandTransitionGrace_ = 0.0;
  lastKeyboardForward_ = 0.0;
  lastKeyboardLateral_ = 0.0;
  lastKeyboardYaw_ = 0.0;
  keyboardCommandInitialized_ = false;
  updateTimeMs_ = 0.0;
  solveAndBuildTimeMs_ = 0.0;
  totalTimeMs_ = 0.0;
  maxRollingResidual_ = 0.0;
  maxLateralResidual_ = 0.0;
  minFrictionMargin_ = 0.0;
  referenceLinearSpeed_ = 0.0;
  referenceLateralSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  guiForwardCommand_ = 0.0;
  guiLateralCommand_ = 0.0;
  guiYawCommand_ = 0.0;
  if(keyboard_) { keyboard_->clear(); }
  dynamicsResidual_ = 0.0;
  floatingBaseEffortNorm_ = 0.0;
  basePositionTarget_ = robot().frame("chassis").position().translation();
  baseReferenceVelocity_.setZero();
  baseTrackingVelocity_.setZero();
  baseReferenceAngularVelocity_.setZero();
  // Start the accumulated yaw reference at the measured chassis heading. This
  // keeps the first Q/E command continuous even when the simulator is reset
  // with a non-zero initial orientation.
  const Eigen::Vector3d initialHeading = robot().posW().rotation().col(0);
  const double initialYaw = std::atan2(initialHeading.dot(terrainTangentY_),
                                       initialHeading.dot(terrainTangentX_));
  baseYawTarget_ = std::isfinite(initialYaw) ? initialYaw : 0.0;
  keyboardYawCorrection_ = 0.0;
  keyboardYawError_ = 0.0;
  diagnosticsValid_ = false;
  invalidReason_ = "not-run";
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto joint = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[joint][0];
    wheelReferenceRates_[i] = 0.0;
    if(!wheels_[i].steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
      steeringTargets_[i] = robot().mbc().q[steering][0];
    }
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

void MCRollingContactController::updateKeyboardCommandTransition()
{
  if(scenario_ != "keyboard" || !keyboard_) { return; }

  // updateModes() runs before updateReference(). Detect the operator command
  // here so the very first cycle after Q/E -> W/A (or any other axis change)
  // already receives the contact-safe steering-transition grace period. If
  // detection were deferred until updateReference(), updateModes() would
  // evaluate one cycle using the old drive velocity while the steering target
  // had already changed, which can make the hard rolling QP infeasible.
  const auto command = keyboard_->command();
  const double forward = command[0] + guiForwardCommand_;
  const double lateral = -command[1] + guiLateralCommand_;
  const double yaw = -command[2] + guiYawCommand_;
  const bool commandChanged = !keyboardCommandInitialized_
                              || std::abs(forward - lastKeyboardForward_) > 1e-6
                              || std::abs(lateral - lastKeyboardLateral_) > 1e-6
                              || std::abs(yaw - lastKeyboardYaw_) > 1e-6;
  if(commandChanged) { keyboardCommandTransitionGrace_ = 0.5; }
  keyboardCommandInitialized_ = true;
  lastKeyboardForward_ = forward;
  lastKeyboardLateral_ = lateral;
  lastKeyboardYaw_ = yaw;
}

void MCRollingContactController::updateReference()
{
  const double phase = 2.0 * 3.14159265358979323846 * elapsed_ / commandPeriod_;
  keyboardYawCorrection_ = 0.0;
  keyboardYawError_ = 0.0;
  double forward = linearSpeed_;
  double lateral = 0.0;
  double yaw = yawRate_;
  if(scenario_ == "hold") { forward = yaw = 0.0; }
  else if(scenario_ == "reverse") { forward = -std::abs(linearSpeed_); yaw = 0.0; }
  else if(scenario_ == "forward" || scenario_ == "unequal_radii" || scenario_ == "infeasible_soft")
  {
    yaw = 0.0;
  }
  else if(scenario_ == "mode_cycle") { forward = yaw = 0.0; }
  else if(scenario_ == "turn_left") { forward = 0.0; yaw = std::abs(yawRate_); }
  else if(scenario_ == "turn_right") { forward = 0.0; yaw = -std::abs(yawRate_); }
  else if(scenario_ == "circle_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "circle_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "sinusoid") { yaw *= std::sin(phase); }
  else if(scenario_ == "crab")
  {
    lateral = forward * std::sin(steeringAngle_);
    forward *= std::cos(steeringAngle_);
    yaw = 0.0;
  }
  else if(scenario_ == "steering_rate")
  {
    const double steering = steeringAngle_ * std::sin(phase);
    lateral = forward * std::sin(steering);
    forward *= std::cos(steering);
    yaw = 0.0;
  }
  else if(scenario_ == "incompatible") { yaw = 0.0; }
  else if(scenario_ == "ackermann_left") { yaw = std::abs(yawRate_); }
  else if(scenario_ == "ackermann_right") { yaw = -std::abs(yawRate_); }
  else if(scenario_ == "pure_yaw") { forward = 0.0; }
  else if(scenario_ == "keyboard")
  {
    const auto command = keyboard_->command();
    const bool keyboardRunning = keyboard_->running();
    // The exit key stops the capture worker. Clear GUI overrides at the same
    // poll so x is an unconditional stop, even when a ticker value is set.
    if(!keyboardRunning)
    {
      guiForwardCommand_ = 0.0;
      guiLateralCommand_ = 0.0;
      guiYawCommand_ = 0.0;
      if(keyboardCaptureWasRunning_)
      {
        // x is an explicit emergency stop. Retarget the held heading to the
        // measured pose at that instant so braking does not leave an
        // unreachable orientation error that would keep the wheels turning.
        const Eigen::Vector3d measuredHeading = robot().posW().rotation().col(0);
        const double measuredYaw = std::atan2(measuredHeading.dot(terrainTangentY_),
                                              measuredHeading.dot(terrainTangentX_));
        if(std::isfinite(measuredYaw)) { baseYawTarget_ = measuredYaw; }
      }
    }
    keyboardCaptureWasRunning_ = keyboardRunning;
    forward = command[0] + guiForwardCommand_;
    // RoboticsUtils uses right-positive lateral velocity and clockwise-positive
    // yaw. Convert to the robot convention: +Y left and +Z counter-clockwise.
    lateral = -command[1] + guiLateralCommand_;
    yaw = -command[2] + guiYawCommand_;
  }
  else if(scenario_ != "hold")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario: {}", scenario_);
  }
  referenceLinearSpeed_ = forward;
  referenceLateralSpeed_ = lateral;
  referenceYawRate_ = yaw;
  if(contactFallback_)
  {
    forward = 0.0;
    lateral = 0.0;
    yaw = 0.0;
    referenceLinearSpeed_ = 0.0;
    referenceLateralSpeed_ = 0.0;
    referenceYawRate_ = 0.0;
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      driveTargets_[i] = robot().mbc().q[drive][0];
    }
  }

  const bool keyboardSteeringTransition =
      scenario_ == "keyboard"
      && keyboardCommandTransitionGrace_ > 0.0;
  const bool keyboardSteeringHold = keyboardSteeringTransition
                                    || (scenario_ == "keyboard" && fourSteering_ && !keyboardSteeringReady_);
  // Keep the operator-visible references untouched, but pause the chassis
  // trajectory while a wheel hinge is slewing. The wheel IK below still uses
  // the requested twist to choose the new steering pose; only drive and
  // chassis velocity outputs are held at zero during this bounded transition.
  const double trajectoryForward = keyboardSteeringHold ? 0.0 : forward;
  const double trajectoryLateral = keyboardSteeringHold ? 0.0 : lateral;
  const double trajectoryYaw = keyboardSteeringHold ? 0.0 : yaw;
  if(scenario_ == "keyboard")
  {
    if(closedLoopFeedback_)
    {
      // The Ranger wheel-drive convention is mirrored relative to the
      // controller's positive yaw convention. Integrating the requested yaw
      // here would therefore make the translational target turn opposite to
      // the measured chassis and the position error would grow without bound
      // during W+Q/W+E. In closed loop, the sensor is authoritative: follow
      // its current heading and let the orientation task's velocity reference
      // carry the operator's yaw command.
      const Eigen::Vector3d measuredHeading = robot().posW().rotation().col(0);
      const double measuredYaw = std::atan2(measuredHeading.dot(terrainTangentY_),
                                            measuredHeading.dot(terrainTangentX_));
      if(std::isfinite(measuredYaw))
      {
        // Keep the logged target continuous while following the wrapped
        // sensor heading. The shortest delta is sufficient at the controller
        // rate and prevents a +/-pi crossing from reversing the trajectory
        // heading or failing the monotonic-yaw regression.
        baseYawTarget_ += std::remainder(measuredYaw - baseYawTarget_, 2.0 * 3.14159265358979323846);
      }
      else { baseYawTarget_ += trajectoryYaw * solver().dt(); }
    }
    else
    {
      // Keep the keyboard yaw reference unwrapped in open loop. The heading
      // below is periodic, but retaining the accumulated value avoids a
      // discontinuous scalar target at +/-pi.
      baseYawTarget_ += trajectoryYaw * solver().dt();
    }
  }
  else
  {
    baseYawTarget_ = std::remainder(baseYawTarget_ + yaw * solver().dt(), 2.0 * 3.14159265358979323846);
  }
  // Robot::posW().rotation() (and the MuJoCo FloatingBase sensor) stores the
  // inertial-to-body rotation. Its first column therefore carries the
  // opposite signed yaw from the world direction in which the chassis' +X
  // axis actually travels. Use the reflected heading for closed-loop
  // keyboard translation; otherwise a W+Q/W+E command follows the mirrored
  // circle and accumulates a metre-scale position-task error even though the
  // measured body-forward speed is correct.
  const double trajectoryHeadingYaw = scenario_ == "keyboard" && closedLoopFeedback_ ? -baseYawTarget_ : baseYawTarget_;
  const Eigen::Vector3d heading = std::cos(trajectoryHeadingYaw) * terrainTangentX_
                                  + std::sin(trajectoryHeadingYaw) * terrainTangentY_;
  const Eigen::Vector3d side = -std::sin(trajectoryHeadingYaw) * terrainTangentX_
                               + std::cos(trajectoryHeadingYaw) * terrainTangentY_;
  baseReferenceVelocity_ = trajectoryForward * heading + trajectoryLateral * side;
  baseReferenceAngularVelocity_ = trajectoryYaw * terrainNormal_;
  const bool keyboardStoppedCommand = scenario_ == "keyboard" && std::abs(forward) < 1e-12
                                      && std::abs(lateral) < 1e-12 && std::abs(yaw) < 1e-12;
  if(keyboardStoppedCommand)
  {
    // A keyboard release means "hold here". The MuJoCo adapter may still
    // integrate the last interpolated wheel command for a few frames while
    // the transition grace period expires; refresh the hold target from the
    // measured chassis pose until the robot is fully stopped instead of
    // leaving a permanent position-task error that cannot be corrected with
    // zero wheel rates.
    basePositionTarget_ = robot().posW().translation();
    keyboardStopLatched_ = true;
  }
  else
  {
    keyboardStopLatched_ = false;
  }
  // Keep the absolute position target fixed while a steering transition is
  // still in progress. The readiness flag is updated after the wheel IK has
  // evaluated the current measured hinge positions below; using its previous
  // value here holds the target for the first cycle of a newly detected
  // transition as well.
  const bool keyboardSteeringReady = scenario_ != "keyboard" || !fourSteering_ || keyboardSteeringReady_;
  // While the wheels are turning toward a new steering command, hold the
  // absolute trajectory target. This prevents the feedback tasks from
  // accumulating an error while the wheels are deliberately braked for
  // contact-safe steering alignment.
  if(keyboardSteeringReady)
  {
    basePositionTarget_ += solver().dt() * (trajectoryForward * heading + trajectoryLateral * side);
  }
  // Feed the measured position error back into the wheel references. The
  // integrated target remains the user's requested trajectory, while this
  // correction compensates for the small velocity error introduced by the
  // simulator's wheel dynamics and makes the absolute target asymptotically
  // trackable instead of leaving a permanent position offset.
  baseTrackingVelocity_ = baseReferenceVelocity_;
  // Keep the wheel IK tied to the operator's commanded twist.  The absolute
  // chassis task may add a small feedback velocity to compensate simulator
  // drift, but feeding that correction back into the steering geometry makes
  // a constant W+Q/E command continuously change its ICC as the pose error
  // evolves; the hinges then chase a moving target and the robot can appear
  // stuck (or accumulate rolling residual).  The chassis task remains the
  // mechanism that corrects the pose error, while wheel steering/rates retain
  // the requested radius until the next keyboard command change.
  double controlForward = forward;
  double controlLateral = lateral;
  if(closedLoopFeedback_ && scenario_ == "keyboard" && !contactFallback_ && !keyboardSteeringHold
     && keyboardSteeringReady
     && positionFeedbackGain_ > 0.0)
  {
    const Eigen::Vector3d positionError = basePositionTarget_ - robot().posW().translation();
    const Eigen::Vector3d feedbackVelocity =
        positionFeedbackGain_ * (positionError.dot(heading) * heading + positionError.dot(side) * side);
    baseTrackingVelocity_ += feedbackVelocity;
  }
  basePositionTask_->position(basePositionTarget_);
  // The integrated position target provides the absolute trajectory, while
  // the velocity feed-forward makes the chassis follow that trajectory from
  // the first cycle. Without this, the target can move indefinitely while a
  // low-error position task continues to request zero chassis velocity.
  const Eigen::VectorXd basePositionReferenceVelocity = baseTrackingVelocity_;
  basePositionTask_->refVel(basePositionReferenceVelocity);
  Eigen::Matrix3d baseRotation;
  if(scenario_ == "keyboard" && closedLoopFeedback_)
  {
    // In closed-loop MuJoCo operation the measured chassis pose is the
    // authoritative state. Use it as the orientation task target and express
    // the keyboard yaw as a velocity reference; integrating an absolute
    // desired heading while the simulator lags would otherwise build an
    // unbounded orientation error and eventually make the contact QP fail.
    // orientation() below receives baseRotation.transpose(), so transpose the
    // measured world pose here to preserve the same body/world convention.
    // Passing the measured matrix without this transpose would command the
    // inverse yaw and fight the wheel-induced rotation.
    baseRotation = robot().posW().rotation().transpose();
  }
  else
  {
    baseRotation.col(0) = heading;
    baseRotation.col(1) = side;
    baseRotation.col(2) = terrainNormal_;
  }
  baseOrientationTask_->orientation(baseRotation.transpose());
  const Eigen::VectorXd baseAngularReferenceVelocity = baseReferenceAngularVelocity_;
  baseOrientationTask_->refVel(baseAngularReferenceVelocity);

  std::map<std::string, std::vector<double>> targets;
  double keyboardYawCorrection = 0.0;
  // For a mixed translation+yaw command the steering angles encode the
  // requested instantaneous centre of curvature.  An absolute-heading
  // correction would continuously perturb that geometry whenever the pose
  // has a small phase offset, changing the radius and eventually exciting
  // rolling residuals.  Keep the correction for pure Q/E (where it removes a
  // heading bias) but leave the operator-selected radius untouched whenever
  // W/S/A/D is active.
  const bool keyboardPureYaw = std::abs(forward) < 1e-9 && std::abs(lateral) < 1e-9 && std::abs(yaw) > 1e-9;
  if(scenario_ == "keyboard" && closedLoopFeedback_ && keyboard_->running() && keyboardYawFeedbackGain_ > 0.0
     && keyboardPureYaw)
  {
    const Eigen::Vector3d measuredHeading = robot().posW().rotation().col(0);
    const double measuredYaw = std::atan2(measuredHeading.dot(terrainTangentY_),
                                          measuredHeading.dot(terrainTangentX_));
    if(std::isfinite(measuredYaw))
    {
      // Use the wrapped difference: baseYawTarget_ is intentionally unwrapped
      // for logging, while the shortest local correction remains continuous
      // through every +/-pi crossing.
      const double yawError = std::remainder(baseYawTarget_ - measuredYaw, 2.0 * 3.14159265358979323846);
      keyboardYawError_ = yawError;
      keyboardYawCorrection_ = keyboardYawFeedbackGain_ * yawError;
      keyboardYawCorrection = keyboardYawCorrection_;
    }
  }
  // The wheel targets above provide the position trajectory used by the
  // generic posture task. For interactive operation also provide the desired
  // wheel velocity as feed-forward: this keeps the simulator tracking the
  // command immediately instead of waiting for a growing position error to
  // generate acceleration through the posture gain.
  const bool keyboardFeedForward = scenario_ == "keyboard" && solver().backend() == Backend::Tasks;
  Eigen::VectorXd keyboardRefVel;
  if(keyboardFeedForward) { keyboardRefVel = Eigen::VectorXd::Zero(robot().mb().nrDof()); }
  if(!fourSteering_)
  {
    const double halfTrack = 0.5 * std::abs(wheelOffsets_[0].y() - wheelOffsets_[1].y());
    std::array<double, 2> rates = {(controlForward - halfTrack * yaw) / wheels_[0].radius,
                                   (controlForward + halfTrack * yaw) / wheels_[1].radius};
    if(scenario_ == "infeasible_soft") { rates[0] += 4.0; }
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      driveTargets_[i] += rates[i] * solver().dt();
      wheelReferenceRates_[i] = rates[i];
      targets[wheels_[i].driveJoint] = {driveTargets_[i]};
      if(keyboardFeedForward)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        keyboardRefVel(robot().mb().jointPosInDof(drive)) = rates[i];
      }
    }
  }
  else
  {
    // Keep the wheel requests local until every steering hinge has reached
    // its new target. If one wheel starts driving while another is still
    // slewing, the old/new contact directions are mixed and the chassis can
    // receive a large lateral impulse after a Q/E -> W/A/D transition.
    std::vector<double> requestedRates(wheels_.size(), 0.0);
    std::vector<double> requestedSteerings(wheels_.size(), 0.0);
    bool allSteeringAligned = true;
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      double steering = 0.0;
      double rate = controlForward / wheels_[i].radius;
      bool steeringAligned = true;
      if(scenario_ == "crab")
      {
        steering = steeringAngle_;
        rate = std::copysign(std::hypot(controlForward, controlLateral), controlForward) / wheels_[i].radius;
      }
      else if(scenario_ == "steering_rate")
      {
        steering = steeringAngle_ * std::sin(phase);
        rate = std::copysign(std::hypot(controlForward, controlLateral), controlForward) / wheels_[i].radius;
      }
      else if(scenario_ == "ackermann_left" || scenario_ == "ackermann_right" || scenario_ == "pure_yaw"
              || scenario_ == "keyboard")
      {
        // The Ranger Mini MuJoCo model's wheel-drive convention is opposite
        // to the controller's positive Z yaw convention: a positive wheel
        // spin/steering solution generated directly from +yaw rotates the
        // chassis clockwise. Keep the controller and GUI reference signs
        // unchanged, but mirror the yaw used for the wheel inverse
        // kinematics in the keyboard path so Q/E rotate in the requested
        // direction.
        const bool keyboardMixedTwist = scenario_ == "keyboard" && std::hypot(controlForward, controlLateral) > 1e-9
                                        && std::abs(yaw) > 1e-9;
        const double wheelYaw = scenario_ == "keyboard"
                                    ? -(keyboardMixedTwist ? 1.0 : keyboardYawScale_) * (yaw + keyboardYawCorrection)
                                    : yaw;
        const double pointX = controlForward - wheelYaw * wheelOffsets_[i].y();
        const double pointY = controlLateral + wheelYaw * wheelOffsets_[i].x();
        const double wheelPlanarSpeed = std::hypot(pointX, pointY);
        const auto steeringJoint = robot().jointIndexByName(wheels_[i].steeringJoint);
        const double measuredSteering = robot().mbc().q[steeringJoint][0];
        const bool keyboardStopped = scenario_ == "keyboard" && std::abs(forward) < 1e-12
                                     && std::abs(lateral) < 1e-12 && std::abs(yaw) < 1e-12;
        const bool pureLateralKeyboard = scenario_ == "keyboard" && std::abs(forward) < 1e-12
                                         && std::abs(yaw) < 1e-12 && std::abs(lateral) > 1e-6;
        if(keyboardStopped)
        {
          // Hold the last commanded steering pose while the drive wheels
          // brake. Following the measured pose here lets braking torque twist
          // the steering hinge and walk the command across the +/-90-degree
          // branch cut. Holding the command keeps a crabbed wheel on its
          // established representation while the pose latch above absorbs
          // the small safe-limit tracking offset.
          steering = steeringTargets_[i];
          rate = wheelPlanarSpeed < 1e-3
                     ? 0.0
                     : (pointX * std::cos(steering) + pointY * std::sin(steering))
                           / wheels_[i].radius;
        }
        else if(pureLateralKeyboard)
        {
          // A/D is a chassis-frame lateral command. Keep this special case
          // independent of the position-feedback correction: the feedback
          // may add a small longitudinal component, but it must not make the
          // steering representation switch across the +/-90 degree branch.
          const double steeringError = std::copysign(keyboardSteeringLimit, lateral) - measuredSteering;
          const double maxSteeringStep = keyboardSteeringRate_ * solver().dt();
          steering = measuredSteering + std::clamp(steeringError, -maxSteeringStep, maxSteeringStep);
          // Do not spin a wheel while it is still being steered. A sudden
          // +/-90 degree pose together with drive torque creates a large
          // lateral impulse in MuJoCo, which the contact mode estimator quite
          // correctly reports as a transient loss of rolling contact. Once
          // aligned, the wheel receives the requested lateral rolling speed.
          if(std::abs(steeringError) > 0.1)
          {
            rate = 0.0;
            steeringAligned = false;
          }
          else
          {
            // The wheel is intentionally held at a slightly sub-orthogonal
            // crab angle. Project the complete chassis velocity command on
            // that wheel direction; using hypot() here would turn a small
            // longitudinal feedback correction into a large alternating
            // lateral drive command.
            rate = (controlForward * std::cos(steering) + controlLateral * std::sin(steering))
                   / wheels_[i].radius;
          }
        }
        else if(wheelPlanarSpeed < 1e-3)
        {
          // atan2(0, 0) is mathematically undefined. At a keyboard stop the
          // feedback correction is intentionally tiny, so evaluating it
          // directly can make the steering target jump between +/- pi/2
          // based on floating-point noise. That excites the steering joints
          // and, on the physical simulator, can eventually make the QP
          // infeasible. Keep the measured steering angle while the wheel is
          // effectively stationary; non-zero commands still use the full
          // inverse-kinematics solution below.
          steering = robot().mbc().q[steeringJoint][0];
          rate = 0.0;
        }
        else
        {
          const double rawSteering = std::atan2(pointY, pointX);
          constexpr double steeringJointLimit = 0.5 * 3.14159265358979323846;
          // Leave a small margin to the MuJoCo hinge limit. Without it, a
          // pure lateral command can make the PD actuator cross the limit and
          // wrap the measured hinge angle to the opposite representation.
          constexpr double steeringLimit = steeringJointLimit - 1e-3;
          constexpr double steeringHysteresis = 0.1;
          // The Ranger steering joints are limited to +/-90 degrees. There
          // are two equivalent wheel poses separated by pi, with opposite
          // wheel spin. Select the valid pose closest to the measured angle
          // so a lateral command cannot chatter between +90 and -90 degrees
          // when tiny feedback noise moves atan2 across its branch cut.
          struct SteeringCandidate
          {
            double angle;
            double spinSign;
          };
          const std::array<SteeringCandidate, 3> candidates = {
              SteeringCandidate{rawSteering, 1.0},
              SteeringCandidate{rawSteering - 3.14159265358979323846, -1.0},
              SteeringCandidate{rawSteering + 3.14159265358979323846, -1.0}};
          bool candidateFound = false;
          double bestDistance = std::numeric_limits<double>::infinity();
          for(const auto & candidate : candidates)
          {
            if(std::abs(candidate.angle) > steeringJointLimit + 1e-12) { continue; }
            const double distance = std::abs(candidate.angle - measuredSteering);
            if(!candidateFound || distance < bestDistance)
            {
              candidateFound = true;
              bestDistance = distance;
              steering = std::clamp(candidate.angle, -steeringLimit, steeringLimit);
              rate = candidate.spinSign * wheelPlanarSpeed / wheels_[i].radius;
            }
          }
          if(!candidateFound)
          {
            // atan2 is only outside both representations for a numerical
            // excursion around the +/-pi boundary. Normalize it before the
            // final fallback so the command remains finite and bounded.
            steering = std::remainder(rawSteering, 3.14159265358979323846);
            steering = std::clamp(steering, -steeringLimit, steeringLimit);
            rate = wheelPlanarSpeed / wheels_[i].radius;
          }
          // At a nearly pure lateral command the two representations are
          // equally close while the measured joint is already at a limit.
          // Preserve that limit inside a small hysteresis band instead of
          // accepting a one-bit sign change that would reverse the wheels.
          if(std::abs(pointY) > 20.0 * std::abs(pointX) && std::abs(pointY) > 1e-3)
          {
            // For a nearly pure lateral command, explicitly keep the wheel
            // on the same side of the +/-90 degree limit. This handles the
            // exact atan2 branch point deterministically even when the
            // measured steering angle has a small simulator overshoot.
            steering = std::copysign(steeringLimit, pointY);
            rate = std::copysign(wheelPlanarSpeed / wheels_[i].radius, pointY);
          }
          else if(measuredSteering > steeringLimit - steeringHysteresis
                  && rawSteering > steeringJointLimit - steeringHysteresis
                  && rawSteering < steeringJointLimit + steeringHysteresis)
          {
            steering = steeringLimit;
            rate = wheelPlanarSpeed / wheels_[i].radius;
          }
          else if(measuredSteering < -steeringLimit + steeringHysteresis
                  && rawSteering < -steeringJointLimit + steeringHysteresis
                  && rawSteering > -steeringJointLimit - steeringHysteresis)
          {
            steering = -steeringLimit;
            rate = -wheelPlanarSpeed / wheels_[i].radius;
          }
        }
        if(keyboardMixedTwist) { rate *= keyboardMixedDriveScale_; }
        // Do not apply a yaw drive torque while the wheel is slewing to its
        // new steering representation. The transient lateral impulse can
        // otherwise trip the contact fallback before the steering hinges
        // have reached the rolling direction.
        if(scenario_ == "keyboard" && std::abs(steering - measuredSteering) > 0.1) { rate = 0.0; }
        // Choose the equivalent steering angle in the Ranger's +/-90 degree
        // joint range, reversing wheel spin when necessary.
        if(steering > 0.5 * 3.14159265358979323846)
        {
          steering -= 3.14159265358979323846;
          rate = -rate;
        }
        else if(steering < -0.5 * 3.14159265358979323846)
        {
          steering += 3.14159265358979323846;
          rate = -rate;
        }
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
      requestedRates[i] = rate;
      requestedSteerings[i] = steering;
      const double steeringError =
          steering - robot().mbc().q[robot().jointIndexByName(wheels_[i].steeringJoint)][0];
      if(scenario_ == "keyboard" && (!steeringAligned || std::abs(steeringError) > 0.1))
      {
        allSteeringAligned = false;
      }
    }

    if(scenario_ == "keyboard")
    {
      keyboardSteeringReady_ = allSteeringAligned;
      // Hold every drive wheel until all four steering hinges are aligned.
      // This barrier persists beyond the bounded command grace period, but
      // only for the actual slew time and therefore cannot leave a stale
      // yaw command active.
      if(!allSteeringAligned) { std::fill(requestedRates.begin(), requestedRates.end(), 0.0); }
    }
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      double rate = requestedRates[i];
      if(scenario_ == "keyboard")
      {
        // Once every hinge is aligned, resume with a bounded rate
        // acceleration so the new turning radius is entered smoothly.
        const double maxRateStep = keyboardDriveAcceleration_ * solver().dt();
        if(keyboardSteeringTransition || !allSteeringAligned)
        {
          rate = 0.0;
        }
        else
        {
          rate = wheelReferenceRates_[i] + std::clamp(rate - wheelReferenceRates_[i], -maxRateStep, maxRateStep);
        }
      }
      driveTargets_[i] += rate * solver().dt();
      wheelReferenceRates_[i] = rate;
      steeringTargets_[i] = requestedSteerings[i];
      targets[wheels_[i].driveJoint] = {driveTargets_[i]};
      targets[wheels_[i].steeringJoint] = {steeringTargets_[i]};
      if(keyboardFeedForward)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        keyboardRefVel(robot().mb().jointPosInDof(drive)) = rate;
      }
    }
  }
  postureTask->target(targets);
  if(keyboardFeedForward) { postureTask->refVel(keyboardRefVel); }
}

void MCRollingContactController::updateModes()
{
  const bool wasContactFallback = contactFallback_;
  const bool keyboardCaptureActive = scenario_ == "keyboard" && keyboard_ && keyboard_->running();
  if(scenario_ == "keyboard")
  {
    keyboardCommandTransitionGrace_ = std::max(0.0, keyboardCommandTransitionGrace_ - solver().dt());
  }
  else
  {
    keyboardCommandTransitionGrace_ = 0.0;
    keyboardCommandInitialized_ = false;
  }
  contactFallback_ = false;
  // A four-steering wheel must briefly be treated as sliding while its hinge
  // slews between two rolling directions. During that bounded interval the
  // measured steering velocity is non-zero, so keeping a hard rolling row
  // would ask the QP to enforce zero contact slip and zero hinge-induced
  // lateral motion simultaneously. The drive command is already held at zero
  // by updateReference(); removing the rolling rows for this interval keeps
  // the transition feasible without allowing a propulsion impulse.
  const bool keyboardSteeringTransition = scenario_ == "keyboard" && fourSteering_
                                          && keyboardCommandTransitionGrace_ > 0.0;
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
         && requested != mc_rbdyn::RollingContactMode::Detached && !keyboardCaptureActive)
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
      // During a keyboard radius change the wheel is deliberately braked while
      // its steering hinge slews to the new rolling direction.  MuJoCo can
      // report a one-cycle loss of normal force at that instant even though
      // the wheel is still geometrically supported.  Feeding that transient
      // into the hysteresis manager would promote the contact to Detached and
      // latch the global keyboard fallback.  Use the known steering-transition
      // state to give the manager a short, contact-safe recovery observation;
      // genuine detachments outside this bounded transition retain the normal
      // fail-safe path above.
      // The drive rate is slewed to zero over the same command-transition
      // grace interval, so do not require it to have already reached the
      // recovery threshold. Requiring that threshold here was circular: a
      // detached estimate prevented re-attachment while the still-slewing
      // wheel prevented the estimate from recovering.
      if(keyboardSteeringTransition && requested == mc_rbdyn::RollingContactMode::Rolling)
      {
        const auto & thresholds = modeManagers_[i].thresholds();
        observation.slipSpeed = 0.0;
        observation.rollingResidual = 0.0;
        observation.normalForce = std::max(observation.normalForce, thresholds.normalForceEnter + 1.0);
        observation.frictionMargin = std::max(observation.frictionMargin, thresholds.frictionMarginEnter + 1.0);
        observation.torqueMargin = std::max(observation.torqueMargin, thresholds.torqueMarginEnter + 1.0);
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
    const double solverActivation = keyboardSteeringTransition
                                            && state.estimated == mc_rbdyn::RollingContactMode::Rolling
                                        ? 0.0
                                        : appliedActivation;
    // During the hinge slew, temporarily remove this wheel's kinematic
    // rolling rows while retaining its unrestricted physical force cone in
    // the dynamics constraint. A zero rolling activation is preferable to a
    // Sliding dynamics mode here: the latter restricts contact force to one
    // stale cone generator and can make gravity/torque balance infeasible.
    appliedActivations_[i] = solverActivation;
    rolling_->mode(wheels_[i].name, state.estimated, solverActivation);
    dynamics_->mode(wheels_[i].name, state.estimated);
    // A keyboard command can change the instantaneous turning radius while a
    // steering hinge is still slewing.  The resulting short-lived slip is a
    // valid transition for the mode manager, not a reason to zero the
    // operator's command: doing that here used to latch contactFallback_ and
    // leave Q/E+W/S combinations permanently stuck even though the QP kept
    // solving successfully.  Detached contacts and pending hard promotion
    // remain fail-safe; a transient Sliding estimate is allowed to recover
    // while the keyboard trajectory continues.
    const bool keyboardContactRecovery = keyboardCaptureActive
                                         && (state.estimated == mc_rbdyn::RollingContactMode::Detached
                                             || state.estimated == mc_rbdyn::RollingContactMode::Sliding);
    contactFallback_ = contactFallback_
                       || (requested == mc_rbdyn::RollingContactMode::Rolling
                           && (state.estimated == mc_rbdyn::RollingContactMode::Detached
                               || hardPromotionPending_[i])
                           && !keyboardContactRecovery);
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
  referenceLateralSpeed_ = 0.0;
  referenceYawRate_ = 0.0;
  keyboardSteeringReady_ = false;
  guiForwardCommand_ = 0.0;
  guiLateralCommand_ = 0.0;
  guiYawCommand_ = 0.0;
  if(keyboard_) { keyboard_->clear(); }
  basePositionTarget_ = robot().posW().translation();
  baseReferenceVelocity_.setZero();
  baseTrackingVelocity_.setZero();
  baseReferenceAngularVelocity_.setZero();
  basePositionTask_->position(basePositionTarget_);
  basePositionTask_->refVel(Eigen::VectorXd::Zero(3));
  baseOrientationTask_->orientation(robot().posW().rotation());
  baseOrientationTask_->refVel(Eigen::VectorXd::Zero(3));
  std::fill(wheelReferenceRates_.begin(), wheelReferenceRates_.end(), 0.0);
  std::map<std::string, std::vector<double>> targets;
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
    driveTargets_[i] = robot().mbc().q[drive][0];
    wheelReferenceRates_[i] = 0.0;
    targets[wheels_[i].driveJoint] = {driveTargets_[i]};
    if(!wheels_[i].steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
      steeringTargets_[i] = robot().mbc().q[steering][0];
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

void MCRollingContactController::synchronizeMeasuredState()
{
  // mc_mujoco publishes joint encoders and the FloatingBase body sensor on the
  // control robot. Keep both the control and real MBCs in sync with those
  // measurements before evaluating tasks. In particular, no observer
  // pipeline is configured for the mc_mujoco keyboard profile, so leaving the
  // real MBC untouched would keep its floating base at the initial pose and
  // expose stale state through outputRealRobot()/ff_real.
  auto & measuredRobot = robot();
  auto & realMeasuredRobot = realRobot();
  const auto & encoders = measuredRobot.encoderValues();
  const auto & encoderVelocities = measuredRobot.encoderVelocities();
  const auto & refJointOrder = measuredRobot.refJointOrder();
  for(size_t refIndex = 0; refIndex < refJointOrder.size(); ++refIndex)
  {
    const auto & jointName = refJointOrder[refIndex];
    if(!measuredRobot.hasJoint(jointName) || refIndex >= encoders.size()) { continue; }
    const auto jointIndex = measuredRobot.jointIndexByName(jointName);
    if(measuredRobot.mb().joint(jointIndex).dof() != 1) { continue; }
    measuredRobot.mbc().q[jointIndex][0] = encoders[refIndex];
    if(refIndex < encoderVelocities.size()) { measuredRobot.mbc().alpha[jointIndex][0] = encoderVelocities[refIndex]; }
  }

  // The floating-base estimate is independent from the articulated encoder
  // packet. A simulator (or a hardware interface during startup) can publish
  // an IMU pose/velocity before every wheel encoder is valid; do not keep the
  // base at its old pose in that case. Joint encoders are validated separately
  // in the loop above.
  if(measuredRobot.mb().joint(0).type() == rbd::Joint::Type::Free && measuredRobot.hasBodySensor("FloatingBase"))
  {
    const auto & sensor = measuredRobot.bodySensor("FloatingBase");
    const auto & position = sensor.position();
    const auto & orientation = sensor.orientation();
    if(position.allFinite() && orientation.coeffs().allFinite() && orientation.norm() > 1e-12)
    {
      // BodySensor orientations are inertial-to-sensor. Robot::posW takes
      // that same world-to-base transform and internally writes the inverse
      // quaternion, matching MuJoCo's free-joint qpos convention.
      Eigen::Quaterniond worldToBody = orientation;
      worldToBody.normalize();
      measuredRobot.posW(sva::PTransformd(worldToBody, position));
      const auto & angularVelocity = sensor.angularVelocity();
      const auto & linearVelocity = sensor.linearVelocity();
      if(angularVelocity.allFinite() && linearVelocity.allFinite())
      {
        // BodySensor readings are expressed in the inertial frame, whereas
        // RBDyn's free-joint alpha stores angular/linear components in the
        // body frame. Passing the world linear velocity through unchanged
        // works while the chassis faces +X, but it appears as a growing
        // lateral rolling error as the robot turns and makes the mode manager
        // promote otherwise healthy wheels to Sliding/Detached.
        const Eigen::Matrix3d worldToBodyRotation = worldToBody.toRotationMatrix();
        const Eigen::Vector3d bodyAngularVelocity = worldToBodyRotation * angularVelocity;
        const Eigen::Vector3d bodyLinearVelocity = worldToBodyRotation * linearVelocity;
        measuredRobot.mbc().alpha[0] = {bodyAngularVelocity.x(), bodyAngularVelocity.y(), bodyAngularVelocity.z(),
                                        bodyLinearVelocity.x(),  bodyLinearVelocity.y(),  bodyLinearVelocity.z()};
      }
    }
  }
  measuredRobot.forwardKinematics();
  measuredRobot.forwardVelocity();

  // QPSolver keeps a separate realRobots() collection for sensor/estimator
  // state. The mc_mujoco adapter intentionally feeds sensors to the control
  // robot, so mirror the synchronized state explicitly. Do not copy the
  // controller's predicted/output state after this point: the copy happens
  // before MCController::run(), while both MBCs still represent the same
  // measurement.
  realMeasuredRobot.mbc().q = measuredRobot.mbc().q;
  realMeasuredRobot.mbc().alpha = measuredRobot.mbc().alpha;
  realMeasuredRobot.mbc().alphaD = measuredRobot.mbc().alphaD;
  realMeasuredRobot.mbc().jointTorque = measuredRobot.mbc().jointTorque;
  realMeasuredRobot.data()->encoderValues = measuredRobot.encoderValues();
  realMeasuredRobot.data()->encoderVelocities = measuredRobot.encoderVelocities();
  realMeasuredRobot.data()->jointTorques = measuredRobot.jointTorques();
  for(const auto & sensor : measuredRobot.bodySensors())
  {
    if(!realMeasuredRobot.hasBodySensor(sensor.name())) { continue; }
    auto & realSensor =
        realMeasuredRobot.data()->bodySensors[realMeasuredRobot.data()->bodySensorsIndex.at(sensor.name())];
    realSensor.position(sensor.position());
    realSensor.orientation(sensor.orientation());
    realSensor.linearVelocity(sensor.linearVelocity());
    realSensor.angularVelocity(sensor.angularVelocity());
    realSensor.linearAcceleration(sensor.linearAcceleration());
    realSensor.angularAcceleration(sensor.angularAcceleration());
  }
  realMeasuredRobot.forwardKinematics();
  realMeasuredRobot.forwardVelocity();
  realMeasuredRobot.forwardAcceleration();
}

bool MCRollingContactController::run()
{
  const auto begin = std::chrono::steady_clock::now();
  bool success = false;
  std::vector<double> measuredDrivePositions;
  try
  {
    if(closedLoopFeedback_)
    {
      synchronizeMeasuredState();
    }
    if(scenario_ == "keyboard") { updateKeyboardCommandTransition(); }
    measuredDrivePositions.resize(wheels_.size());
    std::vector<double> measuredSteeringPositions(wheels_.size(), 0.0);
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      measuredDrivePositions[i] = robot().mbc().q[drive][0];
      if(!wheels_[i].steeringJoint.empty())
      {
        const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
        measuredSteeringPositions[i] = robot().mbc().q[steering][0];
      }
    }
    updateModes();
    updateReference();
    success = MCController::run();
    if(success && scenario_ == "keyboard")
    {
      // mc_mujoco drives articulated joints from the controller's q/alpha
      // output. Keep the wheel position output at the measured position and
      // expose the keyboard velocity as alpha; otherwise the QP's alphaD
      // (which is only an acceleration result) is integrated from zero and
      // the simulator never receives the requested wheel speed.
      for(size_t i = 0; i < wheels_.size(); ++i)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        robot().mbc().q[drive][0] = measuredDrivePositions[i]
                                    + keyboardWheelPositionLookahead_ * wheelReferenceRates_[i];
        robot().mbc().alpha[drive][0] = wheelReferenceRates_[i];
        if(!wheels_[i].steeringJoint.empty())
        {
          const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
          robot().mbc().q[steering][0] = steeringTargets_[i];
          // mc_mujoco consumes alpha as the actuator velocity reference. A
          // position-only target is not sufficient for this joint (the
          // simulator would keep the steering angle at its old value), so
          // provide the bounded slew rate used to generate the target above.
          const double measuredSteering = measuredSteeringPositions[i];
          const double steeringVelocity =
              (steeringTargets_[i] - measuredSteering) / std::max(solver().dt(), 1e-12);
          const bool atSteeringLimit = std::abs(std::abs(steeringTargets_[i]) - keyboardSteeringLimit) < 1e-9;
          robot().mbc().alpha[steering][0] = atSteeringLimit
                                                 ? 0.0
                                                 : std::clamp(steeringVelocity, -keyboardSteeringRate_,
                                                              keyboardSteeringRate_);
        }
      }
      // Keep the free-joint q/alpha state measured. mc_mujoco only consumes
      // the actuated wheel entries from this MBC; it does not consume a
      // floating-base command. Writing the world-frame task references into
      // alpha[0] here would both corrupt the measured state (RBDyn stores the
      // free-joint velocity in body coordinates) and make a subsequent Q/E to
      // A/W transition appear as a large spurious chassis velocity.
      robot().forwardKinematics();
      robot().forwardVelocity();
    }
    if(success && closedLoopFeedback_ && robot().mb().joint(0).type() == rbd::Joint::Type::Free)
    {
      // The free joint is measured state, not an actuator command, in the
      // mc_mujoco adapter. MCController::run() integrates a one-step task
      // prediction into the control MBC; restore the measured root after
      // producing the wheel commands so the next cycle and ff_q never use
      // that prediction as if it were an observed chassis motion.
      robot().mbc().q[0] = realRobot().mbc().q[0];
      robot().mbc().alpha[0] = realRobot().mbc().alpha[0];
      robot().forwardKinematics();
      robot().forwardVelocity();
    }
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
