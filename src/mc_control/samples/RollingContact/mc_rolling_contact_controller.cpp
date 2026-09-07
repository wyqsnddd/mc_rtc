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
  // Ceiling on how fast the rolling-rate reference itself may change, so a step
  // in the commanded twist cannot ask the QP for the whole rate on the very
  // next cycle. It must stay well above the steering slope: the reference is a
  // projection onto the *measured* hinge heading, so a drive reference that
  // changes more slowly than the hinge swings keeps the wheels rolling in a
  // stale direction. This bounds every scenario, hence the name; the
  // keyboard-scoped spelling remains a deprecated alias.
  //
  // This was briefly widened to 100 rad/s^2, tuned only against the
  // kinematic ticker (no contact physics), where it halves the lateral slip
  // across a commanded-twist step (0.291 -> 0.079 m/s). Against closed-loop
  // MuJoCo contact physics that value saturates drive torque at the 35 Nm
  // limit and drives the min normal force to 0 on four-crab and
  // four-ackermann-left. A sweep against MuJoCo (four-crab, Tasks backend)
  // shows a sharp knee, not a gradual tradeoff: max drive torque is ~0.06 Nm
  // and min normal force ~184 N for driveAcceleration in [20, 40], then at 50
  // it jumps to 23 Nm / 0 N with the wheels 90% of the run in contact
  // fallback, and by 80-100 it is pinned at the 35 Nm limit with the wheels
  // detached 95%+ of the run. 20 rad/s^2 keeps a 2x margin below that knee and
  // is the value validated pre-rewrite (commit dff9cc26f1), where all four
  // MuJoCo cases and the ackermann_left/ackermann_right mirror symmetry pass.
  driveAcceleration_ = settings("driveAcceleration", 20.0);
  if(settings.has("keyboardDriveAcceleration"))
  {
    mc_rtc::log::warning("RollingContact keyboardDriveAcceleration is deprecated and now bounds every scenario's "
                         "drive reference; rename it to driveAcceleration");
    if(!settings.has("driveAcceleration")) { driveAcceleration_ = settings("keyboardDriveAcceleration", 20.0); }
  }
  if(!std::isfinite(driveAcceleration_) || driveAcceleration_ <= 0.0)
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>(
        "RollingContact driveAcceleration must be finite and positive");
  }
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
  if(fourSteering_)
  {
    // Take the hinge rate ceiling from the model rather than restating the
    // URDF here, so a model change cannot silently desynchronize it.
    const double modelSteeringRate = [this]()
    {
      double rate = std::numeric_limits<double>::infinity();
      for(const auto & wheel : wheels_)
      {
        const auto joint = robot().jointIndexByName(wheel.steeringJoint);
        rate = std::min({rate, std::abs(robot().vl()[joint][0]), std::abs(robot().vu()[joint][0])});
      }
      return rate;
    }();
    // A non-const default would select Configuration's write-into-reference
    // overload, which returns void.
    maxSteeringRate_ = settings("maxSteeringRate", modelSteeringRate);
    // Convergence time of a steering hinge towards its reference heading. The
    // largest possible re-heading is pi/2, so this value sets the initial
    // demand for that worst case at pi/2 / 0.15 = 10.5 rad/s, just above the
    // 8 rad/s hinge ceiling: the fastest first-order law that still leaves the
    // hinge saturated only for the first instants of the widest swing. Slowing
    // it down is measurably worse - at half this slope the lateral slip across
    // a commanded-twist step doubled, from 0.085 to 0.19 m/s.
    steeringTimeConstant_ = settings("steeringTimeConstant", 0.15);
    if(!std::isfinite(maxSteeringRate_) || maxSteeringRate_ <= 0.0 || !std::isfinite(steeringTimeConstant_)
       || steeringTimeConstant_ <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact maxSteeringRate and steeringTimeConstant must be finite and positive; the steering "
          "joint velocity limits supply the default rate");
    }
  }
  if(scenario_ == "keyboard")
  {
    const double keyboardLinearSpeed = settings("keyboardLinearSpeed", 0.3);
    const double keyboardAngularSpeed = settings("keyboardAngularSpeed", 0.5);
    const int keyboardPollIntervalMs = settings("keyboardPollIntervalMs", 10);
    const std::string keyboardExitKey = settings("keyboardExitKey", std::string{"x"});
    keyboardYawFeedbackGain_ = settings("keyboardYawFeedbackGain", 0.5);
    if(!std::isfinite(keyboardLinearSpeed) || keyboardLinearSpeed <= 0.0 || !std::isfinite(keyboardAngularSpeed)
       || keyboardAngularSpeed <= 0.0 || keyboardPollIntervalMs <= 0 || keyboardExitKey.size() != 1
       || !std::isfinite(keyboardYawFeedbackGain_) || keyboardYawFeedbackGain_ < 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact keyboard speeds and poll interval must be positive, keyboardExitKey one character, "
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
      // Outside the keyboard scenario the GUI is the operator's only entry
      // point, so it publishes the commanded twist directly. The keyboard poll
      // republishes the key state plus these offsets every cycle instead.
      mc_rtc::gui::NumberInput(
          "Forward velocity", [this]() { return guiForwardCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiForwardCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({value, commandedTwist_.y(), commandedTwist_.z()}); }
          }),
      mc_rtc::gui::NumberInput(
          "Lateral velocity", [this]() { return guiLateralCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiLateralCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({commandedTwist_.x(), value, commandedTwist_.z()}); }
          }),
      mc_rtc::gui::NumberInput(
          "Yaw velocity", [this]() { return guiYawCommand_; },
          [this](double value)
          {
            if(!std::isfinite(value)) { return; }
            guiYawCommand_ = value;
            if(scenario_ != "keyboard") { setCommandedTwist({commandedTwist_.x(), commandedTwist_.y(), value}); }
          }),
      mc_rtc::gui::Button(
          "Clear command", [this]()
          {
            guiForwardCommand_ = 0.0;
            guiLateralCommand_ = 0.0;
            guiYawCommand_ = 0.0;
            if(scenario_ != "keyboard") { setCommandedTwist(Eigen::Vector3d::Zero()); }
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
  if(fourSteering_)
  {
    options.trackRotatingRates = true;
    // A rate row predicts rate^+ = rate + dt * S * alphaD, so its coefficients
    // carry dt and its weight enters the objective multiplied by dt^2. Express
    // the defaults as the equivalent acceleration-task weight (1000, twice the
    // chassis tasks) divided by dt^2; the documented option keeps its own
    // "weight on the rate residual" meaning. With a plain weight of 200 the
    // rate rows contribute 200 * dt^2 = 5e-3 and the posture task, at 100,
    // silently keeps ownership of the wheel degrees of freedom.
    const double rateWeightScale = 1.0 / (dt * dt);
    options.rollingRateWeight = settings("rollingRateWeight", 1000.0 * rateWeightScale);
    options.steeringRateWeight = settings("steeringRateWeight", 1000.0 * rateWeightScale);
  }
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
    // The QP's steering-rate rows now own the hinge motion. Keep the posture
    // task on these joints only as a weak regulariser so it cannot fight them.
    const double steeringWeight = settings("steeringPostureWeight", 1.0);
    if(!std::isfinite(steeringStiffness) || steeringStiffness <= 0.0 || !std::isfinite(steeringWeight)
       || steeringWeight <= 0.0)
    {
      mc_rtc::log::error_and_throw<std::invalid_argument>(
          "RollingContact steeringStiffness and steeringPostureWeight must be finite and positive");
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
  steeringRateReferences_.resize(wheels_.size(), 0.0);
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
  logger().addLogEntry("RollingContact_commandedTwist", [this]() -> const Eigen::Vector3d & { return commandedTwist_; });
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
  // control state) so this remains a meaningful runtime model check. The raw
  // BodySensor packet itself only ever lands on robot(): the observer
  // pipeline consumes it to estimate realRobot()'s posW()/velW(), but does
  // not copy the sensor object into realRobot(), so read it from robot()
  // unconditionally and only switch the comparison target (the frame) with
  // closedLoopFeedback_.
  logger().addLogEntry("RollingContact_floating_base_chassis_error", [this]()
                       {
                         if(!robot().hasBodySensor("FloatingBase"))
                         {
                           return std::numeric_limits<double>::infinity();
                         }
                         const auto & sensor = robot().bodySensor("FloatingBase");
                         const sva::PTransformd sensorPose(sensor.orientation(), sensor.position());
                         const auto & measured = closedLoopFeedback_ ? realRobot() : robot();
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
    logger().addLogEntry("RollingContact_" + wheel.name + "_rollingRateRef",
                         [this, i]() { return wheelReferenceRates_[i]; });
    if(!wheel.steeringJoint.empty())
    {
      const auto steering = robot().jointIndexByName(wheel.steeringJoint);
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_position",
                           [this, steering]() { return robot().mbc().q[steering][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_target",
                           [this, i]() { return steeringTargets_[i]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steering_rate",
                           [this, steering]() { return robot().mbc().alpha[steering][0]; });
      logger().addLogEntry("RollingContact_" + wheel.name + "_steeringRateRef",
                           [this, i]() { return steeringRateReferences_[i]; });
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
                             // controlled robot. Use that raw stream directly for the
                             // measured steering angle rather than realRobot(): the
                             // Encoder observer only updates realRobot() when
                             // closedLoopFeedback_ has an ObserverPipelines block
                             // configured and run, and this log entry should stay
                             // meaningful regardless of that.
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
  commandedTwist_.setZero();
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
  steeringRateReferences_.assign(wheels_.size(), 0.0);
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

void MCRollingContactController::setCommandedTwist(const Eigen::Vector3d & twist)
{
  if(!twist.allFinite())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact commanded twist must be finite");
  }
  commandedTwist_ = twist;
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
    // RoboticsUtils uses right-positive lateral velocity and clockwise-positive
    // yaw. Convert to the robot convention: +Y left and +Z counter-clockwise.
    setCommandedTwist({command[0] + guiForwardCommand_, -command[1] + guiLateralCommand_,
                       -command[2] + guiYawCommand_});
  }
  else if(scenario_ != "hold")
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("Unknown RollingContact scenario: {}", scenario_);
  }
  if(fourSteering_)
  {
    // The keyboard, the GUI and the scripted scenarios all reach the wheel
    // references through setCommandedTwist(). A scripted scenario republishes
    // its twist every cycle; "hold" and "keyboard" publish none of their own,
    // so whatever the operator (or a test harness) last commanded stays in
    // effect.
    if(scenario_ != "hold" && scenario_ != "keyboard") { setCommandedTwist({forward, lateral, yaw}); }
    forward = commandedTwist_.x();
    lateral = commandedTwist_.y();
    yaw = commandedTwist_.z();
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
      else { baseYawTarget_ += yaw * solver().dt(); }
    }
    else
    {
      // Keep the keyboard yaw reference unwrapped in open loop. The heading
      // below is periodic, but retaining the accumulated value avoids a
      // discontinuous scalar target at +/-pi.
      baseYawTarget_ += yaw * solver().dt();
    }
  }
  else
  {
    baseYawTarget_ = std::remainder(baseYawTarget_ + yaw * solver().dt(), 2.0 * 3.14159265358979323846);
  }
  // The two branches above leave baseYawTarget_ in two different conventions,
  // so the heading below has to undo the difference.
  //
  // Everywhere except closed-loop keyboard, baseYawTarget_ integrates the
  // commanded yaw rate, so it already is the world heading.
  //
  // Closed-loop keyboard instead copies measuredYaw, which is read off
  // posW().rotation().col(0). Robot::posW().rotation() (and the MuJoCo
  // FloatingBase sensor) is the inertial-to-body map E_0_b, so for a chassis
  // yawed by psi in the world E_0_b = Rz(psi)^T and its first column is
  // (cos psi, -sin psi). measuredYaw is therefore -psi, the negation of the
  // world direction the chassis' +X axis actually travels, and the sign has
  // to be flipped back here. Without the flip a W+Q/W+E command follows the
  // mirrored circle and accumulates a metre-scale position-task error even
  // though the measured body-forward speed is correct.
  //
  // KeyboardClosedLoopYawTargetMirrorsTheMeasuredWorldHeading pins this.
  const double trajectoryHeadingYaw = scenario_ == "keyboard" && closedLoopFeedback_ ? -baseYawTarget_ : baseYawTarget_;
  const Eigen::Vector3d heading = std::cos(trajectoryHeadingYaw) * terrainTangentX_
                                  + std::sin(trajectoryHeadingYaw) * terrainTangentY_;
  const Eigen::Vector3d side = -std::sin(trajectoryHeadingYaw) * terrainTangentX_
                               + std::cos(trajectoryHeadingYaw) * terrainTangentY_;
  baseReferenceVelocity_ = forward * heading + lateral * side;
  baseReferenceAngularVelocity_ = yaw * terrainNormal_;
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
  basePositionTarget_ += solver().dt() * (forward * heading + lateral * side);
  // Feed the measured position error back into the chassis task. The
  // integrated target remains the user's requested trajectory, while this
  // correction compensates for the small velocity error introduced by the
  // simulator's wheel dynamics and makes the absolute target asymptotically
  // trackable instead of leaving a permanent position offset. It deliberately
  // does not reach the wheel references: those follow the commanded twist so a
  // constant command keeps a constant instantaneous centre of curvature.
  baseTrackingVelocity_ = baseReferenceVelocity_;
  if(closedLoopFeedback_ && scenario_ == "keyboard" && !contactFallback_ && positionFeedbackGain_ > 0.0)
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
    std::array<double, 2> rates = {(forward - halfTrack * yaw) / wheels_[0].radius,
                                   (forward + halfTrack * yaw) / wheels_[1].radius};
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
    // Every steering wheel reference is the exact inverse of the expanded
    // four-steering rolling rows for the commanded chassis twist. The QP then
    // tracks both rates through its soft rate rows, so there is no analytic
    // hinge IK, no branch-cut bookkeeping and no drive gating left here.
    const double dt = solver().dt();
    const Eigen::Vector3d twist(forward, lateral, yaw + keyboardYawCorrection);
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      mc_rbdyn::PlanarWheel planar;
      planar.offset = wheelOffsets_[i];
      planar.radius = wheels_[i].radius;
      planar.spinSign = wheels_[i].spinSign;

      const auto steeringJoint = robot().jointIndexByName(wheels_[i].steeringJoint);
      const double measuredSteering = robot().mbc().q[steeringJoint][0];
      planar.steeringAngle = measuredSteering;
      planar.steeringRate = robot().mbc().alpha[steeringJoint][0];

      // Wheel-centre velocity of the commanded planar twist, the same quantity
      // steeringWheelReference() inverts.
      const Eigen::Vector2d point(twist.x() - twist.z() * planar.offset.y(),
                                  twist.y() + twist.z() * planar.offset.x());
      auto reference = mc_rbdyn::steeringWheelReference(planar, twist, measuredSteering);

      // First-order convergence to the reference heading, saturated by the
      // hinge velocity limit. This is deltaDot^ref of the four-steering QP.
      const double steeringRate = std::clamp((reference.steeringAngle - measuredSteering) / steeringTimeConstant_,
                                             -maxSteeringRate_, maxSteeringRate_);

      // The rolling rate must stay consistent with the heading the wheel
      // actually has during the hinge slew, not with the heading it is
      // converging to: a rigid wheel can only roll along its current line, so
      // projecting the commanded wheel-centre velocity on the reference
      // heading would ask the QP to drive the chassis in a stale direction.
      // Both expressions coincide once the hinge has converged, which is what
      // makes the drive-gating of the previous open-loop layer unnecessary.
      double rollingRate =
          reference.commanded
              ? (std::cos(measuredSteering) * point.x() + std::sin(measuredSteering) * point.y())
                    / (wheels_[i].radius * wheels_[i].spinSign)
              : reference.rollingRate;
      if(scenario_ == "incompatible" && i == 0)
      {
        // Deliberately ask the first wheel for a rate no rigid rolling solution
        // can satisfy, so the soft longitudinal mode is exercised end to end.
        rollingRate += 4.0;
      }
      // Bound how fast the reference itself may change. A step in the commanded
      // twist otherwise asks the QP for the whole rate on the next cycle, which
      // saturates the contact friction cone for one tick.
      const double maxRateStep = driveAcceleration_ * dt;
      rollingRate = wheelReferenceRates_[i]
                    + std::clamp(rollingRate - wheelReferenceRates_[i], -maxRateStep, maxRateStep);

      steeringTargets_[i] = measuredSteering + steeringRate * dt;
      steeringRateReferences_[i] = steeringRate;
      wheelReferenceRates_[i] = rollingRate;
      driveTargets_[i] += rollingRate * dt;
      targets[wheels_[i].driveJoint] = {driveTargets_[i]};
      targets[wheels_[i].steeringJoint] = {steeringTargets_[i]};
      if(keyboardFeedForward)
      {
        const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
        keyboardRefVel(robot().mb().jointPosInDof(drive)) = rollingRate;
        keyboardRefVel(robot().mb().jointPosInDof(static_cast<int>(steeringJoint))) = steeringRate;
      }

      // The QP owns both rates from here on. It must receive the same rolling
      // rate that the posture target, the log and the simulator output carry:
      // reference.rollingRate is the pre-projection, pre-slew value.
      rolling_->rotatingRateReference(wheels_[i].name, rollingRate, steeringRate);
    }
  }
  postureTask->target(targets);
  if(keyboardFeedForward) { postureTask->refVel(keyboardRefVel); }
}

void MCRollingContactController::updateModes()
{
  const bool wasContactFallback = contactFallback_;
  const bool keyboardCaptureActive = scenario_ == "keyboard" && keyboard_ && keyboard_->running();
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
    appliedActivations_[i] = appliedActivation;
    rolling_->mode(wheels_[i].name, state.estimated, appliedActivation);
    dynamics_->mode(wheels_[i].name, state.estimated);
    // A commanded twist change can change the instantaneous turning radius
    // while a steering hinge is still slewing.  The resulting short-lived
    // slip is a valid transition for the mode manager, not a reason to zero
    // the command: doing that here used to latch contactFallback_ and leave
    // keyboard Q/E+W/S combinations permanently stuck even though the QP
    // kept solving successfully.  Detached contacts and pending hard
    // promotion remain fail-safe; a transient Sliding/Detached estimate is
    // allowed to recover while the hinge that caused it is still moving.
    //
    // The transient is a property of the hinge motion, not of who issued the
    // twist: a scripted command that swings the reference steering angle
    // (e.g. crab to ackermann in one step) produces exactly the same
    // short-lived slip a keyboard operator triggers, so the recovery
    // allowance is keyed on the hinge still slewing rather than on
    // scenario_ == "keyboard". steeringRateReferences_[i] is only non-zero on
    // the four-steering wheel loop below, and lags by one cycle here because
    // updateModes() runs before this cycle's updateReference().
    const bool hingeSlewing = fourSteering_ && std::abs(steeringRateReferences_[i]) > 1e-6;
    const bool keyboardContactRecovery = (keyboardCaptureActive || hingeSlewing)
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
  guiForwardCommand_ = 0.0;
  guiLateralCommand_ = 0.0;
  guiYawCommand_ = 0.0;
  commandedTwist_.setZero();
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
      steeringRateReferences_[i] = 0.0;
      targets[wheels_[i].steeringJoint] = {robot().mbc().q[steering][0]};
    }
    rolling_->rotatingRateReference(wheels_[i].name, 0.0, 0.0);
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

void MCRollingContactController::syncControlRobotFromSensors()
{
  // mc_mujoco publishes joint encoders and the FloatingBase body sensor on the
  // control robot only. Feed them into the control MBC so the QP built this
  // tick sees the measured state (closed-loop feedback), matching how a real
  // hardware interface's encoder/IMU packet would be consumed. This is
  // independent from state observation: realRobot() is estimated by the
  // Encoder/BodySensor observer pipeline (see ObserverPipelines in the
  // controller configuration), which runs before MCController::run() and
  // reads these same raw sensor values off the control robot, so it does not
  // need (and must not be overwritten by) a copy from here.
  auto & measuredRobot = robot();
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
      syncControlRobotFromSensors();
    }
    measuredDrivePositions.resize(wheels_.size());
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      const auto drive = robot().jointIndexByName(wheels_[i].driveJoint);
      measuredDrivePositions[i] = robot().mbc().q[drive][0];
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
        robot().mbc().q[drive][0] = measuredDrivePositions[i];
        robot().mbc().alpha[drive][0] = wheelReferenceRates_[i];
        if(!wheels_[i].steeringJoint.empty())
        {
          const auto steering = robot().jointIndexByName(wheels_[i].steeringJoint);
          robot().mbc().q[steering][0] = steeringTargets_[i];
          // mc_mujoco consumes alpha as the actuator velocity reference. A
          // position-only target is not sufficient for this joint (the
          // simulator would keep the steering angle at its old value), so
          // publish the same bounded rate reference the QP was given.
          robot().mbc().alpha[steering][0] = steeringRateReferences_[i];
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
      // that prediction as if it were an observed chassis motion. realRobot()
      // is the observer pipeline's estimate for this tick: the Encoder and
      // BodySensor observers run (via runObserverPipelines()) before this
      // run() is called, so realRobot().mbc().q[0]/alpha[0] already reflect
      // the same sensor packet syncControlRobotFromSensors() just wrote into
      // robot(). A caller that drives this controller without running the
      // configured ObserverPipelines every cycle will restore a stale free
      // joint here.
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
