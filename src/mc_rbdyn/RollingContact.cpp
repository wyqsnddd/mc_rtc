/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_rbdyn/RollingContact.h>

#include <mc_rbdyn/Robot.h>

#include <Eigen/Geometry>

#include <RBDyn/Jacobian.h>
#include <RBDyn/MultiBodyConfig.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace mc_rbdyn
{

namespace
{

constexpr double geometryEpsilon = 1e-12;

bool finite(double value)
{
  return std::isfinite(value);
}

template<typename Derived>
void requireFinite(const Eigen::MatrixBase<Derived> & value, const char * name)
{
  if(!value.allFinite()) { throw std::invalid_argument(std::string("Rolling contact ") + name + " is not finite"); }
}

void requireSpinSign(double spinSign)
{
  if(!finite(spinSign) || std::abs(std::abs(spinSign) - 1.0) > geometryEpsilon)
  {
    throw std::invalid_argument("Rolling contact spinSign must be +1 or -1");
  }
}

std::string lowercase(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value;
}

RollingContactDescription validatedDescription(RollingContactDescription description)
{
  description.validate();
  return description;
}

} // namespace

const char * to_string(RollingContactMode mode) noexcept
{
  switch(mode)
  {
    case RollingContactMode::Fixed:
      return "fixed";
    case RollingContactMode::Rolling:
      return "rolling";
    case RollingContactMode::Sliding:
      return "sliding";
    case RollingContactMode::Detached:
      return "detached";
  }
  return "unknown";
}

RollingContactMode rollingContactModeFromString(const std::string & mode)
{
  const auto value = lowercase(mode);
  if(value == "fixed") { return RollingContactMode::Fixed; }
  if(value == "rolling") { return RollingContactMode::Rolling; }
  if(value == "sliding") { return RollingContactMode::Sliding; }
  if(value == "detached") { return RollingContactMode::Detached; }
  throw std::invalid_argument("Unknown rolling contact mode: " + mode);
}

void RollingContactDescription::validate() const
{
  if(name.empty()) { throw std::invalid_argument("Rolling contact name cannot be empty"); }
  if(carrierFrame.empty()) { throw std::invalid_argument("Rolling contact carrierFrame cannot be empty"); }
  if(wheelBody.empty()) { throw std::invalid_argument("Rolling contact wheelBody cannot be empty"); }
  if(driveJoint.empty()) { throw std::invalid_argument("Rolling contact driveJoint cannot be empty"); }
  if(!finite(radius) || radius <= 0.0) { throw std::invalid_argument("Rolling contact radius must be positive"); }
  if(!finite(width) || width < 0.0) { throw std::invalid_argument("Rolling contact width cannot be negative"); }
  if(!finite(friction) || friction < 0.0)
  {
    throw std::invalid_argument("Rolling contact friction cannot be negative");
  }
  requireSpinSign(spinSign);
  if(!finite(activation) || activation < 0.0 || activation > 1.0)
  {
    throw std::invalid_argument("Rolling contact activation must be in [0, 1]");
  }
}

void RollingContactModeThresholds::validate() const
{
  const std::array<double, 13> values = {slipEnter,
                                         slipExit,
                                         residualEnter,
                                         residualExit,
                                         normalForceEnter,
                                         normalForceExit,
                                         frictionMarginEnter,
                                         frictionMarginExit,
                                         torqueMarginEnter,
                                         torqueMarginExit,
                                         minimumDwell,
                                         transitionTime,
                                         filterTimeConstant};
  if(!std::all_of(values.begin(), values.end(), [](double value) { return finite(value); }))
  {
    throw std::invalid_argument("Rolling contact mode thresholds must be finite");
  }
  if(slipExit < 0.0 || slipEnter <= slipExit || residualExit < 0.0 || residualEnter <= residualExit)
  {
    throw std::invalid_argument("Rolling contact slip/residual exit thresholds must be non-negative and below entry");
  }
  if(normalForceExit < 0.0 || normalForceEnter <= normalForceExit)
  {
    throw std::invalid_argument("Rolling contact normal-force entry threshold must exceed the exit threshold");
  }
  if(frictionMarginEnter <= frictionMarginExit || torqueMarginEnter <= torqueMarginExit)
  {
    throw std::invalid_argument("Rolling contact feasibility entry margins must exceed exit margins");
  }
  if(minimumDwell < 0.0 || transitionTime < 0.0 || filterTimeConstant < 0.0)
  {
    throw std::invalid_argument("Rolling contact mode times cannot be negative");
  }
}

RollingContactModeManager::RollingContactModeManager(RollingContactModeThresholds thresholds)
: thresholds_(std::move(thresholds))
{
  thresholds_.validate();
  reset();
}

void RollingContactModeManager::requestedMode(RollingContactMode mode) noexcept
{
  state_.requested = mode;
}

RollingContactMode RollingContactModeManager::requestedMode() const noexcept
{
  return state_.requested;
}

const RollingContactModeState & RollingContactModeManager::state() const noexcept
{
  return state_;
}

const RollingContactModeThresholds & RollingContactModeManager::thresholds() const noexcept
{
  return thresholds_;
}

void RollingContactModeManager::reset(RollingContactMode requested,
                                      RollingContactMode estimated,
                                      double activation)
{
  if(!finite(activation) || activation < 0.0 || activation > 1.0)
  {
    throw std::invalid_argument("Rolling contact mode activation must be in [0, 1]");
  }
  state_ = {};
  state_.requested = requested;
  state_.estimated = estimated;
  state_.activation = activation;
  pendingMode_ = estimated;
  pendingTime_ = 0.0;
}

RollingContactMode RollingContactModeManager::desiredMode(const RollingContactModeObservation & observation) const
{
  if(state_.requested == RollingContactMode::Detached) { return RollingContactMode::Detached; }
  if(observation.normalForce < thresholds_.normalForceExit) { return RollingContactMode::Detached; }

  const bool slipping = state_.filteredSlip > thresholds_.slipEnter
                        || std::abs(observation.rollingResidual) > thresholds_.residualEnter
                        || observation.frictionMargin < thresholds_.frictionMarginExit
                        || observation.torqueMargin < thresholds_.torqueMarginExit;
  const bool recovered = state_.filteredSlip < thresholds_.slipExit
                         && std::abs(observation.rollingResidual) < thresholds_.residualExit
                         && observation.normalForce > thresholds_.normalForceEnter
                         && observation.frictionMargin > thresholds_.frictionMarginEnter
                         && observation.torqueMargin > thresholds_.torqueMarginEnter;

  if(state_.requested == RollingContactMode::Sliding) { return RollingContactMode::Sliding; }
  // A detached contact must first satisfy every recovery threshold before it
  // is allowed to carry force again. Re-contact impact slip must not route it
  // through the kinetic-friction mode while the contact is still unsettled.
  if(state_.estimated == RollingContactMode::Detached)
  {
    return recovered ? state_.requested : RollingContactMode::Detached;
  }
  if(state_.estimated == RollingContactMode::Sliding && !recovered) { return RollingContactMode::Sliding; }
  if(slipping) { return RollingContactMode::Sliding; }
  return state_.requested;
}

void RollingContactModeManager::transitionTo(RollingContactMode mode)
{
  state_.estimated = mode;
  state_.dwellTime = 0.0;
  pendingMode_ = mode;
  pendingTime_ = 0.0;
}

const RollingContactModeState & RollingContactModeManager::update(const RollingContactModeObservation & observation,
                                                                  double dt)
{
  if(!finite(dt) || dt <= 0.0) { throw std::invalid_argument("Rolling contact mode timestep must be positive"); }
  const bool finiteObservation = finite(observation.slipSpeed) && observation.slipSpeed >= 0.0
                                 && finite(observation.rollingResidual) && finite(observation.normalForce)
                                 && finite(observation.frictionMargin) && finite(observation.torqueMargin);
  state_.measurementValid = observation.valid && finiteObservation;
  state_.dwellTime += dt;
  if(!state_.measurementValid)
  {
    state_.invalidReason = observation.valid ? "non-finite-or-negative-measurement" : "measurement-invalid";
    state_.filteredSlip = 0.0;
    transitionTo(RollingContactMode::Detached);
  }
  else
  {
    state_.invalidReason.clear();
    const double filter = thresholds_.filterTimeConstant == 0.0
                              ? 1.0
                              : std::min(1.0, dt / (thresholds_.filterTimeConstant + dt));
    state_.filteredSlip += filter * (observation.slipSpeed - state_.filteredSlip);
    const RollingContactMode desired = desiredMode(observation);
    const bool immediate = desired == RollingContactMode::Detached
                           && (state_.requested == RollingContactMode::Detached
                               || observation.normalForce < thresholds_.normalForceExit);
    if(desired == state_.estimated)
    {
      pendingMode_ = desired;
      pendingTime_ = 0.0;
    }
    else if(immediate)
    {
      transitionTo(desired);
    }
    else
    {
      if(pendingMode_ != desired)
      {
        pendingMode_ = desired;
        pendingTime_ = 0.0;
      }
      pendingTime_ += dt;
      if(pendingTime_ + 1e-12 >= thresholds_.minimumDwell) { transitionTo(desired); }
    }
  }

  const double target = state_.estimated == RollingContactMode::Detached ? 0.0 : 1.0;
  if(thresholds_.transitionTime == 0.0) { state_.activation = target; }
  else
  {
    const double step = dt / thresholds_.transitionTime;
    state_.activation += std::max(-step, std::min(step, target - state_.activation));
  }
  return state_;
}

RollingContactGeometry::RollingContactGeometry(int nrDof) : nrDof_(nrDof)
{
  if(nrDof <= 0) { throw std::invalid_argument("RollingContactGeometry requires a positive number of DoF"); }
  result_.rollingMatrix.setZero(3, nrDof_);
}

const RollingContactGeometryResult & RollingContactGeometry::update(const RollingContactKinematics & input)
{
  if(input.carrierJacobian.rows() != 3 || input.carrierJacobian.cols() != nrDof_)
  {
    throw std::invalid_argument("Rolling contact carrierJacobian must be 3 x nrDof");
  }
  if(input.wheelSelector.rows() != 1 || input.wheelSelector.cols() != nrDof_)
  {
    throw std::invalid_argument("Rolling contact wheelSelector must be 1 x nrDof");
  }
  if(input.steeringSelector.size() != 0
     && (input.steeringSelector.rows() != 1 || input.steeringSelector.cols() != nrDof_))
  {
    throw std::invalid_argument("Rolling contact steeringSelector must be empty or 1 x nrDof");
  }
  if(input.steeringSelector.size() != 0) { requireFinite(input.steeringSelector, "steeringSelector"); }
  if(input.generalizedVelocity.size() != nrDof_)
  {
    throw std::invalid_argument("Rolling contact generalizedVelocity size does not match nrDof");
  }
  requireFinite(input.carrierCenter, "carrierCenter");
  requireFinite(input.wheelAxle, "wheelAxle");
  requireFinite(input.wheelAxleRate, "wheelAxleRate");
  requireFinite(input.terrainNormal, "terrainNormal");
  requireFinite(input.carrierNormalAcceleration, "carrierNormalAcceleration");
  requireFinite(Eigen::Map<const Eigen::VectorXd>(input.carrierJacobian.data(), input.carrierJacobian.size()),
                "carrierJacobian");
  requireFinite(input.wheelSelector, "wheelSelector");
  requireFinite(input.generalizedVelocity, "generalizedVelocity");
  if(!finite(input.radius) || input.radius <= 0.0)
  {
    throw std::invalid_argument("Rolling contact radius must be positive");
  }
  if(!finite(input.width) || input.width < 0.0)
  {
    throw std::invalid_argument("Rolling contact width cannot be negative");
  }
  requireSpinSign(input.spinSign);
  if(!finite(input.steeringRate) || !finite(input.velocityGain))
  {
    throw std::invalid_argument("Rolling contact steeringRate and velocityGain must be finite");
  }

  const double normalNorm = input.terrainNormal.norm();
  if(normalNorm <= geometryEpsilon) { throw std::invalid_argument("Rolling contact terrainNormal is degenerate"); }
  result_.normalDirection = input.terrainNormal / normalNorm;

  const Eigen::Vector3d tangentAxle =
      input.wheelAxle - result_.normalDirection * result_.normalDirection.dot(input.wheelAxle);
  const double axleNorm = tangentAxle.norm();
  if(axleNorm <= geometryEpsilon)
  {
    throw std::invalid_argument("Rolling contact wheel axle is parallel to the terrain normal");
  }
  result_.lateralDirection = tangentAxle / axleNorm;
  result_.rollingDirection = result_.lateralDirection.cross(result_.normalDirection);
  result_.rollingDirection.normalize();
  // Recompute l from the normative l = n x t relation to suppress projection round-off.
  result_.lateralDirection = result_.normalDirection.cross(result_.rollingDirection);
  result_.lateralDirection.normalize();

  const Eigen::Vector3d tangentAxleRate =
      input.wheelAxleRate - result_.normalDirection * result_.normalDirection.dot(input.wheelAxleRate);
  const Eigen::Vector3d projectedLateralRate =
      (Eigen::Matrix3d::Identity() - result_.lateralDirection * result_.lateralDirection.transpose())
      * tangentAxleRate / axleNorm;
  result_.lateralDirectionRate = projectedLateralRate - input.steeringRate * result_.rollingDirection;
  result_.rollingDirectionRate = result_.lateralDirectionRate.cross(result_.normalDirection);
  result_.frame.col(0) = result_.rollingDirection;
  result_.frame.col(1) = result_.lateralDirection;
  result_.frame.col(2) = result_.normalDirection;
  result_.tangentProjector.noalias() =
      Eigen::Matrix3d::Identity() - result_.normalDirection * result_.normalDirection.transpose();

  result_.carrierCenter = input.carrierCenter;
  result_.contactPoint = input.carrierCenter - input.radius * result_.normalDirection;
  result_.lineStart = result_.contactPoint - 0.5 * input.width * result_.lateralDirection;
  result_.lineEnd = result_.contactPoint + 0.5 * input.width * result_.lateralDirection;

  result_.carrierVelocity.noalias() = input.carrierJacobian * input.generalizedVelocity;
  result_.tangentialCarrierVelocity.noalias() = result_.tangentProjector * result_.carrierVelocity;

  result_.rollingMatrix.row(0).noalias() = result_.rollingDirection.transpose() * input.carrierJacobian;
  result_.rollingMatrix.row(0) -= input.radius * input.spinSign * input.wheelSelector;
  result_.measuredRollingRate = input.wheelSelector.dot(input.generalizedVelocity);
  result_.measuredSteeringRate =
      input.steeringSelector.size() == 0 ? 0.0 : input.steeringSelector.dot(input.generalizedVelocity);
  result_.rollingMatrix.row(1).noalias() = result_.lateralDirection.transpose() * input.carrierJacobian;
  result_.rollingMatrix.row(2).noalias() = result_.normalDirection.transpose() * input.carrierJacobian;
  result_.velocityResidual.noalias() = result_.rollingMatrix * input.generalizedVelocity;
  result_.slipVelocity = result_.velocityResidual.x() * result_.rollingDirection
                         + result_.velocityResidual.y() * result_.lateralDirection;

  result_.accelerationBias.x() = result_.rollingDirection.dot(input.carrierNormalAcceleration)
                                 + result_.rollingDirectionRate.dot(result_.carrierVelocity);
  result_.accelerationBias.y() = result_.lateralDirection.dot(input.carrierNormalAcceleration)
                                 + result_.lateralDirectionRate.dot(result_.carrierVelocity);
  result_.accelerationBias.z() = result_.normalDirection.dot(input.carrierNormalAcceleration);
  result_.rhs = -result_.accelerationBias - input.velocityGain * result_.velocityResidual;

  result_.orthonormalError = (result_.frame.transpose() * result_.frame - Eigen::Matrix3d::Identity()).norm();
  result_.rightHandedError =
      (result_.rollingDirection.cross(result_.lateralDirection) - result_.normalDirection).norm();
  if(!result_.frame.allFinite() || !result_.rollingMatrix.allFinite() || !result_.rhs.allFinite())
  {
    throw std::runtime_error("Rolling contact geometry update produced a non-finite result");
  }
  return result_;
}

struct RollingContactRobotGeometry::Impl
{
  Impl(const Robot & robot, RollingContactDescription descriptionIn)
  : description(validatedDescription(std::move(descriptionIn))), robotAddress(&robot),
    carrierBody(robot.frame(description.carrierFrame).body()),
    wheelBodyIndex(robot.bodyIndexByName(description.wheelBody)),
    driveJointIndex(robot.jointIndexByName(description.driveJoint)),
    driveDof(robot.mb().jointPosInDof(static_cast<int>(driveJointIndex))),
    carrierJacobian(robot.mb(), carrierBody, robot.frame(description.carrierFrame).X_b_f().translation()),
    fullJacobian(Eigen::MatrixXd::Zero(6, robot.mb().nrDof())), geometry(robot.mb().nrDof())
  {
    const auto & driveJoint = robot.mb().joint(driveJointIndex);
    if(driveJoint.dof() != 1 || driveJoint.type() != rbd::Joint::Rev)
    {
      throw std::invalid_argument("Rolling contact driveJoint must be a one-DoF revolute joint: "
                                  + description.driveJoint);
    }
    if(!description.steeringJoint.empty())
    {
      const auto steeringIndex = robot.jointIndexByName(description.steeringJoint);
      const auto & steeringJoint = robot.mb().joint(steeringIndex);
      if(steeringJoint.dof() != 1 || steeringJoint.type() != rbd::Joint::Rev)
      {
        throw std::invalid_argument("Rolling contact steeringJoint must be a one-DoF revolute joint: "
                                    + description.steeringJoint);
      }
      steeringDof = robot.mb().jointPosInDof(static_cast<int>(steeringIndex));
    }

    input.carrierJacobian.setZero(3, robot.mb().nrDof());
    input.wheelSelector.setZero(robot.mb().nrDof());
    input.wheelSelector(driveDof) = 1.0;
    if(steeringDof >= 0)
    {
      input.steeringSelector.setZero(robot.mb().nrDof());
      input.steeringSelector(steeringDof) = 1.0;
    }
    input.generalizedVelocity.setZero(robot.mb().nrDof());
    input.radius = description.radius;
    input.width = description.width;
    input.spinSign = description.spinSign;
  }

  void checkRobot(const Robot & robot) const
  {
    if(&robot != robotAddress)
    {
      throw std::invalid_argument("RollingContactRobotGeometry must be updated with the Robot used at construction");
    }
  }

  RollingContactDescription description;
  const Robot * robotAddress;
  std::string carrierBody;
  unsigned int wheelBodyIndex;
  unsigned int driveJointIndex;
  int driveDof;
  int steeringDof = -1;
  rbd::Jacobian carrierJacobian;
  Eigen::MatrixXd fullJacobian;
  RollingContactKinematics input;
  RollingContactGeometry geometry;
};

RollingContactRobotGeometry::RollingContactRobotGeometry(const Robot & robot, RollingContactDescription description)
: impl_(std::make_unique<Impl>(robot, std::move(description)))
{
}

RollingContactRobotGeometry::~RollingContactRobotGeometry() = default;
RollingContactRobotGeometry::RollingContactRobotGeometry(RollingContactRobotGeometry &&) noexcept = default;
RollingContactRobotGeometry & RollingContactRobotGeometry::operator=(RollingContactRobotGeometry &&) noexcept = default;

const RollingContactGeometryResult & RollingContactRobotGeometry::update(const Robot & robot,
                                                                         const Eigen::Vector3d & terrainNormal,
                                                                         double velocityGain)
{
  impl_->checkRobot(robot);
  const auto & frame = robot.frame(impl_->description.carrierFrame);
  const auto & partialJacobian = impl_->carrierJacobian.jacobian(robot.mb(), robot.mbc());
  impl_->carrierJacobian.fullJacobian(robot.mb(), partialJacobian, impl_->fullJacobian);
  impl_->input.carrierJacobian = impl_->fullJacobian.bottomRows<3>();
  impl_->input.carrierCenter = frame.position().translation();
  impl_->input.carrierNormalAcceleration =
      impl_->carrierJacobian.normalAcceleration(robot.mb(), robot.mbc()).linear();
  rbd::paramToVector(robot.mbc().alpha, impl_->input.generalizedVelocity);

  const auto & driveJoint = robot.mb().joint(impl_->driveJointIndex);
  const Eigen::Vector3d localAxle = driveJoint.motionSubspace().topRows<3>().col(0);
  const auto & X_0_wheel = robot.mbc().bodyPosW[impl_->wheelBodyIndex];
  impl_->input.wheelAxle = X_0_wheel.rotation().transpose() * localAxle;
  const Eigen::Vector3d wheelAngularVelocity = robot.mbc().bodyVelW[impl_->wheelBodyIndex].angular();
  impl_->input.wheelAxleRate = wheelAngularVelocity.cross(impl_->input.wheelAxle);
  impl_->input.terrainNormal = terrainNormal;
  impl_->input.steeringRate = 0.0;
  impl_->input.velocityGain = velocityGain;
  return impl_->geometry.update(impl_->input);
}

const RollingContactDescription & RollingContactRobotGeometry::description() const noexcept
{
  return impl_->description;
}

const RollingContactKinematics & RollingContactRobotGeometry::kinematics() const noexcept
{
  return impl_->input;
}

const RollingContactGeometryResult & RollingContactRobotGeometry::result() const noexcept
{
  return impl_->geometry.result();
}

void PlanarWheel::validate() const
{
  requireFinite(offset, "planar wheel offset");
  if(!finite(steeringAngle) || !finite(steeringRate))
  {
    throw std::invalid_argument("Planar wheel steering values must be finite");
  }
  if(!finite(radius) || radius <= 0.0) { throw std::invalid_argument("Planar wheel radius must be positive"); }
  requireSpinSign(spinSign);
}

PlanarRollingResult differentialDriveRollingMatrix(double trackWidth,
                                                   double leftRadius,
                                                   double rightRadius,
                                                   double leftSpinSign,
                                                   double rightSpinSign)
{
  if(!finite(trackWidth) || trackWidth <= 0.0)
  {
    throw std::invalid_argument("Differential-drive track width must be positive");
  }
  if(!finite(leftRadius) || leftRadius <= 0.0 || !finite(rightRadius) || rightRadius <= 0.0)
  {
    throw std::invalid_argument("Differential-drive wheel radii must be positive");
  }
  requireSpinSign(leftSpinSign);
  requireSpinSign(rightSpinSign);

  PlanarRollingResult out;
  out.matrix.setZero(3, 5);
  out.accelerationBias.setZero(3);
  out.matrix(0, 0) = 1.0;
  out.matrix(0, 2) = -0.5 * trackWidth;
  out.matrix(0, 3) = -leftRadius * leftSpinSign;
  out.matrix(1, 0) = 1.0;
  out.matrix(1, 2) = 0.5 * trackWidth;
  out.matrix(1, 4) = -rightRadius * rightSpinSign;
  out.matrix(2, 1) = 1.0;
  return out;
}

PlanarRollingResult steeringRollingMatrix(const std::vector<PlanarWheel> & wheels,
                                          const Eigen::Vector3d & planarVelocity)
{
  if(wheels.empty()) { throw std::invalid_argument("At least one steering wheel is required"); }
  requireFinite(planarVelocity, "planarVelocity");

  PlanarRollingResult out;
  const auto nrWheels = static_cast<Eigen::Index>(wheels.size());
  out.matrix.setZero(2 * nrWheels, 3 + nrWheels);
  out.accelerationBias.setZero(2 * nrWheels);
  for(Eigen::Index i = 0; i < nrWheels; ++i)
  {
    const auto & wheel = wheels[static_cast<size_t>(i)];
    wheel.validate();
    const double c = std::cos(wheel.steeringAngle);
    const double s = std::sin(wheel.steeringAngle);
    const Eigen::Vector2d rolling(c, s);
    const Eigen::Vector2d lateral(-s, c);
    const Eigen::Vector2d rollingRate = wheel.steeringRate * lateral;
    const Eigen::Vector2d lateralRate = -wheel.steeringRate * rolling;
    Eigen::Matrix<double, 2, 3> carrierMap;
    carrierMap << 1.0, 0.0, -wheel.offset.y(), 0.0, 1.0, wheel.offset.x();

    out.matrix.block<1, 3>(2 * i, 0).noalias() = rolling.transpose() * carrierMap;
    out.matrix(2 * i, 3 + i) = -wheel.radius * wheel.spinSign;
    out.matrix.block<1, 3>(2 * i + 1, 0).noalias() = lateral.transpose() * carrierMap;
    out.accelerationBias(2 * i) = rollingRate.dot(carrierMap * planarVelocity);
    out.accelerationBias(2 * i + 1) = lateralRate.dot(carrierMap * planarVelocity);
  }
  return out;
}

Eigen::Matrix<double, 6, 1> contactWrenchAtCarrier(const Eigen::Vector3d & carrierCenter,
                                                   const Eigen::Vector3d & contactPoint,
                                                   const Eigen::Vector3d & force)
{
  requireFinite(carrierCenter, "carrierCenter");
  requireFinite(contactPoint, "contactPoint");
  requireFinite(force, "force");
  Eigen::Matrix<double, 6, 1> wrench;
  wrench.head<3>() = (contactPoint - carrierCenter).cross(force);
  wrench.tail<3>() = force;
  return wrench;
}

} // namespace mc_rbdyn
