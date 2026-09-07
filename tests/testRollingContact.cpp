#include <mc_rbdyn/RollingContact.h>

#include <boost/test/unit_test.hpp>

#include <Eigen/Geometry>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace allocation_probe
{

std::atomic<bool> enabled{false};
std::atomic<size_t> count{0};

void record() noexcept
{
  if(enabled.load(std::memory_order_relaxed)) { count.fetch_add(1, std::memory_order_relaxed); }
}

} // namespace allocation_probe

void * operator new(std::size_t size)
{
  allocation_probe::record();
  if(void * memory = std::malloc(size)) { return memory; }
  throw std::bad_alloc{};
}

void * operator new[](std::size_t size)
{
  return ::operator new(size);
}

void operator delete(void * memory) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory) noexcept
{
  std::free(memory);
}

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void operator delete(void * memory, std::size_t) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory, std::size_t) noexcept
{
  std::free(memory);
}

void * operator new(std::size_t size, std::align_val_t alignment)
{
  allocation_probe::record();
  void * memory = nullptr;
  if(posix_memalign(&memory, static_cast<std::size_t>(alignment), size) != 0) { throw std::bad_alloc{}; }
  return memory;
}

void * operator new[](std::size_t size, std::align_val_t alignment)
{
  return ::operator new(size, alignment);
}

void operator delete(void * memory, std::align_val_t) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory, std::align_val_t) noexcept
{
  std::free(memory);
}

void operator delete(void * memory, std::size_t, std::align_val_t) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory, std::size_t, std::align_val_t) noexcept
{
  std::free(memory);
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace
{

constexpr double tolerance = 1e-11;
// M_PI is not guaranteed by the C++ standard, this codebase spells the constant out.
constexpr double pi = 3.14159265358979323846;

/** Front-left Ranger Mini V3 steering wheel. */
mc_rbdyn::PlanarWheel rangerFrontLeftWheel()
{
  mc_rbdyn::PlanarWheel wheel;
  wheel.offset = Eigen::Vector2d(0.247, 0.182);
  wheel.radius = 0.125;
  wheel.spinSign = 1.0;
  return wheel;
}

Eigen::Vector2d wheelCentreVelocity(const mc_rbdyn::PlanarWheel & wheel, const Eigen::Vector3d & planarTwist)
{
  return Eigen::Vector2d(planarTwist.x() - planarTwist.z() * wheel.offset.y(),
                         planarTwist.y() + planarTwist.z() * wheel.offset.x());
}

void checkVector(const Eigen::VectorXd & actual, const Eigen::VectorXd & expected, double tol = tolerance)
{
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  BOOST_CHECK_SMALL((actual - expected).norm(), tol);
}

/** Right-handed planar rotation by @p angle. */
Eigen::Matrix2d planarRotation(double angle)
{
  Eigen::Matrix2d rotation;
  rotation << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
  return rotation;
}

/** One wheel's two scalar rolling/lateral residuals, propagated in the inertial frame.
 *
 * This reads no constraint row. The chassis yaw psi, the carrier offset rho, the
 * wheel heading delta and the drive rate are each propagated from the physical
 * motion, and the two residuals are the frame-invariant scalars
 * t . p - r sigma thetaDot and l . p. Central-differencing residual() is
 * therefore an oracle for the assembled rows rather than a restatement of them.
 *
 * The chassis-relative parametrisation is the point: `linear` is the chassis-basis
 * linear velocity and `linearRate` its chassis-basis derivative, which is exactly
 * the linear part of the chassis-aligned decision variable. The inertial-frame
 * derivative of the same motion is linearRate + omega * J * linear, and that
 * difference is what the two row conventions disagree about.
 */
struct PlanarWheelMotion
{
  Eigen::Vector2d offset = Eigen::Vector2d::Zero();
  double radius = 0.2;
  double spinSign = 1.0;
  double steeringAngle = 0.0;
  double steeringRate = 0.0;
  Eigen::Vector2d linear = Eigen::Vector2d::Zero();
  double yaw = 0.0;
  Eigen::Vector2d linearRate = Eigen::Vector2d::Zero();
  double yawRate = 0.0;
  double rollingRate = 0.0;
  double rollingAcceleration = 0.0;
  /** Cubic coefficients: they change no value and no first derivative at t = 0.
   *
   * With them at zero, an unsteered wheel's residual is affine in time -- the
   * rotation cancels between the direction and the carrier offset -- so its
   * central difference is exact and there is no truncation error whose order
   * could be measured. ROW-01 therefore switches them on to obtain a fixture
   * with a nonzero third derivative, which is what makes the ORC-01 rate
   * assertion non-vacuous there.
   */
  Eigen::Vector2d linearJerk = Eigen::Vector2d::Zero();
  double yawJerk = 0.0;
  double rollingJerk = 0.0;

  /** [rolling residual, lateral residual] at @p time. */
  Eigen::Vector2d residual(double time) const
  {
    const double cubic = time * time * time / 6.0;
    const double psi = yaw * time + 0.5 * yawRate * time * time + yawJerk * cubic * time / 4.0;
    const Eigen::Matrix2d rotation = planarRotation(psi);
    const Eigen::Vector2d velocity = rotation * (linear + linearRate * time + linearJerk * cubic);
    const Eigen::Vector2d carrier = rotation * offset;
    const double omega = yaw + yawRate * time + yawJerk * cubic;
    const Eigen::Vector2d point = velocity + omega * Eigen::Vector2d(-carrier.y(), carrier.x());
    const double heading = psi + steeringAngle + steeringRate * time;
    const Eigen::Vector2d rolling(std::cos(heading), std::sin(heading));
    const Eigen::Vector2d lateral(-std::sin(heading), std::cos(heading));
    return {rolling.dot(point) - radius * spinSign * (rollingRate + rollingAcceleration * time + rollingJerk * cubic),
            lateral.dot(point)};
  }

  Eigen::Vector2d residualRate(double step) const { return (residual(step) - residual(-step)) / (2.0 * step); }

  /** The chassis-basis twist [vx, vy, omega] at t = 0. */
  Eigen::Vector3d chassisTwist() const { return {linear.x(), linear.y(), yaw}; }

  /** Chassis-aligned decision variable [ax, ay, omegaDot]. */
  Eigen::Vector3d chassisAlignedRate() const { return {linearRate.x(), linearRate.y(), yawRate}; }

  /** Inertial-convention decision variable: the same motion, resolved in a fixed frame. */
  Eigen::Vector3d inertialRate() const
  {
    const Eigen::Vector2d coriolis(-yaw * linear.y(), yaw * linear.x());
    return {linearRate.x() + coriolis.x(), linearRate.y() + coriolis.y(), yawRate};
  }
};

/** Measured convergence order of the central difference of @p motion's residuals. */
double centralDifferenceOrder(const PlanarWheelMotion & motion, const Eigen::Vector2d & analytic)
{
  const double coarse = (motion.residualRate(4e-3) - analytic).lpNorm<Eigen::Infinity>();
  const double fine = (motion.residualRate(1e-3) - analytic).lpNorm<Eigen::Infinity>();
  BOOST_REQUIRE_GT(fine, 0.0);
  return std::log2(coarse / fine) / 2.0;
}

/** Richardson table for a central difference of a vector quantity of time.
 *
 * `error[k]` is the infinity-norm error of (f(h) - f(-h)) / 2h against the
 * analytic derivative at step h0 / 2^k, and `order[k]` is
 * log2(error[k] / error[k+1]): the measured convergence order across one
 * halving. This is the assertion ORC-01 actually asks for. A bare threshold is
 * passed by a wrong derivative whose error happens to be small, but a wrong
 * derivative leaves an h-independent error term, so its measured order collapses
 * towards zero however tight the threshold is.
 */
struct RichardsonReport
{
  std::array<double, 4> error{};
  std::array<double, 3> order{};
};

/** @p value maps a time offset to the differentiated quantity; @p analytic is its claimed derivative. */
template<typename Value>
RichardsonReport richardson(const Value & value, const Eigen::VectorXd & analytic, double h0, const char * what)
{
  RichardsonReport report;
  for(size_t k = 0; k < report.error.size(); ++k)
  {
    const double h = h0 / static_cast<double>(1u << k);
    const Eigen::VectorXd difference = (value(h) - value(-h)) / (2.0 * h);
    BOOST_REQUIRE_EQUAL(difference.size(), analytic.size());
    report.error[k] = (difference - analytic).lpNorm<Eigen::Infinity>();
  }
  for(size_t k = 0; k < report.order.size(); ++k)
  {
    // A quantity whose third derivative vanishes is differenced exactly and has
    // no order to measure: that is a fixture defect, not a pass.
    BOOST_REQUIRE_MESSAGE(report.error[k + 1] > 0.0,
                          what << ": the central difference is exact at h = " << h0 / static_cast<double>(1u << (k + 1))
                               << ", so the fixture exercises no third derivative and the rate assertion is vacuous");
    report.order[k] = std::log2(report.error[k] / report.error[k + 1]);
  }
  return report;
}

/** Assert the ORC-01 pass condition: convention T2's threshold and the rate.
 *
 * The threshold is taken at the convention's own step h = 1e-6; the rate is
 * measured on the coarser sweep, where the O(h^2) truncation still dominates
 * round-off. Measuring the rate at 1e-6 would report the round-off floor's
 * slope instead of the method's order.
 */
template<typename Value>
void checkQuadraticConvergence(const Value & value, const Eigen::VectorXd & analytic, const std::string & what)
{
  const Eigen::VectorXd atConvention = (value(1e-6) - value(-1e-6)) / 2e-6;
  const double conventionError = (atConvention - analytic).lpNorm<Eigen::Infinity>();
  const auto report = richardson(value, analytic, 8e-3, what.c_str());
  std::ostringstream line;
  line << "ORC-01 " << what << ": error at h=1e-6 " << conventionError << ", errors";
  for(const double error : report.error) { line << " " << error; }
  line << ", orders";
  for(const double order : report.order) { line << " " << order; }
  BOOST_TEST_MESSAGE(line.str());
  BOOST_CHECK_MESSAGE(conventionError < 1e-6,
                      what << ": finite-difference error " << conventionError << " exceeds the 1e-6 convention (T2)");
  for(const double order : report.order)
  {
    BOOST_CHECK_MESSAGE(order >= 1.8 && order <= 2.2,
                        what << ": measured convergence order " << order << " is outside [1.8, 2.2]");
  }
}

/** Right-handed 2D rotation of the planar rolling/lateral pair at angle @p heading. */
Eigen::Vector2d planarRolling(double heading)
{
  return {std::cos(heading), std::sin(heading)};
}

Eigen::Vector2d planarLateral(double heading)
{
  return {-std::sin(heading), std::cos(heading)};
}

/** H_i of eq:planar-carrier-map, the chassis-aligned carrier map. */
Eigen::Matrix<double, 2, 3> planarCarrierMap(const Eigen::Vector2d & offset)
{
  Eigen::Matrix<double, 2, 3> map;
  map << 1.0, 0.0, -offset.y(), 0.0, 1.0, offset.x();
  return map;
}

/** The planar chassis twist a set of rolling rows implies for given wheel rates.
 *
 * Least squares over the twist columns only, so it is defined for the
 * over-determined four-steering block as well as for the square T1 one. This is
 * the "solution" the ORC-05 metamorphic relations are stated about; it needs no
 * reference value, which is the point of a metamorphic relation.
 */
Eigen::Vector3d planarTwistFromWheelRates(const mc_rbdyn::PlanarRollingResult & rows,
                                          const Eigen::VectorXd & wheelRates)
{
  const Eigen::MatrixXd twistColumns = rows.matrix.leftCols(3);
  const Eigen::MatrixXd rateColumns = rows.matrix.rightCols(rows.matrix.cols() - 3);
  return twistColumns.completeOrthogonalDecomposition().solve(-rateColumns * wheelRates);
}

mc_rbdyn::RollingContactKinematics nominalInput()
{
  mc_rbdyn::RollingContactKinematics input;
  input.carrierCenter = Eigen::Vector3d(1.0, 2.0, 0.4);
  input.wheelAxle = Eigen::Vector3d::UnitY();
  input.terrainNormal = Eigen::Vector3d::UnitZ();
  input.carrierJacobian.setZero(3, 5);
  input.carrierJacobian.block<3, 3>(0, 0).setIdentity();
  input.wheelSelector.setZero(5);
  input.wheelSelector(3) = 1.0;
  input.steeringSelector.setZero(5);
  input.steeringSelector(2) = 1.0;
  input.generalizedVelocity.resize(5);
  input.generalizedVelocity << 1.0, 0.3, -0.1, 4.0, 0.0;
  input.radius = 0.2;
  input.width = 0.1;
  input.spinSign = 1.0;
  input.velocityGain = 3.0;
  return input;
}

} // namespace

BOOST_AUTO_TEST_CASE(RollingContactModeAndDescriptionValidation)
{
  BOOST_CHECK_EQUAL(mc_rbdyn::to_string(mc_rbdyn::RollingContactMode::Fixed), "fixed");
  BOOST_CHECK(mc_rbdyn::rollingContactModeFromString("ROLLING") == mc_rbdyn::RollingContactMode::Rolling);
  BOOST_CHECK_THROW(mc_rbdyn::rollingContactModeFromString("flying"), std::invalid_argument);

  mc_rbdyn::RollingContactDescription description;
  description.name = "left";
  description.carrierFrame = "left_carrier";
  description.wheelBody = "left_wheel";
  description.driveJoint = "left_drive";
  description.radius = 0.2;
  description.width = 0.08;
  BOOST_CHECK_NO_THROW(description.validate());
  description.activation = 1.1;
  BOOST_CHECK_THROW(description.validate(), std::invalid_argument);
  description.activation = 1.0;
  for(const double invalidRadius : {0.0, -0.1, std::numeric_limits<double>::quiet_NaN()})
  {
    description.radius = invalidRadius;
    BOOST_CHECK_THROW(description.validate(), std::invalid_argument);
  }
  description.radius = 0.2;
  for(const double invalidFriction : {-0.1, std::numeric_limits<double>::quiet_NaN()})
  {
    description.friction = invalidFriction;
    BOOST_CHECK_THROW(description.validate(), std::invalid_argument);
  }
  description.friction = 0.8;
  for(const double invalidWidth : {-0.1, std::numeric_limits<double>::quiet_NaN()})
  {
    description.width = invalidWidth;
    BOOST_CHECK_THROW(description.validate(), std::invalid_argument);
  }
  description.width = 0.08;
  for(const auto & field : {std::string{"name"}, std::string{"carrier"}, std::string{"body"}, std::string{"drive"}})
  {
    auto invalid = description;
    if(field == "name") { invalid.name.clear(); }
    else if(field == "carrier") { invalid.carrierFrame.clear(); }
    else if(field == "body") { invalid.wheelBody.clear(); }
    else { invalid.driveJoint.clear(); }
    BOOST_CHECK_THROW(invalid.validate(), std::invalid_argument);
  }
}

BOOST_AUTO_TEST_CASE(RollingContactModeManagerHysteresisDwellAndRecovery)
{
  mc_rbdyn::RollingContactModeThresholds thresholds;
  thresholds.slipEnter = 0.10;
  thresholds.slipExit = 0.04;
  thresholds.residualEnter = 0.10;
  thresholds.residualExit = 0.04;
  thresholds.normalForceEnter = 10.0;
  thresholds.normalForceExit = 2.0;
  thresholds.frictionMarginEnter = 1.0;
  thresholds.frictionMarginExit = 0.0;
  thresholds.torqueMarginEnter = 1.0;
  thresholds.torqueMarginExit = 0.0;
  thresholds.minimumDwell = 0.03;
  thresholds.transitionTime = 0.05;
  thresholds.filterTimeConstant = 0.0;
  mc_rbdyn::RollingContactModeManager manager(thresholds);
  mc_rbdyn::RollingContactModeObservation observation;
  observation.valid = true;
  observation.normalForce = 20.0;
  observation.frictionMargin = 2.0;
  observation.torqueMargin = 2.0;

  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Detached);
  manager.update(observation, 0.01);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Detached);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Rolling);
  BOOST_CHECK_CLOSE(manager.state().activation, 0.2, 1e-12);

  observation.slipSpeed = thresholds.slipEnter;
  for(int i = 0; i < 4; ++i) { manager.update(observation, 0.01); }
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Rolling);
  observation.slipSpeed = thresholds.slipEnter + 0.001;
  manager.update(observation, 0.01);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Rolling);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Sliding);

  observation.slipSpeed = 0.06;
  for(int i = 0; i < 5; ++i) { manager.update(observation, 0.01); }
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Sliding);
  observation.slipSpeed = thresholds.slipExit - 0.001;
  manager.update(observation, 0.01);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Sliding);
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Rolling);

  observation.normalForce = 1.0;
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Detached);
  manager.requestedMode(mc_rbdyn::RollingContactMode::Fixed);
  observation.normalForce = 20.0;
  for(int i = 0; i < 3; ++i) { manager.update(observation, 0.01); }
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Fixed);

  observation.valid = false;
  manager.update(observation, 0.01);
  BOOST_CHECK(manager.state().estimated == mc_rbdyn::RollingContactMode::Detached);
  BOOST_CHECK(!manager.state().measurementValid);
  BOOST_CHECK_EQUAL(manager.state().invalidReason, "measurement-invalid");
  BOOST_CHECK_THROW(manager.update(observation, 0.0), std::invalid_argument);

  thresholds.slipEnter = thresholds.slipExit;
  BOOST_CHECK_THROW((void)mc_rbdyn::RollingContactModeManager{thresholds}, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(RollingFramePointRowsSlipAndBias)
{
  mc_rbdyn::RollingContactGeometry geometry(5);
  auto input = nominalInput();
  input.carrierNormalAcceleration = Eigen::Vector3d(0.1, 0.2, 0.3);
  input.steeringRate = 2.0;
  const auto & result = geometry.update(input);

  checkVector(result.rollingDirection, Eigen::Vector3d::UnitX());
  checkVector(result.lateralDirection, Eigen::Vector3d::UnitY());
  checkVector(result.normalDirection, Eigen::Vector3d::UnitZ());
  BOOST_CHECK_SMALL(result.orthonormalError, tolerance);
  BOOST_CHECK_SMALL(result.rightHandedError, tolerance);
  checkVector(result.contactPoint, Eigen::Vector3d(1.0, 2.0, 0.2));
  checkVector(result.lineStart, Eigen::Vector3d(1.0, 1.95, 0.2));
  checkVector(result.lineEnd, Eigen::Vector3d(1.0, 2.05, 0.2));

  Eigen::Matrix<double, 3, 5> expectedMatrix = Eigen::Matrix<double, 3, 5>::Zero();
  expectedMatrix(0, 0) = 1.0;
  expectedMatrix(0, 3) = -0.2;
  expectedMatrix(1, 1) = 1.0;
  expectedMatrix(2, 2) = 1.0;
  BOOST_CHECK_SMALL((result.rollingMatrix - expectedMatrix).norm(), tolerance);
  checkVector(result.velocityResidual, Eigen::Vector3d(0.2, 0.3, -0.1));
  checkVector(result.slipVelocity, Eigen::Vector3d(0.2, 0.3, 0.0));
  checkVector(result.tangentialCarrierVelocity, Eigen::Vector3d(1.0, 0.3, 0.0));

  // tdot = 2 l, ldot = -2 t. Hence Gdot*alpha is
  // [0.1 + 2*0.3, 0.2 - 2*1.0, 0.3].
  checkVector(result.accelerationBias, Eigen::Vector3d(0.7, -1.8, 0.3));
  checkVector(result.rhs, Eigen::Vector3d(-1.3, 0.9, 0.0));
}

BOOST_AUTO_TEST_CASE(RollingGeometryRejectsInvalidInputAndRecovers)
{
  mc_rbdyn::RollingContactGeometry geometry(5);
  auto input = nominalInput();
  input.radius = -0.1;
  BOOST_CHECK_THROW(geometry.update(input), std::invalid_argument);
  input = nominalInput();
  BOOST_CHECK_NO_THROW(geometry.update(input));

  input.wheelAxle = input.terrainNormal;
  BOOST_CHECK_THROW(geometry.update(input), std::invalid_argument);
  input = nominalInput();
  input.generalizedVelocity(0) = std::numeric_limits<double>::quiet_NaN();
  BOOST_CHECK_THROW(geometry.update(input), std::invalid_argument);
  input = nominalInput();
  BOOST_CHECK_NO_THROW(geometry.update(input));

  input.steeringSelector.setZero(4);
  BOOST_CHECK_THROW(geometry.update(input), std::invalid_argument);
  input = nominalInput();
  BOOST_CHECK_NO_THROW(geometry.update(input));
}

BOOST_AUTO_TEST_CASE(RollingGeometryPerformsNoHeapAllocationAfterWarmup)
{
  mc_rbdyn::RollingContactGeometry geometry(5);
  auto input = nominalInput();
  geometry.update(input);
  allocation_probe::count.store(0, std::memory_order_relaxed);
  allocation_probe::enabled.store(true, std::memory_order_relaxed);
  double checksum = 0.0;
  for(int cycle = 0; cycle < 10000; ++cycle)
  {
    input.generalizedVelocity(0) = 1.0 + 1e-6 * static_cast<double>(cycle);
    checksum += geometry.update(input).velocityResidual.x();
  }
  allocation_probe::enabled.store(false, std::memory_order_relaxed);
  const size_t allocations = allocation_probe::count.load(std::memory_order_relaxed);
  BOOST_CHECK(std::isfinite(checksum));
  BOOST_CHECK_EQUAL(allocations, 0);
}

BOOST_AUTO_TEST_CASE(DifferentialDriveRowsHaveNoDuplicateLateralConstraint)
{
  const auto result = mc_rbdyn::differentialDriveRollingMatrix(0.6, 0.2, 0.25);
  Eigen::Matrix<double, 3, 5> expected = Eigen::Matrix<double, 3, 5>::Zero();
  expected << 1.0, 0.0, -0.3, -0.2, 0.0, 1.0, 0.0, 0.3, 0.0, -0.25, 0.0, 1.0, 0.0, 0.0, 0.0;
  BOOST_CHECK_SMALL((result.matrix - expected).norm(), tolerance);
  BOOST_CHECK_EQUAL(result.matrix.rows(), 3);

  Eigen::Vector<double, 5> pureRolling;
  pureRolling << 1.0, 0.0, 0.5, 4.25, 4.6;
  BOOST_CHECK_SMALL((result.matrix * pureRolling).norm(), tolerance);
}

BOOST_AUTO_TEST_CASE(PlanarSpecializationsRecoverCommonChassisTwists)
{
  std::mt19937 generator(421337);
  std::uniform_real_distribution<double> rate(-8.0, 8.0);
  std::uniform_real_distribution<double> twist(-2.0, 2.0);
  std::uniform_real_distribution<double> offset(-0.8, 0.8);
  constexpr double track = 0.6;
  constexpr double leftRadius = 0.2;
  constexpr double rightRadius = 0.23;
  const auto differential =
      mc_rbdyn::differentialDriveRollingMatrix(track, leftRadius, rightRadius);
  for(size_t i = 0; i < 100; ++i)
  {
    const double leftRate = rate(generator);
    const double rightRate = rate(generator);
    const double vx = 0.5 * (leftRadius * leftRate + rightRadius * rightRate);
    const double omega = (rightRadius * rightRate - leftRadius * leftRate) / track;
    Eigen::Vector<double, 5> state;
    state << vx, 0.0, omega, leftRate, rightRate;
    BOOST_CHECK_SMALL((differential.matrix * state).norm(), 1e-10);
  }

  std::vector<mc_rbdyn::PlanarWheel> wheels(4);
  for(auto & wheel : wheels)
  {
    wheel.radius = 0.2;
    wheel.offset = Eigen::Vector2d(offset(generator), offset(generator));
  }
  for(size_t sample = 0; sample < 100; ++sample)
  {
    const Eigen::Vector3d chassisTwist(twist(generator), twist(generator), twist(generator));
    Eigen::VectorXd state(7);
    state.head<3>() = chassisTwist;
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      const auto & p = wheels[i].offset;
      const Eigen::Vector2d carrierVelocity(chassisTwist.x() - p.y() * chassisTwist.z(),
                                            chassisTwist.y() + p.x() * chassisTwist.z());
      BOOST_REQUIRE_GT(carrierVelocity.norm(), 1e-8);
      wheels[i].steeringAngle = std::atan2(carrierVelocity.y(), carrierVelocity.x());
      state(static_cast<Eigen::Index>(3 + i)) = carrierVelocity.norm() / wheels[i].radius;
    }
    const auto compatible = mc_rbdyn::steeringRollingMatrix(wheels, chassisTwist);
    BOOST_CHECK_SMALL((compatible.matrix * state).norm(), 1e-10);

    Eigen::VectorXd incompatible = state;
    incompatible(3) += 0.5;
    BOOST_CHECK_GT((compatible.matrix * incompatible).norm(), 0.09);
  }
}

BOOST_AUTO_TEST_CASE(SteeringRowsIncludeDirectionDerivative)
{
  std::vector<mc_rbdyn::PlanarWheel> wheels(1);
  wheels[0].offset = Eigen::Vector2d(0.7, -0.4);
  wheels[0].steeringAngle = 0.37;
  wheels[0].steeringRate = -0.8;
  wheels[0].radius = 0.22;
  const Eigen::Vector3d velocity(0.9, -0.2, 0.6);
  const auto result = mc_rbdyn::steeringRollingMatrix(wheels, velocity);

  Eigen::Vector4d fullVelocity;
  fullVelocity << velocity, 1.3;
  constexpr double dt = 1e-7;
  auto plus = wheels;
  auto minus = wheels;
  plus[0].steeringAngle += wheels[0].steeringRate * dt;
  minus[0].steeringAngle -= wheels[0].steeringRate * dt;
  const auto plusResult = mc_rbdyn::steeringRollingMatrix(plus, velocity);
  const auto minusResult = mc_rbdyn::steeringRollingMatrix(minus, velocity);
  const Eigen::Vector2d finiteDifference =
      (plusResult.matrix * fullVelocity - minusResult.matrix * fullVelocity) / (2.0 * dt);
  BOOST_CHECK_SMALL((finiteDifference - result.accelerationBias).norm(), 2e-9);
}

BOOST_AUTO_TEST_CASE(PlanarRowsPinTheChassisAlignedBasisConventionROW04)
{
  // ROW-04, both branches. The shipped planar rows resolve the twist in the
  // chassis-aligned basis of assumption A1, where H_i is constant and the only
  // direction derivative is the steering one. An assembler that resolved the
  // same motion in an inertial basis would produce rows that differ from these
  // by exactly +omega * (l_i . v) in each rolling row and -omega * (t_i . v) in
  // each lateral row; for an unsteered wheel those are +omega v_y and
  // -omega v_x, the two offsets the testcard names.
  //
  // The oracle is PlanarWheelMotion: it propagates the physical motion in the
  // inertial frame and central-differences the frame-invariant residual, so it
  // shares no algebra with steeringRollingMatrix().
  const Eigen::Vector2d linear(0.9, -0.35);
  constexpr double yaw = 0.6;
  const Eigen::Vector2d linearRate(0.4, 0.7);
  constexpr double yawRate = -0.25;
  BOOST_REQUIRE_GT(std::abs(yaw * linear.x()), 0.1); // the regime the card asks for: omega * v_x != 0

  // --- Four-steering rows -------------------------------------------------
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3}, Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3}, Eigen::Vector2d{-0.45, -0.3}};
  const std::array<double, 4> angles = {0.31, -0.22, 0.47, -0.13};
  const std::array<double, 4> rates = {0.5, -0.3, 0.7, -0.9};
  const std::array<double, 4> driveRates = {3.1, 2.4, -1.2, 0.8};
  const std::array<double, 4> driveAccelerations = {1.5, -2.2, 0.9, 3.3};

  std::vector<mc_rbdyn::PlanarWheel> wheels(4);
  std::vector<PlanarWheelMotion> motions(4);
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    wheels[i].offset = offsets[i];
    wheels[i].steeringAngle = angles[i];
    wheels[i].steeringRate = rates[i];
    wheels[i].radius = 0.2;
    motions[i].offset = offsets[i];
    motions[i].radius = wheels[i].radius;
    motions[i].steeringAngle = angles[i];
    motions[i].steeringRate = rates[i];
    motions[i].linear = linear;
    motions[i].yaw = yaw;
    motions[i].linearRate = linearRate;
    motions[i].yawRate = yawRate;
    motions[i].rollingRate = driveRates[i];
    motions[i].rollingAcceleration = driveAccelerations[i];
  }
  const Eigen::Vector3d twist(linear.x(), linear.y(), yaw);
  const auto rows = mc_rbdyn::steeringRollingMatrix(wheels, twist);

  Eigen::VectorXd chassisAligned(7);
  chassisAligned << linearRate.x(), linearRate.y(), yawRate, driveAccelerations[0], driveAccelerations[1],
      driveAccelerations[2], driveAccelerations[3];
  Eigen::VectorXd inertial = chassisAligned;
  inertial.head<3>() = motions[0].inertialRate();

  const Eigen::VectorXd shipped = rows.matrix * chassisAligned + rows.accelerationBias;
  const Eigen::VectorXd inertialCoefficients = rows.matrix * inertial;
  for(size_t i = 0; i < motions.size(); ++i)
  {
    const auto row = static_cast<Eigen::Index>(2 * i);
    const Eigen::Vector2d analytic = shipped.segment<2>(row);
    // Branch one: the shipped chassis-aligned rows ARE the exact derivative.
    const Eigen::Vector2d measured = motions[i].residualRate(1e-6);
    const double order = centralDifferenceOrder(motions[i], analytic);
    BOOST_TEST_MESSAGE("ROW-04 wheel " << i << " chassis-aligned FD error "
                                       << (measured - analytic).lpNorm<Eigen::Infinity>() << ", order " << order);
    BOOST_CHECK_SMALL((measured - analytic).lpNorm<Eigen::Infinity>(), 1e-6);
    BOOST_CHECK_GE(order, 1.8);
    BOOST_CHECK_LE(order, 2.2);

    // Branch two: the deliberately mutated inertial assembler. Pairing the same
    // row coefficients with the inertial-frame decision variable leaves a bias
    // that differs from the shipped one by exactly the stated offsets.
    const Eigen::Vector2d inertialBias = measured - inertialCoefficients.segment<2>(row);
    const Eigen::Vector2d shippedBias = rows.accelerationBias.segment<2>(row);
    const Eigen::Vector2d rolling(std::cos(angles[i]), std::sin(angles[i]));
    const Eigen::Vector2d lateral(-std::sin(angles[i]), std::cos(angles[i]));
    const Eigen::Vector2d expectedOffset(yaw * lateral.dot(linear), -yaw * rolling.dot(linear));
    BOOST_TEST_MESSAGE("ROW-04 wheel " << i << " inertial offset [" << (inertialBias - shippedBias).transpose()
                                       << "], expected [" << expectedOffset.transpose() << "]");
    BOOST_CHECK_SMALL((inertialBias - shippedBias - expectedOffset).lpNorm<Eigen::Infinity>(), 1e-6);
    // Non-vacuity: the offset the two conventions differ by is not small.
    BOOST_CHECK_GT(expectedOffset.lpNorm<Eigen::Infinity>(), 0.1);
  }

  // --- Differential-drive rows: the offsets the card states verbatim -------
  constexpr double track = 0.6;
  constexpr double leftRadius = 0.2;
  constexpr double rightRadius = 0.23;
  const auto differential = mc_rbdyn::differentialDriveRollingMatrix(track, leftRadius, rightRadius);
  // Rows are [left longitudinal, right longitudinal, lateral]. The lateral row
  // is the one taken at the chassis origin, so its oracle carries a zero offset.
  std::array<PlanarWheelMotion, 3> differentialMotions;
  const std::array<Eigen::Vector2d, 3> differentialOffsets = {
      Eigen::Vector2d{0.0, 0.5 * track}, Eigen::Vector2d{0.0, -0.5 * track}, Eigen::Vector2d::Zero()};
  const std::array<double, 3> radii = {leftRadius, rightRadius, 0.2};
  const std::array<double, 2> wheelRates = {4.25, 4.6};
  const std::array<double, 2> wheelAccelerations = {2.5, -1.75};
  for(size_t i = 0; i < differentialMotions.size(); ++i)
  {
    differentialMotions[i].offset = differentialOffsets[i];
    differentialMotions[i].radius = radii[i];
    differentialMotions[i].linear = linear;
    differentialMotions[i].yaw = yaw;
    differentialMotions[i].linearRate = linearRate;
    differentialMotions[i].yawRate = yawRate;
    if(i < 2)
    {
      differentialMotions[i].rollingRate = wheelRates[i];
      differentialMotions[i].rollingAcceleration = wheelAccelerations[i];
    }
  }
  Eigen::VectorXd differentialAligned(5);
  differentialAligned << linearRate.x(), linearRate.y(), yawRate, wheelAccelerations[0], wheelAccelerations[1];
  Eigen::VectorXd differentialInertial = differentialAligned;
  differentialInertial.head<3>() = differentialMotions[0].inertialRate();
  const Eigen::VectorXd differentialShipped = differential.matrix * differentialAligned + differential.accelerationBias;
  const Eigen::VectorXd differentialInertialRows = differential.matrix * differentialInertial;

  Eigen::Vector3d measured;
  measured << differentialMotions[0].residualRate(1e-6).x(), differentialMotions[1].residualRate(1e-6).x(),
      differentialMotions[2].residualRate(1e-6).y();
  BOOST_CHECK_SMALL((measured - differentialShipped).lpNorm<Eigen::Infinity>(), 1e-6);
  // The two rolling rows pick up +omega v_y, the lateral row -omega v_x: exactly
  // the two offsets the testcard names, asserted as signed values.
  const Eigen::Vector3d offsetsFound = measured - differentialInertialRows - differential.accelerationBias;
  const Eigen::Vector3d expected(yaw * linear.y(), yaw * linear.y(), -yaw * linear.x());
  BOOST_TEST_MESSAGE("ROW-04 differential inertial offsets [" << offsetsFound.transpose() << "], expected ["
                                                              << expected.transpose() << "]");
  BOOST_CHECK_SMALL((offsetsFound - expected).lpNorm<Eigen::Infinity>(), 1e-6);
  BOOST_CHECK_GT(std::abs(expected.z()), 0.5);
}

BOOST_AUTO_TEST_CASE(SteeringReferenceInvertsTheExpandedConstraints)
{
  const auto wheel = rangerFrontLeftWheel();

  const auto forward = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0);
  BOOST_CHECK(forward.commanded);
  BOOST_CHECK_SMALL(forward.steeringAngle, tolerance);
  BOOST_CHECK_CLOSE(forward.rollingRate, 1.0 / 0.125, 1e-9);

  const auto crab = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, 1.0, 0.0), 0.0);
  BOOST_CHECK(crab.commanded);
  BOOST_CHECK_CLOSE(crab.steeringAngle, 0.5 * pi, 1e-9);
  BOOST_CHECK_CLOSE(crab.rollingRate, 1.0 / 0.125, 1e-9);

  // Pure yaw points the wheel centre backwards-left: the raw heading falls outside the
  // hinge range so the flipped branch is used and the wheel must roll backwards.
  const auto yaw = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, 0.0, 1.0), 0.0);
  BOOST_CHECK(yaw.commanded);
  BOOST_CHECK_CLOSE(yaw.steeringAngle, std::atan2(0.247, -0.182) - pi, 1e-9);
  BOOST_CHECK_LT(yaw.rollingRate, 0.0);
  BOOST_CHECK_CLOSE(std::abs(yaw.rollingRate), std::hypot(0.182, 0.247) / 0.125, 1e-9);

  const Eigen::Vector3d twist(0.4, -0.2, 0.3);
  const auto general = mc_rbdyn::steeringWheelReference(wheel, twist, 0.0);
  BOOST_CHECK(general.commanded);
  const Eigen::Vector2d point = wheelCentreVelocity(wheel, twist);
  const double c = std::cos(general.steeringAngle);
  const double s = std::sin(general.steeringAngle);
  BOOST_CHECK_SMALL(c * point.x() + s * point.y() - wheel.radius * wheel.spinSign * general.rollingRate, 1e-12);
  BOOST_CHECK_SMALL(-s * point.x() + c * point.y(), 1e-12);
}

BOOST_AUTO_TEST_CASE(YawCommandRollsTheFrontWheelsInOppositeDirections)
{
  // Pin the yaw convention at the geometry layer: a +z yaw command spins the
  // left and right wheels of an axle in opposite directions. Everything above
  // this (QP rate rows, controller references) inherits that sign, so a silent
  // flip here would mirror every commanded turn.
  auto left = rangerFrontLeftWheel();
  auto right = rangerFrontLeftWheel();
  right.offset.y() = -left.offset.y();

  const Eigen::Vector3d yawCommand(0.0, 0.0, 1.0);
  const auto leftReference = mc_rbdyn::steeringWheelReference(left, yawCommand, 0.0);
  const auto rightReference = mc_rbdyn::steeringWheelReference(right, yawCommand, 0.0);
  BOOST_REQUIRE(leftReference.commanded);
  BOOST_REQUIRE(rightReference.commanded);
  BOOST_CHECK_LT(leftReference.rollingRate * rightReference.rollingRate, 0.0);
  BOOST_CHECK_LT(leftReference.rollingRate, 0.0);
  BOOST_CHECK_GT(rightReference.rollingRate, 0.0);
  // Both wheels sit at the same distance from the chassis centre, so the yaw
  // command asks them for the same speed with opposite signs.
  BOOST_CHECK_CLOSE(std::abs(leftReference.rollingRate), std::abs(rightReference.rollingRate), 1e-9);
  BOOST_CHECK_CLOSE(std::abs(leftReference.rollingRate), std::hypot(0.247, 0.182) / 0.125, 1e-9);
  // The headings mirror across the chassis' longitudinal axis.
  BOOST_CHECK_CLOSE(leftReference.steeringAngle, std::atan2(0.247, -0.182) - pi, 1e-9);
  BOOST_CHECK_CLOSE(rightReference.steeringAngle, std::atan2(0.247, 0.182), 1e-9);
}

BOOST_AUTO_TEST_CASE(SteeringReferenceStaysWithinLimitsAndAvoidsBranchChatter)
{
  const auto wheel = rangerFrontLeftWheel();

  const auto held = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, -1.0, 0.0), -0.5 * pi);
  BOOST_CHECK_CLOSE(held.steeringAngle, -0.5 * pi, 1e-9);
  const auto perturbed = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(1e-9, -1.0, 0.0), -0.5 * pi);
  BOOST_CHECK_SMALL(perturbed.steeringAngle - held.steeringAngle, 1e-3);

  for(int step = 0; step < 360; ++step)
  {
    const double angle = 2.0 * pi * static_cast<double>(step) / 360.0;
    const Eigen::Vector3d twist(std::cos(angle), std::sin(angle), 0.4);
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, twist, 0.0);
    BOOST_CHECK_LE(std::abs(reference.steeringAngle), 0.5 * pi);
    BOOST_CHECK(std::isfinite(reference.rollingRate));
  }
}

BOOST_AUTO_TEST_CASE(SteeringReferenceHoldsPoseForADegenerateCommand)
{
  const auto wheel = rangerFrontLeftWheel();
  const auto reference = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d::Zero(), 0.3);
  BOOST_CHECK_CLOSE(reference.steeringAngle, 0.3, 1e-12);
  BOOST_CHECK_SMALL(reference.rollingRate, tolerance);
  BOOST_CHECK(!reference.commanded);
}

BOOST_AUTO_TEST_CASE(RampGeometryIsRotationEquivariant)
{
  mc_rbdyn::RollingContactGeometry flatGeometry(5);
  mc_rbdyn::RollingContactGeometry rampGeometry(5);
  const auto flatInput = nominalInput();
  const auto flat = flatGeometry.update(flatInput);

  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitY()).toRotationMatrix();
  auto rampInput = flatInput;
  rampInput.carrierCenter = rotation * flatInput.carrierCenter;
  rampInput.wheelAxle = rotation * flatInput.wheelAxle;
  rampInput.terrainNormal = rotation * flatInput.terrainNormal;
  rampInput.carrierJacobian = rotation * flatInput.carrierJacobian;
  const auto & ramp = rampGeometry.update(rampInput);
  checkVector(ramp.rollingDirection, rotation * flat.rollingDirection);
  checkVector(ramp.lateralDirection, rotation * flat.lateralDirection);
  checkVector(ramp.normalDirection, rotation * flat.normalDirection);
  checkVector(ramp.contactPoint, rotation * flat.contactPoint);
  BOOST_CHECK_SMALL((ramp.rollingMatrix - flat.rollingMatrix).norm(), tolerance);
  checkVector(ramp.slipVelocity, rotation * flat.slipVelocity);
}

BOOST_AUTO_TEST_CASE(RichardsonConvergenceOrderPinsTheHandDerivedDerivativesORC01)
{
  // ORC-01. Every derivative below is hand written in the implementation, so
  // each gets the rate assertion rather than a threshold. The step sequence is
  // 8e-3 down to 1e-3: coarse enough that the O(h^2) truncation dominates
  // round-off at every refinement, and the differentiated quantities are all
  // O(1) as the card requires.

  // --- eq:steering-direction-derivatives, read through the shipped rows -----
  // steeringRollingMatrix() reports Gdot * v as accelerationBias. Freezing the
  // velocity and differencing G(t) therefore isolates tdot and ldot exactly.
  std::vector<mc_rbdyn::PlanarWheel> wheels(4);
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3}, Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3}, Eigen::Vector2d{-0.45, -0.3}};
  const std::array<double, 4> angles = {0.31, -0.22, 0.47, -0.13};
  const std::array<double, 4> steeringRates = {0.5, -0.3, 0.7, -0.9};
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    wheels[i].offset = offsets[i];
    wheels[i].steeringAngle = angles[i];
    wheels[i].steeringRate = steeringRates[i];
    wheels[i].radius = 0.2;
  }
  const Eigen::Vector3d twist(0.9, -0.35, 0.6);
  Eigen::VectorXd frozen(7);
  frozen << twist, 3.1, 2.4, -1.2, 0.8;
  const auto rows = mc_rbdyn::steeringRollingMatrix(wheels, twist);
  const auto steeredRows = [&](double time)
  {
    auto moved = wheels;
    for(size_t i = 0; i < moved.size(); ++i) { moved[i].steeringAngle += steeringRates[i] * time; }
    return Eigen::VectorXd(mc_rbdyn::steeringRollingMatrix(moved, twist).matrix * frozen);
  };
  checkQuadraticConvergence(steeredRows, rows.accelerationBias, "steering direction derivatives");

  // --- The three-dimensional frame rates of RollingContactGeometry ----------
  // Branch one: the axle itself rotates and steeringRate is zero.
  mc_rbdyn::RollingContactGeometry geometry(5);
  auto input = nominalInput();
  input.steeringRate = 0.0;
  const Eigen::Vector3d axleAxis = Eigen::Vector3d(0.3, 0.2, 0.9).normalized();
  constexpr double axleRate = 1.7;
  const Eigen::Vector3d axle = input.wheelAxle;
  input.wheelAxleRate = axleRate * axleAxis.cross(axle);
  const auto & spun = geometry.update(input);
  Eigen::VectorXd spunRates(6);
  spunRates << spun.rollingDirectionRate, spun.lateralDirectionRate;
  const auto spunFrame = [&](double time)
  {
    mc_rbdyn::RollingContactGeometry moved(5);
    auto movedInput = input;
    movedInput.wheelAxle = Eigen::AngleAxisd(axleRate * time, axleAxis).toRotationMatrix() * axle;
    const auto & result = moved.update(movedInput);
    Eigen::VectorXd stacked(6);
    stacked << result.rollingDirection, result.lateralDirection;
    return stacked;
  };
  checkQuadraticConvergence(spunFrame, spunRates, "wheel-axle frame rate");

  // Branch two: a static axle plus the steeringRate correction, which is the
  // path the four-steering assembler takes. tdot = deltaDot l, ldot = -deltaDot t.
  auto steeredInput = nominalInput();
  steeredInput.wheelAxleRate.setZero();
  steeredInput.steeringRate = 2.0;
  const auto & steered = geometry.update(steeredInput);
  BOOST_CHECK_SMALL((steered.rollingDirectionRate - steeredInput.steeringRate * steered.lateralDirection).norm(),
                    tolerance);
  BOOST_CHECK_SMALL((steered.lateralDirectionRate + steeredInput.steeringRate * steered.rollingDirection).norm(),
                    tolerance);
  Eigen::VectorXd steeredRates(6);
  steeredRates << steered.rollingDirectionRate, steered.lateralDirectionRate;
  const auto steeredFrame = [&](double time)
  {
    mc_rbdyn::RollingContactGeometry moved(5);
    auto movedInput = steeredInput;
    movedInput.wheelAxle =
        Eigen::AngleAxisd(steeredInput.steeringRate * time, movedInput.terrainNormal.normalized()).toRotationMatrix()
        * steeredInput.wheelAxle;
    const auto & result = moved.update(movedInput);
    Eigen::VectorXd stacked(6);
    stacked << result.rollingDirection, result.lateralDirection;
    return stacked;
  };
  checkQuadraticConvergence(steeredFrame, steeredRates, "steering-rate frame rate");
}

BOOST_AUTO_TEST_CASE(DifferentialRowsAreTheExactDerivativeAtZeroGainROW01)
{
  // ROW-01. The regime the card names is a trajectory that *satisfies* the
  // velocity constraints, so the fixture is built on the constraint manifold
  // first and only then differentiated.
  constexpr double track = 0.6;
  constexpr double leftRadius = 0.2;
  constexpr double rightRadius = 0.23;
  const auto differential = mc_rbdyn::differentialDriveRollingMatrix(track, leftRadius, rightRadius);

  constexpr double leftRate = 4.25;
  constexpr double rightRate = 6.0;
  const double vx = 0.5 * (leftRadius * leftRate + rightRadius * rightRate);
  const double omega = (rightRadius * rightRate - leftRadius * leftRate) / track;
  Eigen::Vector<double, 5> onManifold;
  onManifold << vx, 0.0, omega, leftRate, rightRate;
  BOOST_REQUIRE_SMALL((differential.matrix * onManifold).lpNorm<Eigen::Infinity>(), tolerance);
  BOOST_REQUIRE_GT(std::abs(omega * vx), 0.1); // omega * v_x != 0, so the basis convention is observable

  const Eigen::Vector2d linearRate(0.4, 0.7);
  constexpr double yawRate = -0.25;
  const std::array<double, 2> wheelAccelerations = {2.5, -1.75};
  Eigen::VectorXd chassisAligned(5);
  chassisAligned << linearRate.x(), linearRate.y(), yawRate, wheelAccelerations[0], wheelAccelerations[1];

  // differentialDriveRollingMatrix() prints no acceleration bias at all: in the
  // chassis-aligned basis of A1 the map H_i is constant and the wheels do not
  // steer, so the printed row IS the whole derivative.
  BOOST_CHECK_SMALL(differential.accelerationBias.lpNorm<Eigen::Infinity>(), tolerance);
  const Eigen::VectorXd printed = differential.matrix * chassisAligned + differential.accelerationBias;

  const std::array<Eigen::Vector2d, 3> wheelOffsets = {
      Eigen::Vector2d{0.0, 0.5 * track}, Eigen::Vector2d{0.0, -0.5 * track}, Eigen::Vector2d::Zero()};
  const std::array<double, 3> radii = {leftRadius, rightRadius, 0.2};
  const std::array<double, 2> wheelRates = {leftRate, rightRate};
  std::array<PlanarWheelMotion, 3> motions;
  for(size_t i = 0; i < motions.size(); ++i)
  {
    motions[i].offset = wheelOffsets[i];
    motions[i].radius = radii[i];
    motions[i].linear = Eigen::Vector2d(vx, 0.0);
    motions[i].yaw = omega;
    motions[i].linearRate = linearRate;
    motions[i].yawRate = yawRate;
    if(i < 2)
    {
      motions[i].rollingRate = wheelRates[i];
      motions[i].rollingAcceleration = wheelAccelerations[i];
    }
    // The fixture really is on the manifold: every propagated residual is zero at t = 0.
    const Eigen::Vector2d residual = motions[i].residual(0.0);
    BOOST_CHECK_SMALL(i < 2 ? residual.x() : residual.y(), tolerance);
  }
  const auto rowValues = [&motions](double time)
  {
    Eigen::VectorXd stacked(3);
    stacked << motions[0].residual(time).x(), motions[1].residual(time).x(), motions[2].residual(time).y();
    return stacked;
  };
  // Branch one, exactness. With no steering and a constant twist rate, the
  // chassis-aligned residual is affine in time: the chassis rotation cancels
  // between the direction and the carrier offset. The central difference is
  // therefore exact at any step, and the printed row must reproduce it to
  // working precision, convention T1. This is a stronger statement than any
  // O(h^2) threshold and it is the one that holds here.
  for(const double step : {4e-3, 1e-3, 1e-6})
  {
    const Eigen::VectorXd difference = (rowValues(step) - rowValues(-step)) / (2.0 * step);
    BOOST_CHECK_SMALL((difference - printed).lpNorm<Eigen::Infinity>(), step < 1e-4 ? 1e-9 : 1e-12);
  }

  // Branch two, the convergence rate. The rate assertion needs a nonzero third
  // derivative, so the same trajectory is given cubic terms. They leave the
  // state and every first derivative at t = 0 untouched, hence the analytic row
  // is unchanged and only the truncation error is affected.
  for(size_t i = 0; i < motions.size(); ++i)
  {
    motions[i].linearJerk = Eigen::Vector2d(-3.1, 2.7);
    motions[i].yawJerk = 4.3;
    if(i < 2) { motions[i].rollingJerk = (i == 0 ? 11.0 : -7.0); }
  }
  checkQuadraticConvergence(rowValues, printed, "ROW-01 differential rows");

  // The "at Kp = 0" half of the card, on the shipped assembler that owns Kp.
  // At zero gain the printed right-hand side is exactly -Gdot * alpha, so the
  // row is the plain derivative; at a nonzero gain it is that derivative offset
  // by exactly -Kp times the measured residual and nothing else.
  mc_rbdyn::RollingContactGeometry geometry(5);
  auto input = nominalInput();
  input.velocityGain = 0.0;
  const auto zeroGain = geometry.update(input);
  BOOST_REQUIRE_GT(zeroGain.velocityResidual.lpNorm<Eigen::Infinity>(), 0.1);
  checkVector(zeroGain.rhs, -zeroGain.accelerationBias);
  for(const double gain : {3.0, 50.0})
  {
    input.velocityGain = gain;
    const auto stabilised = geometry.update(input);
    checkVector(stabilised.rhs, -stabilised.accelerationBias - gain * stabilised.velocityResidual);
    BOOST_CHECK_GT((stabilised.rhs - zeroGain.rhs).lpNorm<Eigen::Infinity>(), 0.1);
  }
}

BOOST_AUTO_TEST_CASE(SteeringRowsAreTheExactDerivativeWithNoHdotTermROW05)
{
  // ROW-05. Assumptions A1-A3: chassis-aligned basis, rigid wheel, steering
  // axis through the carrier. Under A1 the carrier map H_i is constant, so the
  // only direction derivative in the printed bias is the steering one.
  const std::array<Eigen::Vector2d, 4> offsets = {Eigen::Vector2d{0.45, 0.3}, Eigen::Vector2d{0.45, -0.3},
                                                  Eigen::Vector2d{-0.45, 0.3}, Eigen::Vector2d{-0.45, -0.3}};
  const std::array<double, 4> angles = {0.31, -0.22, 0.47, -0.13};
  const std::array<double, 4> steeringRates = {0.5, -0.3, 0.7, -0.9};
  const std::array<double, 4> driveRates = {3.1, 2.4, -1.2, 0.8};
  const std::array<double, 4> driveAccelerations = {1.5, -2.2, 0.9, 3.3};
  const Eigen::Vector2d linear(0.9, -0.35);
  constexpr double yaw = 0.6;
  const Eigen::Vector2d linearRate(0.4, 0.7);
  constexpr double yawRate = -0.25;

  std::vector<mc_rbdyn::PlanarWheel> wheels(4);
  std::vector<PlanarWheelMotion> motions(4);
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    wheels[i].offset = offsets[i];
    wheels[i].steeringAngle = angles[i];
    wheels[i].steeringRate = steeringRates[i];
    wheels[i].radius = 0.2;
    motions[i].offset = offsets[i];
    motions[i].radius = wheels[i].radius;
    motions[i].steeringAngle = angles[i];
    motions[i].steeringRate = steeringRates[i];
    motions[i].linear = linear;
    motions[i].yaw = yaw;
    motions[i].linearRate = linearRate;
    motions[i].yawRate = yawRate;
    motions[i].rollingRate = driveRates[i];
    motions[i].rollingAcceleration = driveAccelerations[i];
  }
  const Eigen::Vector3d twist(linear.x(), linear.y(), yaw);
  const auto rows = mc_rbdyn::steeringRollingMatrix(wheels, twist);
  Eigen::VectorXd chassisAligned(7);
  chassisAligned << linearRate.x(), linearRate.y(), yawRate, driveAccelerations[0], driveAccelerations[1],
      driveAccelerations[2], driveAccelerations[3];
  const Eigen::VectorXd printed = rows.matrix * chassisAligned + rows.accelerationBias;

  const auto rowValues = [&](double time)
  {
    Eigen::VectorXd stacked(8);
    for(size_t i = 0; i < motions.size(); ++i) { stacked.segment<2>(static_cast<Eigen::Index>(2 * i)) = motions[i].residual(time); }
    return stacked;
  };
  checkQuadraticConvergence(rowValues, printed, "ROW-05 steering rows");

  // The second half of the card: no Hdot_i term. The printed bias must be
  // exactly the direction-derivative term. Under the inertial convention the
  // same rows would carry Hdot_i xi = -omega^2 rho_i, adding
  // -omega^2 <t_i, rho_i> to each rolling row and -omega^2 <l_i, rho_i> to each
  // lateral one. Those two offsets are asserted absent, and shown to be large
  // enough in this fixture that their absence is a real finding.
  for(size_t i = 0; i < wheels.size(); ++i)
  {
    const auto row = static_cast<Eigen::Index>(2 * i);
    const Eigen::Vector2d rolling = planarRolling(angles[i]);
    const Eigen::Vector2d lateral = planarLateral(angles[i]);
    const Eigen::Vector2d carrierVelocity = planarCarrierMap(offsets[i]) * twist;
    const Eigen::Vector2d directionOnly(steeringRates[i] * lateral.dot(carrierVelocity),
                                        -steeringRates[i] * rolling.dot(carrierVelocity));
    BOOST_CHECK_SMALL((rows.accelerationBias.segment<2>(row) - directionOnly).lpNorm<Eigen::Infinity>(), tolerance);
    const Eigen::Vector2d inertialHdot(-yaw * yaw * rolling.dot(offsets[i]), -yaw * yaw * lateral.dot(offsets[i]));
    BOOST_TEST_MESSAGE("ROW-05 wheel " << i << " absent Hdot term [" << inertialHdot.transpose() << "]");
    BOOST_CHECK_GT(inertialHdot.lpNorm<Eigen::Infinity>(), 0.05);
  }
}

BOOST_AUTO_TEST_CASE(PlanarRowsSatisfyTheMetamorphicRelationsORC05)
{
  // ORC-05, the two relations that are statements about the planar rows: the
  // radius/rate scaling and the left-right mirror. The third relation, heading
  // invariance of the chassis-frame solution, is a statement about the solved
  // whole-body QP and lives in
  // WholeBodySolutionIsChassisHeadingInvariantORC05 in
  // tests/testRollingContactSolver.cpp. Seed recorded so any failure replays.
  constexpr unsigned int seed = 20260907u;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> offsetDraw(-0.8, 0.8);
  std::uniform_real_distribution<double> angleDraw(-1.2, 1.2);
  std::uniform_real_distribution<double> radiusDraw(0.08, 0.35);
  std::uniform_real_distribution<double> rateDraw(-8.0, 8.0);
  std::uniform_real_distribution<double> scaleDraw(0.2, 5.0);
  constexpr size_t draws = 500;

  for(size_t draw = 0; draw < draws; ++draw)
  {
    const std::string where = "ORC-05 seed " + std::to_string(seed) + " draw " + std::to_string(draw);

    // Relation two: r_i -> alpha r_i together with thetaDot_i -> thetaDot_i / alpha
    // leaves the chassis twist the kinematic rows imply unchanged.
    std::vector<mc_rbdyn::PlanarWheel> wheels(4);
    Eigen::VectorXd rates(4);
    for(size_t i = 0; i < wheels.size(); ++i)
    {
      wheels[i].offset = Eigen::Vector2d(offsetDraw(generator), offsetDraw(generator));
      wheels[i].steeringAngle = angleDraw(generator);
      wheels[i].radius = radiusDraw(generator);
      rates(static_cast<Eigen::Index>(i)) = rateDraw(generator);
    }
    const Eigen::Vector3d twist =
        planarTwistFromWheelRates(mc_rbdyn::steeringRollingMatrix(wheels, Eigen::Vector3d::Zero()), rates);
    const double scale = scaleDraw(generator);
    auto scaled = wheels;
    for(auto & wheel : scaled) { wheel.radius *= scale; }
    const Eigen::VectorXd scaledRates = rates / scale;
    const Eigen::Vector3d scaledTwist =
        planarTwistFromWheelRates(mc_rbdyn::steeringRollingMatrix(scaled, Eigen::Vector3d::Zero()), scaledRates);
    BOOST_CHECK_MESSAGE((scaledTwist - twist).lpNorm<Eigen::Infinity>() <= 1e-6 * (1.0 + twist.norm()),
                        where << ": radius scaling moved the chassis twist by "
                              << (scaledTwist - twist).lpNorm<Eigen::Infinity>());

    // The same relation through the inverse map, which shares no code with the
    // rows: scaling the radius must scale the reference rate by 1/alpha and
    // leave the reference heading alone.
    const auto reference = mc_rbdyn::steeringWheelReference(wheels[0], twist, wheels[0].steeringAngle);
    const auto scaledReference = mc_rbdyn::steeringWheelReference(scaled[0], twist, scaled[0].steeringAngle);
    BOOST_CHECK_MESSAGE(std::abs(scaledReference.steeringAngle - reference.steeringAngle) < 1e-9,
                        where << ": radius scaling moved the reference heading");
    BOOST_CHECK_MESSAGE(std::abs(scale * scaledReference.rollingRate - reference.rollingRate)
                            <= 1e-6 * (1.0 + std::abs(reference.rollingRate)),
                        where << ": radius scaling did not scale the reference rate by 1/alpha");

    // Relation three: mirroring left and right mirrors the T1 solution. y flips
    // sign, so the two wheels exchange places and (v_y, omega) change sign.
    const double track = 0.2 + std::abs(offsetDraw(generator));
    const double leftRadius = radiusDraw(generator);
    const double rightRadius = radiusDraw(generator);
    Eigen::Vector2d wheelRates(rateDraw(generator), rateDraw(generator));
    const Eigen::Vector3d direct = planarTwistFromWheelRates(
        mc_rbdyn::differentialDriveRollingMatrix(track, leftRadius, rightRadius), wheelRates);
    const Eigen::Vector3d mirrored = planarTwistFromWheelRates(
        mc_rbdyn::differentialDriveRollingMatrix(track, rightRadius, leftRadius),
        Eigen::Vector2d(wheelRates.y(), wheelRates.x()));
    const Eigen::Vector3d expected(direct.x(), -direct.y(), -direct.z());
    BOOST_CHECK_MESSAGE((mirrored - expected).lpNorm<Eigen::Infinity>() <= 1e-6 * (1.0 + direct.norm()),
                        where << ": the mirrored T1 solution is [" << mirrored.transpose() << "], expected ["
                              << expected.transpose() << "]");
    BOOST_REQUIRE_GT(std::abs(direct.z()), 1e-12); // a zero-yaw draw would make the mirror vacuous
  }
}

BOOST_AUTO_TEST_CASE(ContactWrenchContainsWheelAxisMoment)
{
  const Eigen::Vector3d center(0.0, 0.0, 0.2);
  const Eigen::Vector3d point(0.0, 0.0, 0.0);
  const Eigen::Vector3d force(10.0, 0.0, 30.0);
  const auto wrench = mc_rbdyn::contactWrenchAtCarrier(center, point, force);
  checkVector(wrench.head<3>(), Eigen::Vector3d(0.0, -2.0, 0.0));
  checkVector(wrench.tail<3>(), force);
  BOOST_CHECK_CLOSE(wrench.head<3>().dot(Eigen::Vector3d::UnitY()), -0.2 * force.x(), 1e-12);
}
