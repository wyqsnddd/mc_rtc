/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include <mc_tvm/RollingContactFunction.h>

#include <mc_tvm/Robot.h>

#include <mc_rbdyn/Robot.h>

#include <stdexcept>

namespace mc_tvm
{

RollingContactFunction::RollingContactFunction(const mc_rbdyn::Robot & robot, Eigen::Index rows)
: tvm::function::abstract::LinearFunction(static_cast<int>(rows)), robot_(robot),
  matrix_(Eigen::MatrixXd::Zero(rows, robot.mb().nrDof())), rhs_(Eigen::VectorXd::Zero(rows))
{
  if(rows <= 0) { throw std::invalid_argument("RollingContactFunction requires at least one row"); }
  registerUpdates(Update::Jacobian, &RollingContactFunction::updateJacobian, Update::B,
                  &RollingContactFunction::updateB);
  addOutputDependency<RollingContactFunction>(Output::Jacobian, Update::Jacobian);
  addOutputDependency<RollingContactFunction>(Output::B, Update::B);
  addVariable(robot.tvmRobot().alphaD(), true);
  velocity_.setZero();
}

void RollingContactFunction::set(const Eigen::Ref<const Eigen::MatrixXd> & matrix,
                                 const Eigen::Ref<const Eigen::VectorXd> & rhs)
{
  if(matrix.rows() != size() || matrix.cols() != robot_.mb().nrDof() || rhs.size() != size())
  {
    throw std::invalid_argument("RollingContactFunction coefficient dimensions changed");
  }
  if(!matrix.allFinite() || !rhs.allFinite())
  {
    throw std::invalid_argument("RollingContactFunction coefficients must be finite");
  }
  matrix_ = matrix;
  rhs_ = rhs;
  // These coefficients are supplied by a solver-independent kernel rather
  // than another TVM graph node. Update the active caches immediately so a
  // mode transition or terrain change is visible in the very next solve.
  jacobian_[robot_.tvmRobot().alphaD().get()] = matrix_;
  b_ = -rhs_;
}

void RollingContactFunction::updateJacobian()
{
  jacobian_[robot_.tvmRobot().alphaD().get()] = matrix_;
}

void RollingContactFunction::updateB()
{
  b_ = -rhs_;
}

} // namespace mc_tvm
