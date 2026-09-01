/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_tvm/api.h>

#include <mc_rbdyn/fwd.h>

#include <tvm/function/abstract/LinearFunction.h>

namespace mc_tvm
{

/** Acceleration-level rolling rows over an mc_tvm robot alphaD variable.
 *
 * The solver-independent rolling kernel supplies A and b in A * alphaD = b
 * form. This function exposes the equivalent TVM linear function
 * A * alphaD - b, allowing the same coefficients to be inserted as a hard
 * requirement or a weighted least-squares task.
 */
struct MC_TVM_DLLAPI RollingContactFunction : tvm::function::abstract::LinearFunction
{
  SET_UPDATES(RollingContactFunction, Jacobian, B)

  RollingContactFunction(const mc_rbdyn::Robot & robot, Eigen::Index rows);

  void set(const Eigen::Ref<const Eigen::MatrixXd> & matrix, const Eigen::Ref<const Eigen::VectorXd> & rhs);

  const Eigen::MatrixXd & matrix() const noexcept { return matrix_; }
  const Eigen::VectorXd & rhs() const noexcept { return rhs_; }

private:
  void updateJacobian();
  void updateB();

  const mc_rbdyn::Robot & robot_;
  Eigen::MatrixXd matrix_;
  Eigen::VectorXd rhs_;
};

using RollingContactFunctionPtr = std::shared_ptr<RollingContactFunction>;

} // namespace mc_tvm
