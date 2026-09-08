/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "mc_ranger_trajectory_controller.h"

MULTI_CONTROLLERS_CONSTRUCTOR("RangerTrajectory",
                              RangerTrajectoryController(rm, dt, config, mc_control::MCController::Backend::Tasks),
                              "RangerTrajectory_TVM",
                              RangerTrajectoryController(rm, dt, config, mc_control::MCController::Backend::TVM))
