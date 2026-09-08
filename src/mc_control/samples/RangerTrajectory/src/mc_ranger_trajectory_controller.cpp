/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#include "mc_ranger_trajectory_controller.h"

RangerTrajectoryController::RangerTrajectoryController(mc_rbdyn::RobotModulePtr rm,
                                                       double dt,
                                                       const mc_rtc::Configuration & config,
                                                       Backend backend)
: mc_control::fsm::Controller(rm, dt, config, backend)
{
}
