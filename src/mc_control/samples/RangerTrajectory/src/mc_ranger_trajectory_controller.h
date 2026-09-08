/*
 * Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
 */

#pragma once

#include <mc_control/api.h>
#include <mc_control/fsm/Controller.h>
#include <mc_control/mc_controller.h>

/** Drive the Ranger Mini V3 along a circle, a straight line and a Lissajous
 * curve, chained by an FSM that is entirely described in YAML.
 *
 * Everything that makes this controller do something lives in
 * etc/RangerTrajectory.in.yaml: the rolling-contact constraints on the four
 * steering wheels, the observer pipeline, one `bspline_trajectory` task per
 * curve and the transitions between them. This class exists only to name the
 * controller, pin the supported robot and hand the configuration to
 * mc_control::fsm::Controller -- deliberately, since the point of the sample is
 * that a mobile-base trajectory scenario needs no C++ of its own.
 */
struct MC_CONTROL_DLLAPI RangerTrajectoryController : public mc_control::fsm::Controller
{
  RangerTrajectoryController(mc_rbdyn::RobotModulePtr rm,
                             double dt,
                             const mc_rtc::Configuration & config,
                             Backend backend);

  void supported_robots(std::vector<std::string> & out) const override { out = {"ranger_mini_v3"}; }
};
