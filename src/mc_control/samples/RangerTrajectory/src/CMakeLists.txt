#
# Copyright 2015-2026 CNRS-UM LIRMM, CNRS-AIST JRL
#

set(controller_SRC ${CMAKE_CURRENT_SOURCE_DIR}/mc_ranger_trajectory_controller.cpp)

set(controller_HDR ${CMAKE_CURRENT_SOURCE_DIR}/mc_ranger_trajectory_controller.h)

add_fsm_controller(ranger_trajectory ${controller_SRC} ${controller_HDR})
