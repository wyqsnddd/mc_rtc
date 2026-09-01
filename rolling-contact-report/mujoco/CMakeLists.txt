cmake_minimum_required(VERSION 3.22)

project(rolling_contact_mujoco_runner LANGUAGES CXX)

find_package(mc_mujoco REQUIRED)

add_executable(rolling_contact_mujoco_runner rolling_contact_mujoco_runner.cpp)
target_compile_features(rolling_contact_mujoco_runner PRIVATE cxx_std_17)
target_link_libraries(rolling_contact_mujoco_runner PRIVATE mc_mujoco::mc_mujoco)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rolling_contact_mujoco_runner PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
