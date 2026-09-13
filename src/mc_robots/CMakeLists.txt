#
# Copyright 2015-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

macro(add_robot robot_base)
  set(robot_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${robot_base}.cpp")
  set(robot_HDR "${MC_RTC_SOURCE_DIR}/include/mc_robots/api.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/${robot_base}.h"
  )
  if(MC_RTC_BUILD_STATIC)
    target_sources(mc_rbdyn PRIVATE ${robot_SRC} ${robot_HDR})
    target_include_directories(mc_rbdyn PRIVATE "../../include/mc_robots")
  else()
    add_library(${robot_base} SHARED ${robot_SRC} ${robot_HDR})
    set_target_properties(
      ${robot_base} PROPERTIES COMPILE_FLAGS "-DMC_ROBOTS_EXPORTS" PREFIX ""
    )
    set_target_properties(${robot_base} PROPERTIES FOLDER robots)
    target_link_libraries(${robot_base} PUBLIC mc_rbdyn)
    target_include_directories(${robot_base} PRIVATE "../../include/mc_robots")
    install(
      TARGETS ${robot_base}
      ARCHIVE
        DESTINATION
          "${MC_ROBOTS_LIBRARY_INSTALL_PREFIX}$<$<CONFIG:debug>:${MC_RTC_LOADER_DEBUG_SUFFIX}>"
      LIBRARY
        DESTINATION
          "${MC_ROBOTS_LIBRARY_INSTALL_PREFIX}$<$<CONFIG:debug>:${MC_RTC_LOADER_DEBUG_SUFFIX}>"
      RUNTIME
        DESTINATION
          "${MC_ROBOTS_RUNTIME_INSTALL_PREFIX}$<$<CONFIG:debug>:${MC_RTC_LOADER_DEBUG_SUFFIX}>"
    )
  endif()
endmacro()

add_robot(env)
add_robot(json)

macro(append_aliases PATH VAR NAME)
  file(GLOB URDF_FILES ${PATH}/urdf/*.urdf)
  foreach(URDF ${URDF_FILES})
    get_filename_component(URDF_NAME "${URDF}" NAME_WE)
    list(APPEND ${VAR} "${NAME}/${URDF_NAME}: [${NAME}, \"${PATH}\", ${URDF_NAME}]")
  endforeach()
  string(REPLACE ";" "\n" ${VAR} "${${VAR}}")
endmacro()

macro(generate_aliases PATH ALIASES FILE_IN FILE_OUT)
  set(${ALIASES})
  set(PATH_OUT "${CMAKE_CURRENT_BINARY_DIR}/aliases/${FILE_OUT}")
  configure_file("${FILE_IN}" "${PATH_OUT}")
  if(CMAKE_CONFIGURATION_TYPES)
    file(
      GENERATE
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>/aliases/${FILE_OUT}"
      INPUT "${PATH_OUT}"
    )
  endif()
  install(FILES "${PATH_OUT}"
          DESTINATION "${MC_ROBOTS_RUNTIME_INSTALL_PREFIX}/aliases/"
  )
endmacro()

if(EXISTS "${MC_INT_OBJ_DESCRIPTION_PATH}")
  append_aliases("${MC_INT_OBJ_DESCRIPTION_PATH}" OBJECT_ALIASES object)
  generate_aliases(
    "${MC_INT_OBJ_DESCRIPTION_PATH}" ${OBJECT_ALIASES} object_aliases.in.yml
    object_aliases.yml
  )
  append_aliases("${MC_INT_OBJ_DESCRIPTION_PATH}" ENV_ALIASES env)
endif()

if(EXISTS "${MC_ENV_DESCRIPTION_PATH}")
  append_aliases("${MC_ENV_DESCRIPTION_PATH}" ENV_ALIASES env)
  generate_aliases(
    "${MC_ENV_DESCRIPTION_PATH}" ${ENV_ALIASES} env_aliases.in.yml env_aliases.yml
  )
endif()

add_robot(jvrc1)
add_robot(ranger_mini_v3)

set(RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/ranger_mini_v3_description"
)
set(RANGER_MINI_V3_INSTALL_PATH
    "${CMAKE_INSTALL_FULL_DATADIR}/mc_rtc/ranger_mini_v3"
)
set(RANGER_MINI_V3_DESCRIPTION_PATH
    "${RANGER_MINI_V3_DESCRIPTION_SOURCE_PATH}"
)
configure_file(
  ranger_mini_v3_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/aliases/ranger_mini_v3.yml"
  @ONLY
)
set(RANGER_MINI_V3_DESCRIPTION_PATH "${RANGER_MINI_V3_INSTALL_PATH}")
configure_file(
  ranger_mini_v3_aliases.in.yml
  "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/ranger_mini_v3.yml"
  @ONLY
)
install(
  DIRECTORY ranger_mini_v3_description/
  DESTINATION "${CMAKE_INSTALL_DATADIR}/mc_rtc/ranger_mini_v3"
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/install-aliases/ranger_mini_v3.yml"
  DESTINATION "${MC_ROBOTS_RUNTIME_INSTALL_PREFIX}/aliases"
)
if(TARGET jvrc1)
  set_target_properties(jvrc1 PROPERTIES COMPILE_FLAGS "-DMC_ROBOTS_EXPORTS")
else()
  set_target_properties(mc_rbdyn PROPERTIES COMPILE_FLAGS "-DMC_ROBOTS_EXPORTS")
endif()
