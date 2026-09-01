# Compatibility package for the Ubuntu 24.04 validation host.
#
# The installed ndcurves package is header-only and its imported target links
# only Eigen3::Eigen and Boost::serialization, but ndcurvesConfig.cmake still
# calls find_dependency(eigenpy). The rolling-contact runner does not compile
# or link Python bindings. This empty imported target prevents an unrelated
# Conda eigenpy/Boost.Python installation from contaminating the CPU runner.
if(NOT TARGET eigenpy::eigenpy)
  add_library(eigenpy::eigenpy INTERFACE IMPORTED)
endif()

set(eigenpy_FOUND TRUE)
