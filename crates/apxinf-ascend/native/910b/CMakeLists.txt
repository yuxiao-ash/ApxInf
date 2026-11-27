cmake_minimum_required(VERSION 3.16)
project(apxinf_910b_fusions LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)
if(NOT SOC_VERSION MATCHES "^Ascend910B")
  message(FATAL_ERROR "The 910B fusion library requires SOC_VERSION=Ascend910B...")
endif()
# CANN's top-level package root, before the aarch64-linux component.
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(apxinf_910b_fusions SHARED fusions.cpp)
