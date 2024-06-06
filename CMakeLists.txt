cmake_minimum_required(VERSION 3.27.0)

project(gtfn_perf)

set(GT_NEIGHBOR_FIX CACHE BOOL "Use code reorganization for gtfn horizontal_shift")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_EXTENSIONS OFF)
else()
  message( FATAL_ERROR "CUDA compiler not found." )
endif()

# Find or fetch GridTools
if(NOT _gridtools_already_fetched)
    find_package(GridTools "2" QUIET)
endif()

if (GT_NEIGHBOR_FIX)
  set(GT_TAG "neighbor_read_perf_improvement")
else()
  set(GT_TAG "master")
endif()

if(NOT GridTools_FOUND)
  set(_gridtools_repository "https://github.com/iomaganaris/gridtools.git")
  set(_gridtools_tag        "${GT_TAG}")
  if(NOT _gridtools_already_fetched)
    message(STATUS "Fetching GridTools ${_gridtools_tag} from ${_gridtools_repository}")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    gridtools
    GIT_REPOSITORY ${_gridtools_repository}
    GIT_TAG        ${_gridtools_tag}
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(gridtools)
  set(_gridtools_already_fetched ON CACHE INTERNAL "")
endif()

find_package(Python 3.9.6 COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind module
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(copy_neighbor_gtfn copy_neighbor_gtfn_bindings.cpp)
target_include_directories(copy_neighbor_gtfn PRIVATE include GridTools)
target_link_libraries(copy_neighbor_gtfn PUBLIC GridTools::gridtools GridTools::stencil_naive GridTools::storage_gpu)
set_source_files_properties(copy_neighbor_gtfn_bindings.cpp PROPERTIES LANGUAGE CUDA)
