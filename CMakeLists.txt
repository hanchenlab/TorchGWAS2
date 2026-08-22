# =============================================================================
# TorchGWAS2 — CMake build configuration
# =============================================================================
# Three supported ways to build, all using THIS SAME FILE:
#
#   1. Docker (recommended, zero setup required)
#        See Dockerfile. All dependencies land in fixed, known locations
#        (oneAPI MKL, apt-installed Boost/GMP/MPFR, SuiteSparse/Eigen/
#        Armadillo under /usr/local) which are auto-detected below.
#
#   2. Conda environment
#        conda env create -f environment.yml
#        conda activate torchgwas2
#        mkdir build && cd build && cmake .. && make -j$(nproc)
#        The active $CONDA_PREFIX is auto-detected and searched first.
#
# Every dependency location below is a CMake cache variable with an
# auto-detect default, so none of these flags are required unless
# auto-detection fails for your setup.
# =============================================================================

cmake_minimum_required(VERSION 3.14)

# ---- Compiler ---------------------------------------------------------------
# GCC 13 is required for the C++20 features used in this project.
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc-13)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++-13)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(GEM)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(FetchContent)

# If a conda environment is active, search it first for everything
# (headers, libraries, cmake config files) before falling back to system
# paths. This is what makes build mode (2) work with zero extra flags.
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Detected active conda environment: $ENV{CONDA_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# ---- Sources ------------------------------------------------------------
file(GLOB SRC "src/*.cpp")
# plink-2.0 pgenlib sources are no longer compiled — BED reading uses
# standard C file I/O only (no LGPL dependency).
file(GLOB C_SOURCES
    "thirdparty/zstd-1.5.5/lib/common/*.c"
    "thirdparty/zstd-1.5.5/lib/compress/*.c"
    "thirdparty/zstd-1.5.5/lib/decompress/*.c"
    "thirdparty/libdeflate-1.18/lib/*.c"
    "thirdparty/libdeflate-1.18/lib/x86/*.c"
    "thirdparty/libdeflate-1.18/lib/arm/*.c"
)
set(SOURCES ${SRC} ${C_SOURCES})

# ---- fmt / spdlog / pybind11 -----------------------------------------------
# Fetched from source in all three build modes — small, fast, and avoids
# version drift between Docker/conda/manual setups.
set(FMT_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
set(FMT_SHARED OFF CACHE BOOL "" FORCE)
set(FMT_HEADER_ONLY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.0.2
)
FetchContent_MakeAvailable(fmt)

set(SPDLOG_FMT_EXTERNAL ON  CACHE BOOL "" FORCE)   # use fmt::fmt, not spdlog's bundled fmt
set(SPDLOG_BUILD_SHARED  OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH   OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC     ON  CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)

set(PYBIND11_PYTHON_VERSION "" CACHE STRING "Python version to build the pybind11 module against (leave empty to auto-detect whichever python3 is active/installed)")
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ---- Eigen (header-only) ----------------------------------------------------
# Search order: -DEIGEN_INCLUDE_DIR override -> conda/system install -> fetch from source.
set(EIGEN_INCLUDE_DIR "" CACHE PATH "Directory containing the Eigen/ headers (set if auto-detect fails)")
if(EIGEN_INCLUDE_DIR)
    message(STATUS "Using user-specified Eigen at ${EIGEN_INCLUDE_DIR}")
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE ${EIGEN_INCLUDE_DIR})
else()
    find_path(EIGEN_SYSTEM_DIR NAMES Eigen/Dense PATH_SUFFIXES eigen3)
    if(EIGEN_SYSTEM_DIR)
        message(STATUS "Found system/conda Eigen at ${EIGEN_SYSTEM_DIR}")
        add_library(eigen INTERFACE)
        target_include_directories(eigen INTERFACE ${EIGEN_SYSTEM_DIR})
    else()
        message(STATUS "Eigen not found locally — fetching 3.4.0 from source")
        FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen
            GIT_TAG 3.4.0
        )
        FetchContent_GetProperties(eigen)
        if(NOT eigen_POPULATED)
            FetchContent_Populate(eigen)
        endif()
        add_library(eigen INTERFACE)
        target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})
    endif()
endif()

# ---- Armadillo (header-only, linked directly against MKL/LAPACK) -----------
set(ARMADILLO_INCLUDE_DIR "" CACHE PATH "Directory containing the armadillo header (set if auto-detect fails)")
if(ARMADILLO_INCLUDE_DIR)
    message(STATUS "Using user-specified Armadillo at ${ARMADILLO_INCLUDE_DIR}")
    add_library(armadillo INTERFACE)
    target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIR})
else()
    # Armadillo's header can live in one of two layouts depending on how it
    # was installed:
    #   - flat (apt/conda-forge):  <root>/armadillo            (a FILE)
    #   - nested (this Dockerfile): <root>/armadillo/armadillo  (a FILE one dir deeper,
    #                                because `cp -r .../include /usr/local/include/armadillo`
    #                                makes "armadillo" itself a directory)
    # find_path() alone can't tell files from directories, so check explicitly.
    set(_armadillo_search_roots /usr/local/include /usr/include)
    if(DEFINED ENV{CONDA_PREFIX})
        list(PREPEND _armadillo_search_roots "$ENV{CONDA_PREFIX}/include")
    endif()

    set(ARMADILLO_SYSTEM_DIR "")
    foreach(_root ${_armadillo_search_roots})
        if(EXISTS "${_root}/armadillo" AND NOT IS_DIRECTORY "${_root}/armadillo")
            set(ARMADILLO_SYSTEM_DIR "${_root}")
            break()
        elseif(EXISTS "${_root}/armadillo/armadillo")
            set(ARMADILLO_SYSTEM_DIR "${_root}/armadillo")
            break()
        endif()
    endforeach()

    if(ARMADILLO_SYSTEM_DIR)
        message(STATUS "Found system/conda Armadillo at ${ARMADILLO_SYSTEM_DIR}")
        add_library(armadillo INTERFACE)
        target_include_directories(armadillo INTERFACE ${ARMADILLO_SYSTEM_DIR})
    else()
        message(STATUS "Armadillo not found locally — fetching 14.0.x from source")
        FetchContent_Declare(armadillo
            GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code
            GIT_TAG 14.0.x
        )
        FetchContent_Populate(armadillo)
        add_library(armadillo INTERFACE)
        target_include_directories(armadillo INTERFACE ${armadillo_SOURCE_DIR}/include)
    endif()
endif()
target_compile_definitions(armadillo INTERFACE
    ARMA_USE_LAPACK
    ARMA_USE_BLAS
    ARMA_DONT_USE_WRAPPER
    ARMA_DONT_USE_OPENMP
)

# ---- Boost ------------------------------------------------------------------
# BOOST_ROOT (cache var, env var, or -D flag) is honored automatically by
# CMake's FindBoost module if set; otherwise it searches conda (via
# CMAKE_PREFIX_PATH above) and system paths (e.g. apt's libboost-*-dev).
set(BOOST_ROOT "" CACHE PATH "Root directory of a custom Boost install (leave empty to auto-detect)")
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "Link Boost statically")
find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options thread system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# ---- Intel MKL ----------------------------------------------------------
# Search order:
#   1. -DMKLROOT=... or $MKLROOT env var   (oneAPI layout: MKLROOT/lib/intel64)
#   2. conda 'mkl-devel' package           (CONDA_PREFIX/lib, flat layout)
#   3. /opt/intel/oneapi/mkl/latest        (default oneAPI location — matches Dockerfile)
if(NOT DEFINED MKLROOT AND DEFINED ENV{MKLROOT})
    set(MKLROOT $ENV{MKLROOT})
endif()

if(MKLROOT)
    message(STATUS "Using MKLROOT=${MKLROOT}")
    set(MKL_INCLUDE_DIR ${MKLROOT}/include)
    set(MKL_LIB_DIR ${MKLROOT}/lib/intel64)
elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/mkl.h")
    message(STATUS "Using conda-installed MKL at $ENV{CONDA_PREFIX}")
    set(MKL_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    set(MKL_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
else()
    set(MKLROOT "/opt/intel/oneapi/mkl/latest")
    message(STATUS "MKLROOT not set — defaulting to ${MKLROOT}")
    set(MKL_INCLUDE_DIR ${MKLROOT}/include)
    set(MKL_LIB_DIR ${MKLROOT}/lib/intel64)
endif()
message(STATUS "MKL include dir: ${MKL_INCLUDE_DIR}")
message(STATUS "MKL lib dir: ${MKL_LIB_DIR}")
include_directories(${MKL_INCLUDE_DIR})
link_directories(${MKL_LIB_DIR})
add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_BLAS_CAPITALS -DARMA_BLAS_UNDERSCORE)

find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
find_library(MKL_SEQUENTIAL NAMES mkl_sequential PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
find_library(MKL_CORE       NAMES mkl_core       PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
if(NOT MKL_INTEL_LP64 OR NOT MKL_SEQUENTIAL OR NOT MKL_CORE)
    message(FATAL_ERROR
        "Could not find MKL libraries in ${MKL_LIB_DIR}.\n"
        "  - Docker users: this should never happen, please open an issue.\n"
        "  - Conda users: make sure the environment is active (conda activate torchgwas2).\n"
        "  - Manual users: pass -DMKLROOT=/path/to/mkl at configure time.")
endif()
set(MKL_LIBRARIES ${MKL_INTEL_LP64} ${MKL_SEQUENTIAL} ${MKL_CORE} pthread m dl)
set(BLAS_LIBRARIES ${MKL_LIBRARIES})
set(LAPACK_LIBRARIES ${MKL_LIBRARIES})

# ---- GMP / MPFR ---------------------------------------------------------
set(GMP_ROOT "" CACHE PATH "Root directory of a custom GMP install (leave empty to auto-detect)")
set(MPFR_ROOT "" CACHE PATH "Root directory of a custom MPFR install (leave empty to auto-detect)")

find_path(GMP_INCLUDE_DIR gmp.h HINTS ${GMP_ROOT}/include)
find_library(GMP_LIBRARY NAMES gmp libgmp HINTS ${GMP_ROOT}/lib)
find_path(MPFR_INCLUDE_DIR mpfr.h HINTS ${MPFR_ROOT}/include)
find_library(MPFR_LIBRARY NAMES mpfr libmpfr HINTS ${MPFR_ROOT}/lib)

if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR
        "GMP not found. Install libgmp-dev (apt) / gmp (conda-forge), "
        "or pass -DGMP_ROOT=/path/to/gmp.")
endif()
if(NOT MPFR_INCLUDE_DIR OR NOT MPFR_LIBRARY)
    message(FATAL_ERROR
        "MPFR not found. Install libmpfr-dev (apt) / mpfr (conda-forge), "
        "or pass -DMPFR_ROOT=/path/to/mpfr.")
endif()
set(GMP_LIBRARIES ${GMP_LIBRARY})
set(MPFR_LIBRARIES ${MPFR_LIBRARY})
include_directories(${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR})
message(STATUS "Using GMP:  ${GMP_LIBRARY}")
message(STATUS "Using MPFR: ${MPFR_LIBRARY}")

# ---- SuiteSparse ----------------------------------------------------------
# Default search matches a standard installed layout — <prefix>/include or
# <prefix>/include/suitesparse, plus <prefix>/lib — which is what both
# `make install` (Docker) and the conda-forge 'suitesparse' package produce.
# Pass -DSUITESPARSE_ROOT=/path for anything else (e.g. an in-tree build dir).
set(SUITESPARSE_ROOT "" CACHE PATH "Root directory of a custom SuiteSparse install (leave empty to auto-detect)")

find_path(SUITESPARSE_INCLUDE_DIR NAMES cholmod.h
    HINTS ${SUITESPARSE_ROOT}/include
    PATH_SUFFIXES suitesparse)

set(_suitesparse_libnames cholmod cxsparse spqr umfpack camd ccolamd colamd amd suitesparseconfig)
set(SUITESPARSE_LIBRARIES "")
foreach(_lib ${_suitesparse_libnames})
    find_library(_suitesparse_${_lib} NAMES ${_lib} HINTS ${SUITESPARSE_ROOT}/lib)
    if(_suitesparse_${_lib})
        list(APPEND SUITESPARSE_LIBRARIES ${_suitesparse_${_lib}})
    endif()
endforeach()

if(NOT SUITESPARSE_INCLUDE_DIR OR NOT SUITESPARSE_LIBRARIES)
    message(FATAL_ERROR
        "SuiteSparse not found. Build/install it (see Dockerfile for the exact recipe), "
        "install the conda-forge 'suitesparse' package, "
        "or pass -DSUITESPARSE_ROOT=/path/to/suitesparse.")
endif()
message(STATUS "SuiteSparse include: ${SUITESPARSE_INCLUDE_DIR}")
message(STATUS "SuiteSparse libs: ${SUITESPARSE_LIBRARIES}")

# ---- Build the pybind11 module ------------------------------------------
add_library(Mygen MODULE ${SOURCES})
target_include_directories(Mygen PRIVATE ${SUITESPARSE_INCLUDE_DIR})
target_include_directories(Mygen PRIVATE ${pybind11_INCLUDE_DIRS})
message(STATUS "pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")

target_link_libraries(Mygen PRIVATE
    pybind11::module
    fmt
    spdlog::spdlog
    eigen
    armadillo
    Boost::thread
    Boost::system
    Boost::filesystem
    Boost::program_options
    ${SUITESPARSE_LIBRARIES}
    ${GMP_LIBRARIES}
    ${MPFR_LIBRARIES}
    ${MKL_LIBRARIES}
    m
    dl
)

set_target_properties(Mygen PROPERTIES
    PREFIX ""  # remove the default 'lib' prefix so Python can import it directly
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pymodules"
)

# ---- Optimization flags ------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -ffast-math -fno-math-errno -DNDEBUG")
