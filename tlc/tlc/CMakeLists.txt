cmake_minimum_required(VERSION 3.16)
project(tlc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TLC_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(TLC_LIBFUZZER "Build libFuzzer target (clang)" OFF)
option(TLC_PROFILE "Build with -pg for gprof" OFF)
option(TLC_BUILD_ANALOGS "Build analog adapters (needs Eigen3, Qt for analog1)" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)
if(TLC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()
if(TLC_PROFILE)
  add_compile_options(-pg -g)
  add_link_options(-pg)
endif()

# ---------- библиотека ----------
add_library(tlc src/tropical.cpp src/solver.cpp)
target_include_directories(tlc PUBLIC include)

add_executable(tlc_cli apps/tlc_cli.cpp)
target_link_libraries(tlc_cli tlc)
set_target_properties(tlc_cli PROPERTIES OUTPUT_NAME tlc)

# ---------- тесты, фаззер, бенчмарк ----------
enable_testing()
add_executable(tlc_tests tests/test_main.cpp)
target_link_libraries(tlc_tests tlc)
add_test(NAME unit COMMAND tlc_tests ${CMAKE_SOURCE_DIR}/data)

add_executable(tlc_fuzz fuzz/fuzz_tlc.cpp)
target_link_libraries(tlc_fuzz tlc)
add_test(NAME fuzz_smoke COMMAND tlc_fuzz 5000 1)

if(TLC_LIBFUZZER)
  add_executable(tlc_libfuzzer fuzz/fuzz_tlc.cpp)
  target_compile_definitions(tlc_libfuzzer PRIVATE TLC_LIBFUZZER)
  target_compile_options(tlc_libfuzzer PRIVATE -fsanitize=fuzzer,address)
  target_link_options(tlc_libfuzzer PRIVATE -fsanitize=fuzzer,address)
  target_link_libraries(tlc_libfuzzer tlc)
endif()

add_executable(bench_kernels bench/bench_kernels.cpp src/tropical.cpp)
target_include_directories(bench_kernels PRIVATE include)

# ---------- аналоги (собираются на машине, где есть их исходники) ----------
# -DTLC_BUILD_ANALOGS=ON -DANALOG1_DIR=/path/to/analog1 -DANALOG2_DIR=/path/to/analog2
if(TLC_BUILD_ANALOGS)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  if(ANALOG2_DIR)
    # cli.cpp ожидает ../lib/struct_a.h относительно себя, поэтому собираем копию в дереве аналога
    configure_file(analogs/analog2_maxalgebra_cli/cli.cpp ${ANALOG2_DIR}/cli/cli_stand.cpp COPYONLY)
    file(GLOB ANALOG2_LIB_SRC ${ANALOG2_DIR}/lib/*.cpp)
    add_executable(analog2_cli ${ANALOG2_DIR}/cli/cli_stand.cpp ${ANALOG2_LIB_SRC})
    target_include_directories(analog2_cli PRIVATE include ${ANALOG2_DIR}/lib)
    target_link_libraries(analog2_cli Eigen3::Eigen)
  endif()
  if(ANALOG1_DIR)
    find_package(Qt6 COMPONENTS Core QUIET)
    if(NOT Qt6_FOUND)
      find_package(Qt5 COMPONENTS Core REQUIRED)
      set(QTCORE Qt5::Core)
    else()
      set(QTCORE Qt6::Core)
    endif()
    # все .cpp аналога, кроме GUI (main.cpp, окна) — подставляется исправленный decision_model.cpp
    file(GLOB_RECURSE ANALOG1_SRC ${ANALOG1_DIR}/*.cpp)
    list(FILTER ANALOG1_SRC EXCLUDE REGEX "(main|decision_model|StartMenu|Window|Dialog|Widget)\\.cpp$")
    add_executable(analog1_cli analogs/analog1_decision_model/headless_cli.cpp
                               analogs/analog1_decision_model/decision_model.cpp ${ANALOG1_SRC})
    target_include_directories(analog1_cli PRIVATE include ${ANALOG1_DIR} ${ANALOG1_DIR}/include)
    target_link_libraries(analog1_cli Eigen3::Eigen ${QTCORE})
  endif()
endif()
