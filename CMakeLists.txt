cmake_minimum_required(VERSION 3.18) # Required for CUDA language support
project(chfem LANGUAGES C CUDA)

# Set the C and CUDA standard
set(CMAKE_C_STANDARD 99)

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

# Optional: specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Include directories
include_directories("chfem")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${Python_INCLUDE_DIRS})
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NUMPY_INCLUDE_DIR})

# File GLOB_RECURSE to collect all the source files
file(GLOB_RECURSE CUDA_SOURCES "chfem/src/**/*.cu")
file(GLOB_RECURSE C_SOURCES "chfem/src/**/*.c")

# chfem_exec
set(MAIN_SOURCE "chfem/src/main.c")
add_executable(chfem_exec ${CUDA_SOURCES} ${C_SOURCES} ${MAIN_SOURCE})
target_compile_options(chfem_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${OpenMP_C_FLAGS}>)
target_link_libraries(chfem_exec PUBLIC OpenMP::OpenMP_C)
set_target_properties(chfem_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_NAME "chfem_exec")

# wrapper library python API
set(WRAPPER_SOURCE "chfem/src/chfem_wrapper.c")
add_library(wrapper SHARED ${CUDA_SOURCES} ${C_SOURCES} ${MAIN_SOURCE} ${WRAPPER_SOURCE})
target_compile_options(wrapper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${OpenMP_C_FLAGS}>)
target_compile_options(wrapper PRIVATE $<$<COMPILE_LANGUAGE:C>:-fopenmp>)
target_link_libraries(wrapper PUBLIC OpenMP::OpenMP_C)
set_target_properties(wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(wrapper PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/chfem")
set_target_properties(wrapper PROPERTIES PREFIX "")  # avoid the "lib" prefix
set_target_properties(wrapper PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/chfem" OUTPUT_NAME "wrapper.cpython-312-x86_64-linux-gnu")

# chfem_test
set(TEST_SOURCE "test/test_chfem.c")
add_executable(chfem_test ${CUDA_SOURCES} ${C_SOURCES} ${TEST_SOURCE})
target_compile_options(chfem_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${OpenMP_C_FLAGS}>)
target_link_libraries(chfem_test PUBLIC OpenMP::OpenMP_C)
set_target_properties(chfem_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_NAME "chfem_test")
