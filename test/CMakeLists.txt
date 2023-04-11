SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/build/test")

# set(CMAKE_EXE_LINKER_FLAGS "-static")
include_directories("${PROJECT_SOURCE_DIR}/yhccl_allreduce_pjt")
include_directories("/usr/local/glex/include")
link_directories("${PROJECT_BINARY_DIR}/build/lib/" "/usr/local/glex/lib" "/public3/home/sc53841/papi-6.0.0/build/lib/")
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/test TESTSRC)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/test/baidu-allreduce TESTSRC)
add_compile_options(-O0)
add_compile_options(-std=c99)	
add_definitions(-w)
FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile})
    target_link_libraries(${mainname} -lmpi -lyhccl -lpthread -lm -lrt -lstdc++ ${NUMALIB} ${PAPILIB})

    # target_link_libraries(${mainname} libic.a)
    target_link_libraries(${mainname} -lrt -lmpi -lpthread -lrt -lm)
endforeach()

file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
add_compile_options(-std=c++11)

foreach(mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile})
    target_link_libraries(${mainname} -lyhccl -lpthread -lm -lrt -lstdc++ ${NUMALIB} ${PAPILIB})

    # target_link_libraries(${mainname}  -lyhccl -lpthread -lm -lrt -lstdc++)
    # target_link_libraries(${mainname} -lrt -lglex -lmpi -lmpi_cxx -lglex  -lpthread -lrt -lm -lstdc++) ${MPI_LIBRARY}
    # target_link_libraries(${mainname} -lrt -lglex -lmpi   -lpthread -lrt -lm -lstdc++)
endforeach()
