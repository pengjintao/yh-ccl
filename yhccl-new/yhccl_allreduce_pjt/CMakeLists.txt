SET(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/build/lib")

add_compile_options(-O3)

# add_compile_options(-O1)

# add_compile_options(-fcoroutines)
add_compile_options(-std=c++11)
add_compile_options(-fopenmp)
add_compile_options(-fpermissive)
add_definitions(-w)
add_compile_options(-fpermissive)
set(CMAKE_C_FLAGS "-fPIC")

include_directories("${PROJECT_SOURCE_DIR}/yhccl_allreduce_pjt")

# include_directories("/usr/local/glex/include")
# link_directories("/usr/local/glex/lib/")
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/yhccl_allreduce_pjt GLEXCOLL_SRC)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/yhccl_allreduce_pjt/allreduce_algorithms GLEXCOLL_SRC)
add_library(yhccl SHARED ${GLEXCOLL_SRC})
add_library(yhccl_static STATIC ${GLEXCOLL_SRC})
set_target_properties(yhccl_static PROPERTIES OUTPUT_NAME "yhccl")
set_target_properties(yhccl PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(yhccl_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
