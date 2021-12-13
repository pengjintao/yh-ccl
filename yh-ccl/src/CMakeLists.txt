SET(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/build/lib")
add_compile_options(-O3)
add_compile_options(-std=c++11)
add_compile_options(-fopenmp)
add_definitions(-w)
add_compile_options(-fpermissive)
set(CMAKE_C_FLAGS "-fPIC")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("/usr/local/glex/include")
link_directories("/usr/local/glex/lib/")
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src GLEXCOLL_SRC)
add_library(yhccl STATIC ${GLEXCOLL_SRC})  

                       