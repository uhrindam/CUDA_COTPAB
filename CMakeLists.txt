cmake_minimum_required(VERSION 3.3.0)
project(COTPAB)

find_package(OpenCV 3.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h*)

cuda_add_executable(COTPAB ${SOURCE_FILES} "COTPAB/kernel.cu")
target_link_libraries(COTPAB ${OpenCV_LIBS})