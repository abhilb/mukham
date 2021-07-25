cmake_minimum_required(VERSION 3.13)
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Relase")
    set(CMAKE_BUILD_TYPE "Release")
endif()

project(Mukham VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(Mukham main.cpp
		      imgui/imgui.cpp
		      imgui/imgui_demo.cpp
		      imgui/imgui_draw.cpp
		      imgui/imgui_tables.cpp
		      imgui/imgui_widgets.cpp
		      imgui/backends/imgui_impl_sdl.cpp
		      imgui/backends/imgui_impl_opengl2.cpp)

target_compile_definitions(Mukham PUBLIC DMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
target_include_directories(Mukham PUBLIC "imgui")
target_include_directories(Mukham PUBLIC "imgui/backends")
target_include_directories(Mukham PUBLIC "tvm/include")
target_include_directories(Mukham PUBLIC "dlpack/include")
target_include_directories(Mukham PUBLIC "dmlc-core/include")
target_include_directories(Mukham PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(Mukham PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(Mukham PUBLIC ${OpencV_INCLUDE_DIRS})

target_link_directories(Mukham PUBLIC ${CMAKE_SOURCE_DIR}/tvm/libs)

target_link_libraries(Mukham PUBLIC ${SDL2_LIBRARIES})
target_link_libraries(Mukham PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(Mukham PUBLIC ${OpenCV_LIBS})
target_link_libraries(Mukham PUBLIC tvm_runtime)
