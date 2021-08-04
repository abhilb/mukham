cmake_minimum_required(VERSION 3.13)
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

project(Mukham VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

set(TVM_SRC ${CMAKE_SOURCE_DIR}/tvm)

add_executable(${PROJECT_NAME} src/main.cpp
		      imgui/imgui.cpp
		      imgui/imgui_demo.cpp
		      imgui/imgui_draw.cpp
		      imgui/imgui_tables.cpp
		      imgui/imgui_widgets.cpp
		      imgui/backends/imgui_impl_sdl.cpp
		      imgui/backends/imgui_impl_opengl2.cpp
              ${TVM_SRC}/apps/howto_deploy/tvm_runtime_pack.cc)

target_compile_definitions(${PROJECT_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
target_include_directories(${PROJECT_NAME} PUBLIC "imgui")
target_include_directories(${PROJECT_NAME} PUBLIC "imgui/backends")
target_include_directories(${PROJECT_NAME} PUBLIC "tvm/include")
target_include_directories(${PROJECT_NAME} PUBLIC ${TVM_SRC}/3rdparty/dlpack/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${TVM_SRC}/3rdparty/dmlc-core/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpencV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/spdlog/include)

target_link_libraries(${PROJECT_NAME} PUBLIC ${SDL2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
