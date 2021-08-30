cmake_minimum_required(VERSION 3.16)
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

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.22
    )
FetchContent_MakeAvailable(dlib)

add_executable(${PROJECT_NAME} src/main.cpp
    src/tvm_blazeface.cpp
    src/tvm_facemesh.cpp
    src/dlib_face_detection.cpp
    src/opencv_face_detection.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_sdl.cpp
    imgui/backends/imgui_impl_opengl2.cpp
    implot/implot.cpp
    implot/implot_items.cpp
    ${TVM_SRC}/apps/howto_deploy/tvm_runtime_pack.cc)

target_compile_definitions(${PROJECT_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TVM_EXPORTS)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "imgui")
target_include_directories(${PROJECT_NAME} PUBLIC "imgui/backends")
target_include_directories(${PROJECT_NAME} PUBLIC "implot")
target_include_directories(${PROJECT_NAME} PUBLIC "tvm/include")
target_include_directories(${PROJECT_NAME} PUBLIC ${TVM_SRC}/3rdparty/dlpack/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${TVM_SRC}/3rdparty/dmlc-core/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpencV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/spdlog/include)
target_include_directories(${PROJECT_NAME} PUBLIC dlib::dlib)

target_link_libraries(${PROJECT_NAME} PUBLIC ${SDL2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC dlib::dlib)

if(NOT WIN32)
target_link_libraries(${PROJECT_NAME} PUBLIC "stdc++fs")
endif()

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models/ $<TARGET_FILE_DIR:${PROJECT_NAME}>)

