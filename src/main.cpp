
#include <SDL.h>
#include <SDL_opengl.h>
#include <stdio.h>

#include <algorithm>
#include <deque>
#include <filesystem>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <utility>

#include "imgui.h"
#include "imgui_impl_opengl2.h"
#include "imgui_impl_sdl.h"
#include "implot.h"
#include "spdlog/spdlog.h"
#include "tvm_blazeface.h"
#include "tvm_facemesh.h"

unsigned int display_image_width = 512;
unsigned int display_image_height = 512;

namespace fs = std::filesystem;

struct RollingBuffer {
    int _history;
    std::vector<float> y;
    std::vector<float> x;
    RollingBuffer(int history) {
        _history = history;
        y.reserve(history);
        x = std::vector<float>(history);
        std::iota(x.begin(), x.end(), 0);
    }

    void AddPoint(float data) {
        y.push_back(data);
        if (y.size() > _history) y.erase(y.begin());
    }
};

void LoadImage(const unsigned char *image_data, const int &height,
               const int &width, GLuint *texture) {
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, image_data);
    *texture = image_texture;
}

void RenderImage(const cv::Mat &frame, const int &width, const int &height) {
    cv::Mat display_frame;
    cv::resize(frame, display_frame, cv::Size(width, height), cv::INTER_LINEAR);

    // LOad image into texture
    GLuint image_texture;
    LoadImage(display_frame.ptr(), display_image_width, display_image_height,
              &image_texture);
    ImGui::Image((void *)(intptr_t)image_texture,
                 ImVec2(display_image_width, display_image_height));
}

int main(int, char **) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
        0) {
        spdlog::error("Error: {}\n", SDL_GetError());
        return -1;
    }

    //
    RollingBuffer processing_data(100);

    // Load models
    auto cwd = fs::current_path();
    auto model_path = cwd / "face_landmark.so";
    int batch_size = 5;
    auto face_mesh_detector =
        tvm_facemesh::TVM_Facemesh(model_path, batch_size);
    std::vector<cv::Mat> frames;
    std::deque<cv::Mat> render_frames;
    cv::Mat last_frame;

    // Setup window
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                          SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Window *window =
        SDL_CreateWindow("Mukham", SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);  // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
    // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
    // Enable Gamepad Controls

    // Setup Dear ImGui style
    // ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL2_Init();

    // Our state
    bool record_video = false;
    bool play_video = false;
    int video_src = 0;
    std::string btn_txt{"Start Video"};
    std::string play_btn_txt{"Play"};
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    bool is_camera_open = false;
    int prev_video_src = 0;
    unsigned int nb_frames = 0;
    cv::VideoCapture camera;

    // Main loop
    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) done = true;
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Mukham");

            ImGui::RadioButton("Camera", &video_src, 0);
            ImGui::SameLine();
            ImGui::RadioButton("TestVideo", &video_src, 1);

            if (video_src != prev_video_src) {
                camera.release();
                prev_video_src = video_src;
                spdlog::info("Camera released\n");
                is_camera_open = false;
            }

            if (video_src == 0) {
                if (!is_camera_open) {
                    spdlog::info("Opening the camera\n\n");
                    is_camera_open = camera.open(0);
                    camera.set(cv::CAP_PROP_FRAME_HEIGHT,
                               (double)display_image_height);
                    camera.set(cv::CAP_PROP_FRAME_WIDTH,
                               (double)display_image_width);
                }
                // If the video src is camera then show the start video
                // and stop video buttons
                if (ImGui::Button(btn_txt.c_str())) {
                    record_video = !record_video;
                    if (record_video)
                        btn_txt = std::string{"Stop Video"};
                    else
                        btn_txt = std::string{"Start Video"};
                }
            } else {
                const cv::String test_video_fname{"closeup_1.mp4"};

                if (!is_camera_open) {
                    is_camera_open = camera.open(test_video_fname);
                    nb_frames = camera.get(cv::CAP_PROP_FRAME_COUNT);
                }
                // Show the play and pause button
                if (ImGui::Button(play_btn_txt.c_str())) {
                    record_video = !record_video;
                    if (record_video)
                        play_btn_txt = std::string{"Pause"};
                    else
                        play_btn_txt = std::string{"Play"};
                }
                ImGui::Text(
                    "https://github.com/intel-iot-devkit/sample-videos/blob/"
                    "master/face-demographics-walking.mp4");
            }

            ImGui::Text("Frames = %d", counter);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::End();

            auto x_min = processing_data.x.front();
            auto x_max = processing_data.x.back();
            auto y_min = *std::min_element(processing_data.y.begin(),
                                           processing_data.y.end());
            auto y_max = *std::max_element(processing_data.y.begin(),
                                           processing_data.y.end());
            ImPlot::SetNextPlotLimits(x_min, x_max, y_min - 5, y_max + 5,
                                      ImGuiCond_Always);
            ImPlot::BeginPlot("##Processing Time", "Frames", "Time (ms)",
                              ImVec2(-1, -1));
            ImPlot::PlotLine("Processing time", processing_data.x.data(),
                             processing_data.y.data(),
                             processing_data._history);
            ImPlot::EndPlot();

            ImGui::Begin("Video");
            if (record_video) {
                // Get the video frame

                cv::Mat frame;
                auto retval = camera.read(frame);

                if (retval) {
                    // Convert the frame to RGB
                    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
                    counter++;

                    // Reset the video
                    if (video_src == 1 && counter == nb_frames) {
                        camera.set(cv::CAP_PROP_POS_FRAMES, 0);
                        counter = 0;
                    }

                    if (frames.size() == batch_size) {
                        std::vector<tvm_facemesh::TVM_FacemeshResult> results;
                        face_mesh_detector.Detect(frames, results);

                        for (int frame_idx = 0; frame_idx < frames.size();
                             ++frame_idx) {
                            auto const &result = results[frame_idx];
                            auto &frame = frames[frame_idx];
                            processing_data.AddPoint(result.processing_time);
                            if (result.has_face) {
                                for (auto &point : result.mesh) {
                                    cv::circle(frame, point, 3,
                                               cv::Scalar(0, 255, 0));
                                }
                            } else {
                                spdlog::info("{}: No face. face score {}",
                                             counter + frame_idx,
                                             result.face_score);
                            }
                            render_frames.push_back(frame);
                            frame.release();
                        }
                        frames.clear();
                    } else {
                        frames.push_back(frame);
                    }
                }
            }

            if (!render_frames.empty()) {
                auto render_frame = render_frames.front();
                RenderImage(render_frame, display_image_width,
                            display_image_height);
                last_frame = render_frame;
                render_frames.pop_front();
            } else {
                if (last_frame.rows > 0 && last_frame.cols > 0)
                    RenderImage(last_frame, display_image_width,
                                display_image_height);
            }
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
