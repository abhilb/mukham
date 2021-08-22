
#include <SDL.h>
#include <SDL_opengl.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <utility>

#include "dlib_face_detection.h"
#include "imgui.h"
#include "imgui_impl_opengl2.h"
#include "imgui_impl_sdl.h"
#include "implot.h"
#include "opencv_face_detection.h"
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
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::min();
    float x_min = 0;
    float x_max = 0;

    RollingBuffer(int history) {
        _history = history;
        y.reserve(history);
        x = std::vector<float>(history);
        std::iota(x.begin(), x.end(), 0);
    }

    void AddPoint(float data) {
        y.push_back(data);
        if (y.size() > _history) y.erase(y.begin());
        x_min = x.front();
        x_max = x.back();
        y_min = std::min(y_min, data);
        y_max = std::max(y_max, data);
    }
};

class ImageRenderer {
   public:
    ImageRenderer(int width, int height) {
        assert(width == 0);
        assert(height == 0);

        _width = width;
        _height = height;

        glGenTextures(1, &_texture);
        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        auto dummy_image = cv::imread("noimage.jpg");
        cv::cvtColor(dummy_image, dummy_image, cv::COLOR_BGR2RGB);
        cv::resize(dummy_image, _image, cv::Size(_width, _height),
                   cv::INTER_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, _image.ptr());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    int GetWidth() const { return _width; }

    int GetHeight() const { return _height; }

    void RenderImage() {
        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, _image.ptr());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void UpdateImage(cv::Mat &image) {
        if (image.rows != _height || image.cols != _width)
            cv::resize(image, _image, cv::Size(_width, _height),
                       cv::INTER_LINEAR);
        else
            _image = image.clone();
    }

    void UpdateAndRender(cv::Mat &image) {
        if (image.rows > 0 && image.cols > 0) {
            UpdateImage(image);
            RenderImage();
        }
    }

    GLuint GetTextureId() const { return _texture; }

   private:
    GLuint _texture;
    int _height;
    int _width;
    cv::Mat _image;
};

int main(int, char **) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
        0) {
        spdlog::error("Error: {}\n", SDL_GetError());
        return -1;
    }

    RollingBuffer processing_data(100);
    RollingBuffer face_detect_time(100);

    // Load models
    auto cwd = fs::current_path();
    auto model_path = cwd / "face_landmark.so";
    int batch_size = 8;
    auto face_mesh_detector =
        tvm_facemesh::TVM_Facemesh(model_path, batch_size);
    std::vector<cv::Mat> frames;
    std::deque<cv::Mat> render_frames;
    std::deque<cv::Mat> face_images;
    std::deque<cv::Mat> face_landmark_images;
    cv::Mat last_frame;

    auto dlib_hog_face_detector = dlib_facedetect::DlibFaceDetectHog();
    auto opencv_lbp_face_detector = opencv_facedetect::OpenCVFaceDetectLBP();

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

    // Init the image renderer and display dummy image
    ImageRenderer img_renderer(display_image_width, display_image_height);
    img_renderer.RenderImage();

    ImageRenderer face_renderer(256, 256);
    ImageRenderer landmark_renderer(256, 256);

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

    int face_detect_model = 0;

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
                const cv::String test_video_fname{"demo2.mp4"};

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

            ImGui::Begin("Face mesh processing time");
            ImPlot::SetNextPlotLimitsX(processing_data.x_min,
                                       processing_data.x_max, ImGuiCond_Always);
            ImPlot::SetNextPlotLimitsY(processing_data.y_min - 5,
                                       processing_data.y_max + 5,
                                       ImGuiCond_Always);
            if (ImPlot::BeginPlot("##Processing Time 1", "Frames", "Time (ms)",
                                  ImVec2(-1, -1))) {
                // Face mesh processing time plots
                ImPlot::PlotLine("Facemesh", processing_data.x.data(),
                                 processing_data.y.data(),
                                 processing_data._history);
                ImPlot::EndPlot();
            }
            ImGui::End();

            ImGui::Begin("Face detection processing time");
            ImPlot::SetNextPlotLimitsX(face_detect_time.x_min,
                                       face_detect_time.x_max,
                                       ImGuiCond_Always);
            ImPlot::SetNextPlotLimitsY(face_detect_time.y_min - 5,
                                       face_detect_time.y_max + 5,
                                       ImGuiCond_Always);
            if (ImPlot::BeginPlot("##Processing Time 2", "Frames", "Time (ms)",
                                  ImVec2(-1, -1))) {
                // Face detection processing time plots
                ImPlot::PlotLine("dlib face detect", face_detect_time.x.data(),
                                 face_detect_time.y.data(),
                                 face_detect_time._history);
                ImPlot::EndPlot();
            }
            ImGui::End();

            ImGui::Begin("Face");
            if (!face_images.empty()) {
                face_renderer.UpdateAndRender(face_images.front());
                face_images.pop_front();
                ImGui::Image((void *)(intptr_t)(face_renderer.GetTextureId()),
                             ImVec2(face_renderer.GetWidth(),
                                    face_renderer.GetHeight()));
            }
            ImGui::End();

            ImGui::Begin("Landmarks");
            if (!face_landmark_images.empty()) {
                landmark_renderer.UpdateAndRender(face_landmark_images.front());
                face_landmark_images.pop_front();
                ImGui::Image(
                    (void *)(intptr_t)(landmark_renderer.GetTextureId()),
                    ImVec2(landmark_renderer.GetWidth(),
                           landmark_renderer.GetHeight()));
            }
            ImGui::End();

            ImGui::Begin("Parameters");
            if (ImGui::CollapsingHeader("Face detection")) {
                ImGui::RadioButton("Dlib HOG Face detection",
                                   &face_detect_model, 0);
                ImGui::RadioButton("OpenCV LBP Face detection",
                                   &face_detect_model, 1);
            }
            ImGui::End();

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

                    cv::Mat small_frame;
                    cv::resize(frame, small_frame, cv::Size(0, 0), 0.5, 0.5,
                               cv::INTER_LINEAR);
                    cv::Mat adjusted_frame;
                    small_frame.convertTo(adjusted_frame, -1, 2.0, 10);
                    render_frames.push_back(adjusted_frame);

                    auto start = std::chrono::steady_clock::now();
                    std::vector<cv::Rect2d> faces;

                    switch (face_detect_model) {
                        case 0:
                            faces = dlib_hog_face_detector.DetectFace(
                                adjusted_frame);
                            break;
                        case 1:
                            faces = opencv_lbp_face_detector.DetectFace(
                                adjusted_frame);
                            break;
                    }

                    auto end = std::chrono::steady_clock::now();
                    auto face_detect_duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count();
                    face_detect_time.AddPoint(face_detect_duration);

                    for (cv::Rect2d &face : faces) {
                        if (frames.size() == batch_size) {
                            std::vector<tvm_facemesh::TVM_FacemeshResult>
                                results;
                            face_mesh_detector.Detect(frames, results);

                            for (int frame_idx = 0; frame_idx < frames.size();
                                 ++frame_idx) {
                                auto &image = frames[frame_idx];
                                auto const &result = results[frame_idx];
                                processing_data.AddPoint(
                                    result.processing_time);
                                if (result.has_face) {
                                    for (auto &point : result.mesh) {
                                        cv::circle(image, point, 0,
                                                   cv::Scalar(0, 255, 0), -1);
                                    }
                                } else {
                                    spdlog::info("{}: No face. face score {}",
                                                 counter + frame_idx,
                                                 result.face_score);
                                }
                                face_landmark_images.push_back(image);
                            }
                            frames.clear();
                        } else {
                            auto start_row = std::max<float>(0, face.y - 20);
                            auto end_row = std::min<float>(
                                face.y + face.height + 20, small_frame.rows);
                            auto start_col = std::max<float>(0, face.x - 20);
                            auto end_col = std::min<float>(
                                face.x + face.width + 20, small_frame.cols);

                            auto face_image =
                                small_frame(cv::Range(start_row, end_row),
                                            cv::Range(start_col, end_col));
                            face_images.push_back(face_image);
                            frames.push_back(face_image);
                        }
                    }
                }
            }

            if (!render_frames.empty()) {
                img_renderer.UpdateAndRender(render_frames.front());
                render_frames.pop_front();
            }

            GLuint texture_id = img_renderer.GetTextureId();
            ImGui::Image((void *)(intptr_t)(texture_id),
                         ImVec2(display_image_width, display_image_height));
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
