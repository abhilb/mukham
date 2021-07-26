
#include <SDL.h>
#include <SDL_opengl.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <utility>
#include <string>

#include "dlpack/dlpack.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"

#include "imgui.h"
#include "imgui_impl_opengl2.h"
#include "imgui_impl_sdl.h"

void LoadImage(const unsigned char* image_data, const int& height, const int& width,  GLuint* texture)
{
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data);
    *texture = image_texture;
}

int main(int, char**) {
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
      0) {
    printf("Error: %s\n", SDL_GetError());
    return -1;
  }

  // Setup window
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
  SDL_WindowFlags window_flags =
      (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                        SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window =
      SDL_CreateWindow("Mukham", SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, gl_context);
  SDL_GL_SetSwapInterval(1);  // Enable vsync

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
  // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
  // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsClassic();

  // Setup Platform/Renderer backends
  ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL2_Init();

  // Our state
  bool show_demo_window = true;
  bool show_another_window = false;
  bool record_video = true;
  std::string btn_txt { "Stop Video" };
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // TVM related
  int ndim = 4;
  int device_type = kDLCPU;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_id = 0;
  int64_t shape[4] = { 1, 640, 480, 3 };

  DLTensor* input_tensor;
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input_tensor);
  TVMArrayFree(input_tensor);

  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("/home/pi/work/mukham/build/version-RFB-640/mod.so");
#if 1
      tvm::runtime::Module gmod =  mod_factory.GetFunction("default")(device_type);
      tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
      tvm::runtime::PackedFunc get_outptu = gmod.GetFunction("get_output");
      tvm::runtime::PackedFunc run = gmod.GetFunction("run");
#endif

  cv::VideoCapture camera(0);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
  camera.set(cv::CAP_PROP_FRAME_WIDTH,  480);

  if (!camera.isOpened())
      printf("Failed to open the camera");

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
      if (ImGui::Button(btn_txt.c_str()))
      {
	  record_video = !record_video;
	  if(record_video)
	      btn_txt = "Stop Video";
	  else
	      btn_txt = "Start Video";
      }

      ImGui::Text("Frames = %d", counter);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();

      ImGui::Begin("Video");
      if(record_video) {
	  cv::Mat frame;
	  camera >> frame;
	  counter++;

	  if (frame.size[0] > 0 && frame.size[1])
	  {
	      auto width = 640;
	      auto height = 480;

	      cv::Mat scaled_frame;
	      cv::resize(frame, scaled_frame, cv::Size(width, height), cv::INTER_LINEAR);
	      GLuint image_texture;
	      LoadImage(scaled_frame.ptr(), height, width, &image_texture);
	      ImGui::Image((void*)(intptr_t)image_texture, ImVec2(width, height));
	  }
      }
      ImGui::End();
    }


    // Rendering
    ImGui::Render();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }

  // Cleanup
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(gl_context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
