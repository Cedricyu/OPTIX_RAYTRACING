// ======================================================================== //
// Copyright 2018-2019 Ingo Wald
// ======================================================================== //

#include "SampleRenderer.h"

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <vector>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

namespace osc {

    struct CameraController
    {
        vec3f from;
        vec3f at;
        vec3f up{ 0.f, 1.f, 0.f };

        float moveSpeed = 1.0f;
        float rotateSpeed = 0.005f;

        bool rotating = false;
        bool orbiting = false;

        double lastX = 0.0;
        double lastY = 0.0;
        bool firstMouse = true;

        CameraController(const Camera& cam)
            : from(cam.from), at(cam.at), up(cam.up) {}

        vec3f viewDir() const {
            return normalize(at - from);
        }

        vec3f right() const {
            return normalize(cross(viewDir(), up));
        }

        void beginMouse(double x, double y) {
            lastX = x;
            lastY = y;
            firstMouse = false;
        }

        void endMouse() {
            firstMouse = true;
            rotating = orbiting = false;
        }

        void mouseMove(double x, double y) {
            if (firstMouse) {
                beginMouse(x, y);
                return;
            }

            float dx = float(x - lastX);
            float dy = float(y - lastY);
            lastX = x;
            lastY = y;

            if (rotating) {
                vec3f dir = viewDir();
                vec3f r = right();

                dir = normalize(dir
                    + dx * rotateSpeed * r
                    - dy * rotateSpeed * up);

                at = from + dir;
            }

            if (orbiting) {
                vec3f offset = from - at;
                vec3f r = normalize(cross(up, offset));

                offset = offset
                    + dx * rotateSpeed * r * length(offset)
                    - dy * rotateSpeed * up * length(offset);

                from = at + normalize(offset) * length(offset);
            }
        }

        void dolly(float amount) {
            vec3f dir = viewDir();
            from += dir * amount;
            at += dir * amount;
        }

        void move(const vec3f& dir, float dt) {
            from += dir * dt * moveSpeed;
            at += dir * dt * moveSpeed;
        }

        Camera toCamera() const {
            return Camera{ from, at, up };
        }
    };

    CameraController* gCam = nullptr;

    void mouse_button_cb(GLFWwindow* window, int button, int action, int mods)
    {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse && ImGui::IsAnyItemHovered())
            return;

        double x, y;
        glfwGetCursorPos(window, &x, &y);

        if (action == GLFW_PRESS) {
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                gCam->rotating = true;
                gCam->beginMouse(x, y);
            }
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                gCam->orbiting = true;
                gCam->beginMouse(x, y);
            }
        }

        if (action == GLFW_RELEASE) {
            gCam->endMouse();
        }
    }

    void cursor_pos_cb(GLFWwindow* window, double x, double y)
    {

        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse && ImGui::IsAnyItemHovered())
            return;

        if (gCam->rotating || gCam->orbiting) {
            gCam->mouseMove(x, y);
        }
    }

    extern "C" int main(int ac, char** av)
    {
        const char* modelPath =
            "C:/Users/USER/Desktop/OPTIX_RAYTRACING/models/photo_studio/photo_studio.obj";

        try {
           
            Model* model = loadOBJ(modelPath);

            Camera camera = {
                vec3f(0.2f, 2.0f, 0.0f),
                vec3f(0.1f, 2.0f, 0.0f),
                vec3f(0.f, 1.f, 0.f)
            };

            SampleRenderer sample(model);
            sample.setCamera(camera);

            glfwInit();
            GLFWwindow* window = glfwCreateWindow(
                800, 600,
                "OptiX + ImGui",
                nullptr, nullptr
            );

            CameraController camCtrl(camera);
            gCam = &camCtrl;

            glfwSetMouseButtonCallback(window, mouse_button_cb);
            glfwSetCursorPosCallback(window, cursor_pos_cb);

            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGui::StyleColorsDark();

            ImGui_ImplGlfw_InitForOpenGL(window, false);
            ImGui_ImplOpenGL3_Init("#version 130");

            vec2i fbSize(1600, 1200);
            sample.resize(fbSize);

            std::vector<uint32_t> pixels(fbSize.x * fbSize.y);

            GLuint fbTexture = 0;
            glGenTextures(1, &fbTexture);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                sample.setCamera(gCam->toCamera());
                sample.render();
                sample.downloadPixels(pixels.data());

                glBindTexture(GL_TEXTURE_2D, fbTexture);
                glTexImage2D(
                    GL_TEXTURE_2D, 0,
                    GL_RGBA8,
                    fbSize.x, fbSize.y,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    pixels.data()
                );

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

                ImGui::Begin(
                    "Viewport",
                    nullptr,
                    ImGuiWindowFlags_NoDecoration |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoNavFocus
                );

                ImVec2 viewportSize = ImGui::GetContentRegionAvail();

                ImGui::Image(
                    (ImTextureID)(intptr_t)fbTexture,
                    viewportSize,
                    ImVec2(0, 1),
                    ImVec2(1, 0)
                );

                ImGui::End();

                ImGui::Render();
                int w, h;
                glfwGetFramebufferSize(window, &w, &h);
                glViewport(0, 0, w, h);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                glfwSwapBuffers(window);
            }

            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            glfwDestroyWindow(window);
            glfwTerminate();
        }
        catch (std::runtime_error& e) {
            std::cerr << "FATAL ERROR: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }

} // namespace osc
