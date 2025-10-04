// PlasmaDX-Clean - Modern, clean particle system with RT lighting
// Entry point - kept minimal and clean

#include "core/Application.h"
#include "utils/Logger.h"
#include <windows.h>
#include <exception>

// Agility SDK exports
extern "C" {
    __declspec(dllexport) extern const unsigned int D3D12SDKVersion = 618;
    __declspec(dllexport) extern const char* D3D12SDKPath = "D3D12\\";
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    try {
        // Initialize logging
        Logger::Initialize("PlasmaDX-Clean");

        LOG_INFO("=== PlasmaDX-Clean Starting ===");
        LOG_INFO("Clean architecture, modern design");
        LOG_INFO("Target: 100K particles with RT lighting");

        // Create and run application
        Application app;

        if (!app.Initialize(hInstance, nCmdShow)) {
            LOG_ERROR("Failed to initialize application");
            MessageBoxA(nullptr, "Failed to initialize application. Check logs folder.", "Init Error", MB_OK | MB_ICONERROR);
            return -1;
        }

        // Main loop
        int exitCode = app.Run();

        // Clean shutdown
        app.Shutdown();

        LOG_INFO("=== PlasmaDX-Clean Exiting (Code: {})", exitCode);
        return exitCode;

    } catch (const std::exception& e) {
        LOG_CRITICAL("Unhandled exception: {}", e.what());
        MessageBoxA(nullptr, e.what(), "Critical Error", MB_OK | MB_ICONERROR);
        return -1;
    } catch (...) {
        LOG_CRITICAL("Unknown exception occurred");
        MessageBoxA(nullptr, "Unknown error occurred", "Critical Error", MB_OK | MB_ICONERROR);
        return -1;
    }
}

