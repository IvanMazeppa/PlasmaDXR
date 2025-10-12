// PlasmaDX-Clean - Modern, clean particle system with RT lighting
// Entry point - kept minimal and clean

#include "core/Application.h"
#include "utils/Logger.h"
#ifdef USE_PIX
#include "debug/PIXCaptureHelper.h"
#endif
#include <windows.h>
#include <exception>
#include <string>
#include <vector>

// Agility SDK exports
extern "C" {
    __declspec(dllexport) extern const unsigned int D3D12SDKVersion = 618;
    __declspec(dllexport) extern const char* D3D12SDKPath = "D3D12\\";
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR lpCmdLine, int nCmdShow) {
    try {
        // Initialize logging
        Logger::Initialize("PlasmaDX-Clean");

        LOG_INFO("=== PlasmaDX-Clean Starting ===");
        LOG_INFO("Clean architecture, modern design");
        LOG_INFO("Target: 100K particles with RT lighting");

#ifdef USE_PIX
        // Initialize PIX capture system (checks for auto-capture env vars)
        Debug::PIXCaptureHelper::Initialize();
#else
        LOG_INFO("[PIX] PIX support disabled (USE_PIX not defined)");
#endif

        // Parse command line into argc/argv
        int argc = 0;
        char** argv = nullptr;
        if (lpCmdLine && lpCmdLine[0]) {
            // Simple command line parsing (Windows provides a single string)
            std::string cmdLine = lpCmdLine;
            std::vector<std::string> args;
            args.push_back("PlasmaDX-Clean.exe"); // argv[0]

            size_t pos = 0;
            while (pos < cmdLine.length()) {
                // Skip whitespace
                while (pos < cmdLine.length() && isspace(cmdLine[pos])) pos++;
                if (pos >= cmdLine.length()) break;

                // Extract argument
                size_t start = pos;
                while (pos < cmdLine.length() && !isspace(cmdLine[pos])) pos++;
                args.push_back(cmdLine.substr(start, pos - start));
            }

            argc = static_cast<int>(args.size());
            argv = new char*[argc];
            for (int i = 0; i < argc; i++) {
                argv[i] = new char[args[i].length() + 1];
                strcpy_s(argv[i], args[i].length() + 1, args[i].c_str());
            }
        }

        // Create and run application
        Application app;

        if (!app.Initialize(hInstance, nCmdShow, argc, argv)) {
            LOG_ERROR("Failed to initialize application");
            MessageBoxA(nullptr, "Failed to initialize application. Check logs folder.", "Init Error", MB_OK | MB_ICONERROR);

            // Cleanup argv
            if (argv) {
                for (int i = 0; i < argc; i++) delete[] argv[i];
                delete[] argv;
            }
            return -1;
        }

        // Cleanup argv
        if (argv) {
            for (int i = 0; i < argc; i++) delete[] argv[i];
            delete[] argv;
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

