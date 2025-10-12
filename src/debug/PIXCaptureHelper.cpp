#include "PIXCaptureHelper.h"
#include <iostream>
#include <cstdio>

namespace Debug {

// Static member initialization
HMODULE PIXCaptureHelper::s_pixModule = nullptr;
bool PIXCaptureHelper::s_initialized = false;
bool PIXCaptureHelper::s_autoCapture = false;
int PIXCaptureHelper::s_captureFrame = -1;
bool PIXCaptureHelper::s_captureInProgress = false;
int PIXCaptureHelper::s_captureStartFrame = -1;

void PIXCaptureHelper::Initialize() {
    if (s_initialized) {
        return;
    }

    s_initialized = true;

    // Health check: Log initialization start
    std::printf("[PIX] PIXCaptureHelper::Initialize() called\n");
    std::fflush(stdout);

#ifdef USE_PIX
    // Load WinPixGpuCapturer.dll from PIX installation
    s_pixModule = PIXLoadLatestWinPixGpuCapturerLibrary();

    if (s_pixModule) {
        std::printf("[PIX] ✓ GPU Capturer DLL loaded successfully\n");
        std::fflush(stdout);
    } else {
        std::printf("[PIX] ✗ GPU Capturer DLL NOT loaded - PIX captures disabled\n");
        std::printf("[PIX]   Ensure PIX is installed at: C:\\Program Files\\Microsoft PIX\\\n");
        std::fflush(stdout);
        return;
    }
#else
    std::printf("[PIX] ✗ USE_PIX not defined - PIX support disabled at compile time\n");
    std::fflush(stdout);
    return;
#endif

    // Check for automatic capture environment variables
    const char* autoEnv = std::getenv("PIX_AUTO_CAPTURE");
    const char* frameEnv = std::getenv("PIX_CAPTURE_FRAME");

    if (autoEnv && std::string(autoEnv) == "1") {
        s_autoCapture = true;
        s_captureFrame = frameEnv ? std::atoi(frameEnv) : 120; // Default to frame 120 (~2s @ 60fps)

        std::printf("[PIX] ✓ Auto-capture ENABLED - will capture at frame %d\n", s_captureFrame);
        std::printf("[PIX]   Environment: PIX_AUTO_CAPTURE=1, PIX_CAPTURE_FRAME=%d\n", s_captureFrame);
        std::fflush(stdout);
    } else {
        std::printf("[PIX] Auto-capture DISABLED (no PIX_AUTO_CAPTURE=1 env var)\n");
        std::fflush(stdout);
    }
}

bool PIXCaptureHelper::CheckAutomaticCapture(int frameNumber) {
    if (!s_initialized || !s_autoCapture || !s_pixModule) {
        return false;
    }

#ifdef USE_PIX
    // If capture is not yet started and we've reached the trigger frame
    if (!s_captureInProgress && frameNumber >= s_captureFrame) {
        std::printf("[PIX] Frame %d: Starting capture...\n", frameNumber);
        std::fflush(stdout);

        // Begin capture - filename will be set by pixtool via save-capture
        HRESULT hr = PIXBeginCapture(PIX_CAPTURE_GPU, nullptr);

        if (SUCCEEDED(hr)) {
            s_captureInProgress = true;
            s_captureStartFrame = frameNumber;
            std::printf("[PIX] ✓ Capture started successfully\n");
            std::fflush(stdout);
        } else {
            std::printf("[PIX] ✗ Capture START failed (HRESULT: 0x%08X)\n", hr);
            std::fflush(stdout);
            // Still exit to avoid infinite loop
            PostQuitMessage(0);
            return true;
        }
    }

    // If capture is in progress, end it after one frame
    if (s_captureInProgress && frameNumber > s_captureStartFrame) {
        std::printf("[PIX] Frame %d: Ending capture...\n", frameNumber);
        std::fflush(stdout);

        // End capture - discard=FALSE to save the capture
        HRESULT hr = PIXEndCapture(FALSE);

        if (SUCCEEDED(hr)) {
            std::printf("[PIX] ✓ Capture ended successfully\n");
            std::fflush(stdout);
        } else {
            std::printf("[PIX] ✗ Capture END failed (HRESULT: 0x%08X)\n", hr);
            std::fflush(stdout);
        }

        // Exit application so pixtool can save the capture
        std::printf("[PIX] Exiting application for capture save...\n");
        std::fflush(stdout);
        PostQuitMessage(0);
        return true;
    }
#endif

    return false;
}

bool PIXCaptureHelper::BeginCapture(const wchar_t* filename) {
    if (!s_pixModule) {
        std::printf("[PIX] ✗ Cannot begin capture - PIX not loaded\n");
        std::fflush(stdout);
        return false;
    }

#ifdef USE_PIX
    std::printf("[PIX] Beginning manual capture...\n");
    std::fflush(stdout);

    HRESULT hr = PIXBeginCapture(PIX_CAPTURE_GPU, nullptr);

    if (SUCCEEDED(hr)) {
        s_captureInProgress = true;
        std::printf("[PIX] ✓ Manual capture started\n");
        std::fflush(stdout);
        return true;
    } else {
        std::printf("[PIX] ✗ Manual capture START failed (HRESULT: 0x%08X)\n", hr);
        std::fflush(stdout);
        return false;
    }
#else
    return false;
#endif
}

bool PIXCaptureHelper::EndCapture() {
    if (!s_pixModule || !s_captureInProgress) {
        std::printf("[PIX] ✗ Cannot end capture - no capture in progress\n");
        std::fflush(stdout);
        return false;
    }

#ifdef USE_PIX
    std::printf("[PIX] Ending manual capture...\n");
    std::fflush(stdout);

    HRESULT hr = PIXEndCapture(FALSE);

    if (SUCCEEDED(hr)) {
        s_captureInProgress = false;
        std::printf("[PIX] ✓ Manual capture ended\n");
        std::fflush(stdout);
        return true;
    } else {
        std::printf("[PIX] ✗ Manual capture END failed (HRESULT: 0x%08X)\n", hr);
        std::fflush(stdout);
        return false;
    }
#else
    return false;
#endif
}

bool PIXCaptureHelper::IsAvailable() {
    return s_pixModule != nullptr;
}

} // namespace Debug
