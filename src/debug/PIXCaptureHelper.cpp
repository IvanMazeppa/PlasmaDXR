#include "PIXCaptureHelper.h"
#include "../utils/Logger.h"
#include <string>

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

    LOG_INFO("[PIX] Initializing PIX capture system...");

#ifdef USE_PIX
    // Load WinPixGpuCapturer.dll from PIX installation
    s_pixModule = PIXLoadLatestWinPixGpuCapturerLibrary();

    if (s_pixModule) {
        LOG_INFO("[PIX] GPU Capturer DLL loaded successfully");
    } else {
        LOG_WARN("[PIX] GPU Capturer DLL NOT loaded - PIX captures disabled");
        LOG_WARN("[PIX] Ensure WinPixGpuCapturer.dll is in exe directory or PIX is installed");
        return;
    }
#else
    LOG_WARN("[PIX] USE_PIX not defined - PIX support disabled at compile time");
    return;
#endif

    // Check for automatic capture environment variables
    const char* autoEnv = std::getenv("PIX_AUTO_CAPTURE");
    const char* frameEnv = std::getenv("PIX_CAPTURE_FRAME");

    if (autoEnv && std::string(autoEnv) == "1") {
        s_autoCapture = true;
        s_captureFrame = frameEnv ? std::atoi(frameEnv) : 120; // Default to frame 120 (~2s @ 60fps)

        LOG_INFO("[PIX] Auto-capture ENABLED - will capture at frame {}", s_captureFrame);
        LOG_INFO("[PIX] Environment: PIX_AUTO_CAPTURE=1, PIX_CAPTURE_FRAME={}", s_captureFrame);
    } else {
        LOG_INFO("[PIX] Auto-capture DISABLED (no PIX_AUTO_CAPTURE=1 env var)");
    }
}

bool PIXCaptureHelper::CheckAutomaticCapture(int frameNumber) {
    if (!s_initialized || !s_autoCapture || !s_pixModule) {
        return false;
    }

#ifdef USE_PIX
    // If capture is not yet started and we've reached the trigger frame
    if (!s_captureInProgress && frameNumber >= s_captureFrame) {
        LOG_INFO("[PIX] Frame {}: Starting capture...", frameNumber);

        // When launched with programmatic-capture, pass NULL to let pixtool manage the filename
        // PIX will use a temporary file and pixtool will save it with the specified name
        HRESULT hr = PIXBeginCapture(PIX_CAPTURE_GPU, nullptr);

        if (SUCCEEDED(hr)) {
            s_captureInProgress = true;
            s_captureStartFrame = frameNumber;
            LOG_INFO("[PIX] Capture started successfully");
        } else {
            LOG_ERROR("[PIX] Capture START failed (HRESULT: 0x{:08X})", static_cast<unsigned int>(hr));
            // Still exit to avoid infinite loop
            PostQuitMessage(0);
            return true;
        }
    }

    // If capture is in progress, end it after one frame
    if (s_captureInProgress && frameNumber > s_captureStartFrame) {
        LOG_INFO("[PIX] Frame {}: Ending capture...", frameNumber);

        // End capture - discard=FALSE to save the capture
        HRESULT hr = PIXEndCapture(FALSE);

        if (SUCCEEDED(hr)) {
            LOG_INFO("[PIX] Capture ended successfully");
        } else {
            LOG_ERROR("[PIX] Capture END failed (HRESULT: 0x{:08X})", static_cast<unsigned int>(hr));
        }

        // Exit application so pixtool can save the capture
        LOG_INFO("[PIX] Exiting application for capture save...");
        PostQuitMessage(0);
        return true;
    }
#endif

    return false;
}

bool PIXCaptureHelper::BeginCapture(const wchar_t* filename) {
    if (!s_pixModule) {
        LOG_WARN("[PIX] Cannot begin capture - PIX not loaded");
        return false;
    }

#ifdef USE_PIX
    LOG_INFO("[PIX] Beginning manual capture...");

    // Set up capture parameters
    PIXCaptureParameters captureParams = {};
    captureParams.GpuCaptureFileName = filename ? filename : L"D:\\Users\\dilli\\AndroidStudioProjects\\PlasmaDX-Clean\\pix\\Captures\\manual_capture.wpix";

    HRESULT hr = PIXBeginCapture(PIX_CAPTURE_GPU, &captureParams);

    if (SUCCEEDED(hr)) {
        s_captureInProgress = true;
        LOG_INFO("[PIX] Manual capture started");
        return true;
    } else {
        LOG_ERROR("[PIX] Manual capture START failed (HRESULT: 0x{:08X})", static_cast<unsigned int>(hr));
        return false;
    }
#else
    return false;
#endif
}

bool PIXCaptureHelper::EndCapture() {
    if (!s_pixModule || !s_captureInProgress) {
        LOG_WARN("[PIX] Cannot end capture - no capture in progress");
        return false;
    }

#ifdef USE_PIX
    LOG_INFO("[PIX] Ending manual capture...");

    HRESULT hr = PIXEndCapture(FALSE);

    if (SUCCEEDED(hr)) {
        s_captureInProgress = false;
        LOG_INFO("[PIX] Manual capture ended");
        return true;
    } else {
        LOG_ERROR("[PIX] Manual capture END failed (HRESULT: 0x{:08X})", static_cast<unsigned int>(hr));
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
