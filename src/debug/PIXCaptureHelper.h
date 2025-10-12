#pragma once

// PIX Capture Helper - Enables autonomous GPU capture via PIX programmatic API
// Integrates with pixtool.exe for fully automated debugging workflows

#include <windows.h>
#include <cstdlib>
#include <string>

// Forward declarations for PIX types
#ifdef USE_PIX
#define USE_PIX_SUPPORTED_ARCHITECTURE
#include "pix3.h"  // Local PIX header (in same directory)
#endif
// Note: HMODULE is already defined by windows.h included above

namespace Debug {

/// <summary>
/// Helper class for PIX GPU capture integration
/// Supports environment-variable-triggered automatic capture for autonomous workflows
/// </summary>
class PIXCaptureHelper {
public:
    /// <summary>
    /// Initialize PIX capture system - call once at startup
    /// Loads WinPixGpuCapturer.dll and checks for auto-capture environment variables
    /// Prints health check messages to stdout for diagnostics
    /// </summary>
    static void Initialize();

    /// <summary>
    /// Check if automatic capture should trigger on this frame
    /// Call every frame with incrementing frame number
    /// If capture is triggered, begins capture, waits one frame, ends capture, and exits app
    /// </summary>
    /// <param name="frameNumber">Current frame number (0-indexed)</param>
    /// <returns>True if capture was triggered (app will exit)</returns>
    static bool CheckAutomaticCapture(int frameNumber);

    /// <summary>
    /// Manually trigger a PIX capture (alternative to automatic mode)
    /// Begins capture immediately, caller must call EndCapture() after rendering
    /// </summary>
    /// <param name="filename">Optional filename for capture (nullptr = default)</param>
    /// <returns>True if capture started successfully</returns>
    static bool BeginCapture(const wchar_t* filename = nullptr);

    /// <summary>
    /// End a manually-triggered PIX capture
    /// </summary>
    /// <returns>True if capture ended successfully</returns>
    static bool EndCapture();

    /// <summary>
    /// Check if PIX capture system is available and loaded
    /// </summary>
    /// <returns>True if WinPixGpuCapturer.dll is loaded</returns>
    static bool IsAvailable();

private:
    static HMODULE s_pixModule;
    static bool s_initialized;
    static bool s_autoCapture;
    static int s_captureFrame;
    static bool s_captureInProgress;
    static int s_captureStartFrame;

    // Prevent instantiation
    PIXCaptureHelper() = delete;
    ~PIXCaptureHelper() = delete;
};

} // namespace Debug
