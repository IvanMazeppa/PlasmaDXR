#include "Logger.h"
#include <iostream>
#include <windows.h>

std::ofstream Logger::s_file;
std::mutex Logger::s_mutex;
bool Logger::s_initialized = false;

void Logger::Initialize(const std::string& appName) {
    {
        std::lock_guard<std::mutex> lock(s_mutex);

        if (s_initialized) return;

        // Create logs directory
        CreateDirectoryA("logs", nullptr);

        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        struct tm timeinfo;
        localtime_s(&timeinfo, &time_t);

        std::stringstream filename;
        filename << "logs/" << appName << "_"
                 << std::put_time(&timeinfo, "%Y%m%d_%H%M%S")
                 << ".log";

        s_file.open(filename.str());
        if (!s_file.is_open()) {
            // Can't log if file won't open, but don't abort
            s_initialized = false;
            return;
        }
        s_initialized = true;
    } // Release lock before calling Log()

    Log(Level::Info, "=== " + appName + " Log Session Started ===");
}

void Logger::Shutdown() {
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_file.is_open()) {
        s_file.close();
    }
    s_initialized = false;
}

void Logger::Log(Level level, const std::string& message) {
    if (!s_initialized) return;

    std::lock_guard<std::mutex> lock(s_mutex);

    std::string timestamp = GetTimestamp();
    std::string levelStr = GetLevelString(level);
    std::string logLine = "[" + timestamp + "] [" + levelStr + "] " + message;

    // Write to file
    if (s_file.is_open()) {
        s_file << logLine << std::endl;
        s_file.flush();
    }

    // Write to console
    std::cout << logLine << std::endl;

    // Also output to debug console in Visual Studio
#ifdef _WIN32
    OutputDebugStringA((logLine + "\n").c_str());
#endif
}

void Logger::ReplaceFirst(std::string& str, const std::string& from, const std::string& to) {
    size_t pos = str.find(from);
    if (pos != std::string::npos) {
        str.replace(pos, from.length(), to);
    }
}

std::string Logger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
    localtime_s(&timeinfo, &time_t);

    std::stringstream ss;
    ss << std::put_time(&timeinfo, "%H:%M:%S");
    return ss.str();
}

std::string Logger::GetLevelString(Level level) {
    switch (level) {
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO";
        case Level::Warning: return "WARN";
        case Level::Error: return "ERROR";
        case Level::Critical: return "CRIT";
        default: return "????";
    }
}