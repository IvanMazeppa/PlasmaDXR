#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>

// Simple logging system
class Logger {
public:
    enum Level {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };

    static void Initialize(const std::string& appName);
    static void Shutdown();

    static void Log(Level level, const std::string& message);

    // Helper macros defined below class
    template<typename... Args>
    static void LogFormat(Level level, const std::string& format, Args... args) {
        // Simple format string replacement
        std::string msg = format;
        (ReplaceFirst(msg, "{}", std::to_string(args)), ...);
        Log(level, msg);
    }

private:
    static void ReplaceFirst(std::string& str, const std::string& from, const std::string& to);
    static std::string GetTimestamp();
    static std::string GetLevelString(Level level);

    static std::ofstream s_file;
    static std::mutex s_mutex;
    static bool s_initialized;
};

// Convenience macros
#define LOG_DEBUG(msg, ...) Logger::LogFormat(Logger::DEBUG, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) Logger::LogFormat(Logger::INFO, msg, ##__VA_ARGS__)
#define LOG_WARN(msg, ...) Logger::LogFormat(Logger::WARNING, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) Logger::LogFormat(Logger::ERROR, msg, ##__VA_ARGS__)
#define LOG_CRITICAL(msg, ...) Logger::LogFormat(Logger::CRITICAL, msg, ##__VA_ARGS__)