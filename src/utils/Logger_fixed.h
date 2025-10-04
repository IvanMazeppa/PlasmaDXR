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

    // Helper for format string replacement - base case
    static void LogFormat(Level level, const std::string& format) {
        Log(level, format);
    }

    // Helper for format string replacement - recursive case
    template<typename T, typename... Args>
    static void LogFormat(Level level, const std::string& format, T value, Args... args) {
        std::string msg = format;
        ReplaceFirst(msg, "{}", std::to_string(value));
        LogFormat(level, msg, args...);
    }

    // Specialization for string literals
    template<typename... Args>
    static void LogFormat(Level level, const std::string& format, const char* value, Args... args) {
        std::string msg = format;
        ReplaceFirst(msg, "{}", std::string(value));
        LogFormat(level, msg, args...);
    }

    // Specialization for std::string
    template<typename... Args>
    static void LogFormat(Level level, const std::string& format, const std::string& value, Args... args) {
        std::string msg = format;
        ReplaceFirst(msg, "{}", value);
        LogFormat(level, msg, args...);
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
