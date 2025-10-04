#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <type_traits>

// Simple logging system
class Logger {
public:
    enum class Level {
        Debug,
        Info,
        Warning,
        Error,
        Critical
    };

    static void Initialize(const std::string& appName);
    static void Shutdown();

    static void Log(Level level, const std::string& message);

    // Helper for format string replacement - base case
    static void LogFormat(Level level, const std::string& format) {
        Log(level, format);
    }

    // Helper - convert value to string (specializations)
    static std::string ToString(const char* value) {
        return std::string(value);
    }

    static std::string ToString(const std::string& value) {
        return value;
    }

    static std::string ToString(bool value) {
        return value ? "true" : "false";
    }

    // Template for numeric types
    template<typename T>
    static std::string ToString(const T& value,
        typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr) {
        return std::to_string(value);
    }

    // Helper for format string replacement - recursive case
    template<typename T, typename... Args>
    static void LogFormat(Level level, const std::string& format, T value, Args... args) {
        std::string msg = format;
        ReplaceFirst(msg, "{}", ToString(value));
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
#define LOG_DEBUG(msg, ...) Logger::LogFormat(Logger::Level::Debug, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) Logger::LogFormat(Logger::Level::Info, msg, ##__VA_ARGS__)
#define LOG_WARN(msg, ...) Logger::LogFormat(Logger::Level::Warning, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) Logger::LogFormat(Logger::Level::Error, msg, ##__VA_ARGS__)
#define LOG_CRITICAL(msg, ...) Logger::LogFormat(Logger::Level::Critical, msg, ##__VA_ARGS__)