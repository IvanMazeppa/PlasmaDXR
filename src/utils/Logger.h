#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <cstdint>

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

    // Helper - convert value to string with format specifier support
    static std::string FormatValue(const char* value, const std::string& formatSpec) {
        return std::string(value);
    }

    static std::string FormatValue(const std::string& value, const std::string& formatSpec) {
        return value;
    }

    static std::string FormatValue(bool value, const std::string& formatSpec) {
        return value ? "true" : "false";
    }

    // Template for numeric types with format specifier support
    template<typename T>
    static std::string FormatValue(const T& value, const std::string& formatSpec,
        typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr) {

        // Empty format spec = default decimal
        if (formatSpec.empty()) {
            return std::to_string(value);
        }

        std::stringstream ss;

        // Parse format specifier (e.g., ":08X", ":.2f", ":04X", ":016X")
        if (formatSpec[0] == ':') {
            std::string spec = formatSpec.substr(1);

            // Hex formats: :08X, :04X, :016X
            if (spec.find('X') != std::string::npos || spec.find('x') != std::string::npos) {
                bool uppercase = (spec.find('X') != std::string::npos);

                // Extract width (e.g., "08" from "08X")
                size_t width = 0;
                bool zeroPad = false;
                if (spec.length() > 1) {
                    if (spec[0] == '0') {
                        zeroPad = true;
                        width = std::stoi(spec.substr(0, spec.length() - 1));
                    } else {
                        width = std::stoi(spec.substr(0, spec.length() - 1));
                    }
                }

                // Format as hex
                ss << "0x";
                if (zeroPad && width > 0) {
                    ss << std::setfill('0') << std::setw(width);
                }
                if (uppercase) {
                    ss << std::uppercase << std::hex << static_cast<uint64_t>(value);
                } else {
                    ss << std::hex << static_cast<uint64_t>(value);
                }
                return ss.str();
            }

            // Float formats: :.2f, :.3f
            else if (spec.find('f') != std::string::npos) {
                // Extract precision (e.g., "2" from ".2f")
                size_t dotPos = spec.find('.');
                if (dotPos != std::string::npos && dotPos + 1 < spec.length()) {
                    int precision = std::stoi(spec.substr(dotPos + 1, spec.length() - dotPos - 2));
                    ss << std::fixed << std::setprecision(precision) << static_cast<double>(value);
                    return ss.str();
                }
            }
        }

        // Fallback: default decimal
        return std::to_string(value);
    }

    // Helper for format string replacement - recursive case
    template<typename T, typename... Args>
    static void LogFormat(Level level, const std::string& format, T value, Args... args) {
        std::string msg = format;

        // Find next placeholder (either {} or {:...})
        size_t startPos = msg.find('{');
        if (startPos != std::string::npos) {
            size_t endPos = msg.find('}', startPos);
            if (endPos != std::string::npos) {
                // Extract placeholder including format specifier
                std::string placeholder = msg.substr(startPos, endPos - startPos + 1);

                // Extract format specifier (e.g., ":08X" from "{:08X}")
                std::string formatSpec;
                if (placeholder.length() > 2) {
                    formatSpec = placeholder.substr(1, placeholder.length() - 2);
                }

                // Format value and replace placeholder
                std::string formatted = FormatValue(value, formatSpec);
                msg.replace(startPos, placeholder.length(), formatted);
            }
        }

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