#include "onnx_utils.h"
#include <filesystem>
#include <optional>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

std::string findLatestOnnxFile(const std::string& directory) {
    std::string latestFile;
    std::optional<fs::file_time_type> latestTime;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
            auto ftime = fs::last_write_time(entry);
            if (!latestTime.has_value() || ftime > *latestTime) {
                latestTime = ftime;
                latestFile = entry.path().string();
            }
        }
    }
    return latestFile;
}

std::string getEnginePath(const std::string& onnxPath) {
    size_t lastDot = onnxPath.find_last_of(".");
    return onnxPath.substr(0, lastDot) + ".engine";
}
