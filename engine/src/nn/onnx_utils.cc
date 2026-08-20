#include "nn/onnx_utils.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

std::string resolveModelPath(const std::string& explicitPath) {
    if (!explicitPath.empty()) {
        return explicitPath;
    }
    const std::vector<std::string> searchDirs = {
        "./models",
        "./engine/models",
        "../models",
        "./networks",
        "./engine/networks"
    };
    for (const auto& dir : searchDirs) {
        if (fs::is_directory(dir)) {
            std::string latest = findLatestOnnxFile(dir);
            if (!latest.empty()) {
                return latest;
            }
        }
    }
    return "";
}

std::string findLatestOnnxFile(const std::string& directory) {
    std::string latestFile;
    std::optional<fs::file_time_type> latestTime;

    if (!fs::is_directory(directory)) {
        return latestFile;
    }

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

std::string getEnginePath(const std::string& onnxPath, const std::string& precision,
                          int batchSize, int deviceId, const std::string& version) {
    fs::path onnx = fs::weakly_canonical(onnxPath);
    std::string modelName = onnx.stem().string();
    std::string directory = onnx.parent_path().string();
    
    std::string engineName = modelName + "_" + precision + "_b" + std::to_string(batchSize) 
                           + "_gpu" + std::to_string(deviceId) + "_" + version + ".engine";
    
    return directory.empty() ? engineName : directory + "/" + engineName;
}

std::string computeFileSignature(const std::string& path, std::string_view buildDescriptor) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }

    constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET_BASIS;
    auto addBytes = [&](const char* data, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            hash ^= static_cast<unsigned char>(data[index]);
            hash *= FNV_PRIME;
        }
    };

    std::array<char, 64 * 1024> buffer;
    while (file) {
        file.read(buffer.data(), buffer.size());
        addBytes(buffer.data(), static_cast<size_t>(file.gcount()));
    }
    if (!file.eof()) {
        return {};
    }
    addBytes(buildDescriptor.data(), buildDescriptor.size());

    std::ostringstream signature;
    signature << std::hex << std::setfill('0') << std::setw(16) << hash;
    return signature.str();
}
