#include "nn/onnx_utils.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <vector>
#include <chrono>

#include "Fairy-Stockfish/src/misc.h"

namespace fs = std::filesystem;

std::string resolveModelPath(const std::string& explicitPath) {
    if (!explicitPath.empty()) {
        return explicitPath;
    }
    std::vector<fs::path> searchDirs;
    if (!Stockfish::CommandLine::binaryDirectory.empty()) {
        const fs::path executableDir = Stockfish::CommandLine::binaryDirectory;
        searchDirs.emplace_back(executableDir / "models");
        searchDirs.emplace_back(executableDir / ".." / "models");
    }
    searchDirs.insert(searchDirs.end(), {
        "./models",
        "./engine/models",
        "../models",
        "./networks",
        "./engine/networks"
    });
    for (const auto& dir : searchDirs) {
        if (fs::is_directory(dir)) {
            std::string latest = findLatestOnnxFile(dir.string());
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

bool isQuantizedOnnx(const std::string& onnxPath) {
    // Q/DQ op types are stored as plain strings in the protobuf; the common
    // suffix matches both "QuantizeLinear" and "DequantizeLinear". Chunks
    // overlap by the needle length so a match cannot straddle them.
    static constexpr std::string_view NEEDLE = "uantizeLinear";
    std::ifstream file(onnxPath, std::ios::binary);
    if (!file) {
        return false;
    }
    std::vector<char> buffer(1 << 20);
    size_t carried = 0;
    while (file) {
        file.read(buffer.data() + carried, static_cast<std::streamsize>(buffer.size() - carried));
        const size_t available = carried + static_cast<size_t>(file.gcount());
        if (std::string_view(buffer.data(), available).find(NEEDLE) != std::string_view::npos) {
            return true;
        }
        carried = std::min(available, NEEDLE.size() - 1);
        std::copy(buffer.begin() + static_cast<std::ptrdiff_t>(available - carried),
                  buffer.begin() + static_cast<std::ptrdiff_t>(available), buffer.begin());
    }
    return false;
}

std::string getEnginePath(const std::string& onnxPath, const std::string& precision,
                          int batchSize, int deviceId, const std::string& version) {
    fs::path onnx = fs::weakly_canonical(onnxPath);
    std::string modelName = onnx.stem().string();
    std::string directory = onnx.parent_path().string();
    // A quantized network runs its convolutions and matrix products in INT8
    // (FP16 elsewhere); name its plan for that rather than the FP16 default.
    const std::string tag = precision == "fp16" && isQuantizedOnnx(onnxPath) ? "int8" : precision;
    // A model already named for its precision ("net-int8", "net_fp16") does
    // not need the tag a second time.
    auto isSeparator = [](char c) { return c == '-' || c == '_' || c == '.'; };
    bool named = false;
    for (size_t at = modelName.find(tag); at != std::string::npos && !named;
         at = modelName.find(tag, at + 1)) {
        const size_t end = at + tag.size();
        named = (at == 0 || isSeparator(modelName[at - 1]))
            && (end == modelName.size() || isSeparator(modelName[end]));
    }
    
    std::string engineName = modelName + (named ? "" : "_" + tag) + "_b" + std::to_string(batchSize) 
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
