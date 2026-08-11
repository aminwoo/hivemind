#pragma once

#include <string>
#include <string_view>

// Returns the path to the latest ONNX file in the given directory, or an empty string if none found.
std::string findLatestOnnxFile(const std::string& directory);

// Converts an ONNX path to a TensorRT engine path
// Format: [model_name]_[precision]_[batch_size]_[device]_[version].engine
std::string getEnginePath(const std::string& onnxPath, const std::string& precision, 
                          int batchSize, int deviceId, const std::string& version);

// Returns a deterministic signature of file contents and build configuration.
std::string computeFileSignature(const std::string& path, std::string_view buildDescriptor);
