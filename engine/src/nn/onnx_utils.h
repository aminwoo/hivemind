#pragma once

#include <string>
#include <string_view>

// Resolves an ONNX model path from an explicit path or by scanning default model directories.
std::string resolveModelPath(const std::string& explicitPath = "");

// Returns the path to the latest ONNX file in the given directory, or an empty string if none found.
std::string findLatestOnnxFile(const std::string& directory);

// Whether an ONNX model contains quantize/dequantize (INT8) nodes.
bool isQuantizedOnnx(const std::string& onnxPath);

// Converts an ONNX path to a TensorRT engine path
// Format: [model_name]_[precision]_[batch_size]_[device]_[version].engine
// A quantized model's "fp16" precision is named "int8", and the precision is
// left out when the model name already contains it as a word ("net-int8").
std::string getEnginePath(const std::string& onnxPath, const std::string& precision, 
                          int batchSize, int deviceId, const std::string& version);

// Returns a deterministic signature of file contents and build configuration.
std::string computeFileSignature(const std::string& path, std::string_view buildDescriptor);
