#pragma once
#include <string>

// Returns the path to the latest ONNX file in the given directory, or an empty string if none found.
std::string findLatestOnnxFile(const std::string& directory);

// Converts an ONNX path to a TensorRT engine path (.onnx -> .engine)
std::string getEnginePath(const std::string& onnxPath);
