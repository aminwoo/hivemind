#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "onnx_utils.h"

TEST(OnnxUtilsTest, MissingDirectoryHasNoLatestModel) {
    const auto missingDirectory = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-missing-directory";
    std::filesystem::remove_all(missingDirectory);

    EXPECT_TRUE(findLatestOnnxFile(missingDirectory.string()).empty());
}

TEST(OnnxUtilsTest, FileSignatureIncludesContentsAndBuildDescriptor) {
    const auto modelPath = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-signature.onnx";
    {
        std::ofstream model(modelPath, std::ios::binary | std::ios::trunc);
        model << "model-a";
    }

    const std::string original = computeFileSignature(modelPath.string(), "config-a");
    EXPECT_FALSE(original.empty());
    EXPECT_EQ(original, computeFileSignature(modelPath.string(), "config-a"));
    EXPECT_NE(original, computeFileSignature(modelPath.string(), "config-b"));

    {
        std::ofstream model(modelPath, std::ios::binary | std::ios::trunc);
        model << "model-b";
    }
    EXPECT_NE(original, computeFileSignature(modelPath.string(), "config-a"));

    std::filesystem::remove(modelPath);
}

TEST(OnnxUtilsTest, MissingFileHasNoSignature) {
    const auto missingPath = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-missing-model.onnx";
    std::filesystem::remove(missingPath);
    EXPECT_TRUE(computeFileSignature(missingPath.string(), "config").empty());
}