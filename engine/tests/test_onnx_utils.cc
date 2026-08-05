#include <gtest/gtest.h>

#include <filesystem>

#include "onnx_utils.h"

TEST(OnnxUtilsTest, MissingDirectoryHasNoLatestModel) {
    const auto missingDirectory = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-missing-directory";
    std::filesystem::remove_all(missingDirectory);

    EXPECT_TRUE(findLatestOnnxFile(missingDirectory.string()).empty());
}