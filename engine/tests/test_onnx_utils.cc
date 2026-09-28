#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "nn/onnx_utils.h"
#include "Fairy-Stockfish/src/misc.h"

TEST(OnnxUtilsTest, ResolveModelPathReturnsExplicitPathWhenProvided) {
    EXPECT_EQ(resolveModelPath("/custom/path/model.onnx"), "/custom/path/model.onnx");
}

TEST(OnnxUtilsTest, MissingDirectoryHasNoLatestModel) {
    const auto missingDirectory = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-missing-directory";
    std::filesystem::remove_all(missingDirectory);

    EXPECT_TRUE(findLatestOnnxFile(missingDirectory.string()).empty());
}

TEST(OnnxUtilsTest, ResolveModelPathSearchesBesideExecutable) {
    const auto bundle = std::filesystem::temp_directory_path()
        / "hivemind-onnx-utils-bundle";
    const auto models = bundle / "models";
    const auto modelPath = models / "bundled.onnx";
    std::filesystem::remove_all(bundle);
    std::filesystem::create_directories(models);
    {
        std::ofstream model(modelPath);
        model << "test";
    }

    const std::string previous = Stockfish::CommandLine::binaryDirectory;
    Stockfish::CommandLine::binaryDirectory = bundle.string();
    EXPECT_EQ(std::filesystem::path(resolveModelPath()), modelPath);
    Stockfish::CommandLine::binaryDirectory = previous;
    std::filesystem::remove_all(bundle);
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

TEST(OnnxUtilsTest, QuantizedModelPlansAreNamedInt8) {
    const std::filesystem::path directory = std::filesystem::temp_directory_path();
    const std::filesystem::path plain = directory / "hivemind_plain_test.onnx";
    const std::filesystem::path quantized = directory / "hivemind_quantized_test.onnx";
    {
        std::ofstream(plain, std::ios::binary) << std::string(3 << 20, 'x') << "Conv";
        // Place the op type across the scanner's 1 MiB chunk boundary.
        std::ofstream(quantized, std::ios::binary)
            << std::string((1 << 20) - 5, 'x') << "DequantizeLinear" << std::string(100, 'x');
    }
    EXPECT_FALSE(isQuantizedOnnx(plain.string()));
    EXPECT_TRUE(isQuantizedOnnx(quantized.string()));
    EXPECT_NE(getEnginePath(plain.string(), "fp16", 8, 0, "v3").find("_fp16_b8_"), std::string::npos);
    EXPECT_NE(getEnginePath(quantized.string(), "fp16", 8, 0, "v3").find("_int8_b8_"), std::string::npos);

    // The precision is not repeated when the model name already carries it,
    // and only the true precision counts.
    const std::filesystem::path namedInt8 = directory / "net-int8-v2.onnx";
    const std::filesystem::path namedFp16 = directory / "net_fp16.onnx";
    const std::filesystem::path misnamed = directory / "net-int8.onnx";
    std::filesystem::copy_file(quantized, namedInt8, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(plain, namedFp16, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(plain, misnamed, std::filesystem::copy_options::overwrite_existing);
    const auto name = [](const std::filesystem::path& path) {
        return std::filesystem::path(getEnginePath(path.string(), "fp16", 8, 0, "v3")).filename().string();
    };
    EXPECT_EQ(name(namedInt8), "net-int8-v2_b8_gpu0_v3.engine");
    EXPECT_EQ(name(namedFp16), "net_fp16_b8_gpu0_v3.engine");
    EXPECT_EQ(name(misnamed), "net-int8_fp16_b8_gpu0_v3.engine");
    for (const auto& path : {plain, quantized, namedInt8, namedFp16, misnamed}) {
        std::filesystem::remove(path);
    }
}
