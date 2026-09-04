#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "nn/backend_compat.h"

TEST(BackendCompatTest, HalfHasOnnxFloat16Layout) {
#if defined(HIVEMIND_BACKEND_TENSORRT) || defined(HIVEMIND_ORT_FP16)
    EXPECT_EQ(sizeof(__half), 2U);
#else
    EXPECT_EQ(sizeof(__half), sizeof(float));
#endif
}

TEST(BackendCompatTest, HalfRoundTripsExactlyRepresentableValues) {
    for (const float value : {
             0.0f,
             -0.0f,
             1.0f,
             -2.0f,
             65504.0f,
             0.00006103515625f,
             0.000000059604644775390625f,
         }) {
        EXPECT_FLOAT_EQ(__half2float(__float2half_rn(value)), value) << value;
    }
}

TEST(BackendCompatTest, HalfPreservesInfinityAndNan) {
    EXPECT_TRUE(std::isinf(__half2float(
        __float2half_rn(std::numeric_limits<float>::infinity()))));
    EXPECT_TRUE(std::isnan(__half2float(
        __float2half_rn(std::numeric_limits<float>::quiet_NaN()))));
}
