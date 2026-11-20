// Game-, GPU- and SDK-free regression for the temporal upscalers' camera math.
//   clang++ -std=c++20 -O2 -Wall -Wextra -Werror gpu/vk/tests/dlss_math_test.cpp -o dlss_math_test
#include <cmath>
#include <cstdio>
#include <limits>

#include "../dlss_math.h"

#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);                                         \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static bool close(float a, float b, float epsilon = 1e-4f) { return std::fabs(a - b) <= epsilon; }

int main() {
    using namespace mkwvk::dlss_math;

    constexpr float near_z = 0.1f, far_z = 1000.0f;
    constexpr float pi = 3.14159265358979323846f;
    const float fov_y = 60.0f * pi / 180.0f;
    const float p11 = 1.0f / std::tan(fov_y * 0.5f);
    const float a = -(far_z + near_z) / (far_z - near_z);
    const float b = -2.0f * far_z * near_z / (far_z - near_z);

    CameraProjection finite;
    CHECK(camera_projection(p11, a, b, finite));
    CHECK(close(finite.near_z, near_z));
    // A finite far plane is recovered from two nearly equal float values (a and -1), so allow its matrix precision.
    CHECK(close(finite.far_z, far_z, 1.0f));
    CHECK(close(finite.fov_y, fov_y));
    CHECK(!finite.infinite);

    CameraProjection infinite;
    CHECK(camera_projection(p11, -1.0f, -2.0f * near_z, infinite));
    CHECK(close(infinite.near_z, near_z));
    CHECK(infinite.far_z == std::numeric_limits<float>::max());
    CHECK(infinite.infinite);

    CameraProjection invalid;
    CHECK(!camera_projection(0.0f, a, b, invalid));
    CHECK(!camera_projection(p11, 0.0f, 1.0f, invalid));

    const float clip_jitter[2] = {2.0f / 1920.0f, -2.0f / 1080.0f};
    float pixel_jitter[2];
    jitter_to_pixels(clip_jitter, 1920, 1080, pixel_jitter);
    CHECK(close(pixel_jitter[0], 1.0f));
    CHECK(close(pixel_jitter[1], 1.0f));

    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float result[16];
    CHECK(reprojection(identity, identity, result));
    for (int i = 0; i < 16; ++i) CHECK(close(result[i], identity[i]));
    const float singular[16]{};
    CHECK(!reprojection(singular, identity, result));
    return 0;
}
