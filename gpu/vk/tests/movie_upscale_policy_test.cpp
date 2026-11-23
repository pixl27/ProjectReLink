// Game- and GPU-free regression for the movie-network policy.
//   clang++ -std=c++20 -O2 -Wall -Wextra -Werror gpu/vk/tests/movie_upscale_policy_test.cpp -o movie_upscale_policy_test
#include <cstdio>

#include "../movie_upscale_policy.h"

#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);                                        \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    using mkwvk::movie_upscale_policy::Mode;
    using mkwvk::movie_upscale_policy::parse;
    using mkwvk::movie_upscale_policy::use_linear;

    CHECK(parse(nullptr) == Mode::Automatic);
    CHECK(parse("") == Mode::Automatic);
    CHECK(parse("automatic") == Mode::Automatic);
    CHECK(parse("linear") == Mode::Linear);

    CHECK(use_linear(Mode::Linear, false, false, true, false));
    CHECK(!use_linear(Mode::Linear, false, false, true, true));
    CHECK(!use_linear(Mode::Linear, false, false, false, false));
    CHECK(!use_linear(Mode::Linear, false, true, true, false));

    CHECK(use_linear(Mode::Automatic, true, false, true, false));
    CHECK(!use_linear(Mode::Automatic, false, false, true, false));
    return 0;
}
