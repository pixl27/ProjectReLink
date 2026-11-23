// Hookshot native engine: two narrow-phase leaves (docs/native-engine/physics-loops.md section 12): the support point of
// a small vertex set along a direction (guest 0x817472e0) and the relative placement of two precise transforms (guest
// 0x817036a0). Each computes what the guest function computes, operation for operation (hl/math/Exact.h), or returns
// false having written nothing when a value is one the exactness rules do not cover.
#ifndef HL_MATH_SUPPORT_H
#define HL_MATH_SUPPORT_H

#include <cstdint>

#include "hl/math/Bounds.h"

namespace hl::math {

// The constants the guest function reads from its module's data, given by the OS layer from the guest's memory at each call
// (so the native code assumes none of them): the jump table's bytes by n & 3 (0 for a count of 1 mod 4, 14 for 2, 40 for 3,
// 88 for 0), the large negative the two-vertex case pads its lanes with, the lane indices 0..3 and the step added to
// them each round, and the table that turns the three-vertex case's comparison mask into an index.
struct SupportTables {
    uint8_t jump[4];
    float pad;
    uint32_t index[4];
    uint32_t step[4];
    int8_t select[8];
};

// The vertex of `vertices` (n of them, 12 bytes each, x y z) with the largest dot product with dir (x, y, z, each product rounded, summed as
// (x dx + y dy) + z dz): the first of equal ones within a lane, the first lane among equal pairs. out receives that vertex's
// x, y, z and, in the fourth word, its index with 0x3f000000 or-ed in. Gives up for n < 1 or a jump byte it does not know.
bool support_point(const uint8_t* vertices, int32_t n, const float dir[4], const SupportTables& t, uint8_t out[16]);

// The placement of transform b relative to transform a (guest 0x817036a0: x1 = a, x2 = b, x0 = the result): the rotation
// rows as a's columns times b's rows (each lane (c2 b_z + (c0 b_x + c1 b_y))), and the translation (b's high + low minus
// a's high + low, in binary64) rotated by a's rows and split again into a float and the float's remainder. The result is
// 80 bytes: the rows (48), the high part (x, y, z, z), the low part (3 floats) and a zero word.
struct RelativeTransform {
    float row[3][4];
    float hi[4];
    float lo[3];
    uint32_t zero;
};
static_assert(sizeof(RelativeTransform) == 80);
bool relative_transform(const PreciseTransform& a, const PreciseTransform& b, RelativeTransform& out);

}   // namespace hl::math

#endif
