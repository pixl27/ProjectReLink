// The item wheel's letters off Windows (gpu/vk/vk_wheel_text.h) with the bundled Libertinus Sans Bold: tests/test_wheel_text.py
#include "../gpu/vk/vk_wheel_text.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace mkwvk::wheel_text;

static int failures = 0;
static void check(const char* what, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}
// what vk_wheel.cpp's text_mask makes: the text with pad pixels and one more each side
struct Mask {
    int w = 0, h = 0, tw = 0, th = 0;
    std::vector<uint8_t> cov;
};
static Mask mask(const Face& f, const char* s, int px, int pad) {
    Mask m;
    measure(f, s, px, &m.tw, &m.th);
    m.w = m.tw + 2 * pad + 2, m.h = m.th + 2 * pad + 2;
    m.cov.assign((size_t)m.w * m.h, 0);
    draw(f, s, px, pad + 1, pad + 1, m.cov.data(), m.w, m.h);
    return m;
}
static long ink(const Mask& m) {
    long n = 0;
    for (uint8_t v : m.cov) n += v;
    return n;
}
// coverage on the mask's outer ring (the letters stay inside it)
static long ring(const Mask& m) {
    long n = 0;
    for (int y = 0; y < m.h; y++)
        for (int x = 0; x < m.w; x++)
            if (x == 0 || y == 0 || x == m.w - 1 || y == m.h - 1) n += m.cov[(size_t)y * m.w + x];
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    std::vector<unsigned char> data;
    if (FILE* fp = std::fopen(argv[1], "rb")) {
        unsigned char buf[65536];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) data.insert(data.end(), buf, buf + n);
        std::fclose(fp);
    }
    const Face f = face(data.data(), (unsigned)data.size());
    check("the bundled font reads", f.ok);

    // UTF-8: two-, three- and four-byte letters, and a broken sequence as '?'
    check("codepoints: ASCII", codepoints("Move") == std::vector<int>({'M', 'o', 'v', 'e'}));
    check("codepoints: a with diaeresis", codepoints("Best\xc3\xa4tigen")[4] == 0xE4);
    check("codepoints: three and four bytes", codepoints("\xe2\x82\xac\xf0\x9f\x98\x80") == std::vector<int>({0x20AC, 0x1F600}));
    check("codepoints: broken as '?'", codepoints("a\xc3z") == std::vector<int>({'a', '?', 'z'}));

    for (const char* s : {"Move", "Confirm", "Cancel", "Best\xc3\xa4tigen", "D\xc3\xa9placer", "Deku Stick"}) {
        const Mask m = mask(f, s, 28, 3);
        const std::string what = std::string("\"") + s + "\" at 28 px: ";
        check((what + "a size").c_str(), m.tw > 0 && m.th > 0);
        check((what + "letters drawn").c_str(), ink(m) > 0);
        check((what + "inside the mask").c_str(), ring(m) == 0);
        check((what + "a line of about the size").c_str(), m.th >= 28 && m.th <= 28 * 2);
    }
    // a wider text measures wider; twice the size, about twice the width
    int a = 0, b = 0, h = 0;
    measure(f, "Confirm", 24, &a, &h);
    measure(f, "Confirm Confirm", 24, &b, &h);
    check("a longer text is wider", b > a);
    measure(f, "Cancel", 20, &a, &h);
    measure(f, "Cancel", 40, &b, &h);
    check("twice the size, about twice the width", b >= 2 * a - 2 && b <= 2 * a + 2);
    // the accented letter is drawn: as wide as the plain one's word, give or take, not a missing letter's nothing
    measure(f, "Best\xc3\xa4tigen", 24, &a, &h);
    measure(f, "Bestatigen", 24, &b, &h);
    check("an accented letter has its width", a >= b - 1 && a <= b + 1);
    // the empty text: no width, no letters
    const Mask e = mask(f, "", 24, 2);
    check("the empty text: no width, no ink", e.tw == 0 && ink(e) == 0);
    // not a font: 0 by 0 and nothing drawn (vk_wheel.cpp's text_mask then makes the 0 by 0 mask the wheel skips)
    const unsigned char junk[64] = {1, 2, 3};
    const Face bad = face(junk, sizeof junk);
    measure(bad, "Move", 24, &a, &h);
    check("not a font: not ok, 0 by 0", !bad.ok && a == 0 && h == 0);
    check("no font: not ok", !face(nullptr, 0).ok);

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
