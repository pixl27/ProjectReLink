// (relink, from Longshot) The item wheel, Twilight Princess style: host/oot_wheel.c's state (the page's 8 slots, the equipped
// item, the stick's pick, the item being moved) drawn here, on the CPU, into a picture of the game's picture's size.
// vk_framegen.cpp framegen_wheel_backdrop lays it over the scene, blurred and dimmed (the game's own item row and HUD gone).
// Longshot draws it with Dear ImGui; ReLink's menu is drawn with GDI, so this draws its shapes itself, anti-aliased by their
// distance to each pixel, and its text with GDI (white on black, the letters' coverage).
//
// The game's own icons and names, made from the player's RomFS at every build (nx2/wheel_assets.py, nx2/wheel_names.py) into
// <game folder>/wheel/: icons.bin keeps the game's ASTC blocks, decoded here once by suyu's decoder; names.bin each item's
// name in the game's languages, the one in language.txt taken. Without them the slots show the host's English labels.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "vk_paths.h"
#include "vk_wheel.h"
#include "vk_internal.h"
#include "vk_menu.h"
#include "video_core/textures/astc.h"

namespace mkwvk {
namespace {

// ---- what is drawn on: premultiplied RGBA, float while drawing ----
struct Rgba {
    float r, g, b, a;   // (straight colour, 0..1; a its opacity)
};
constexpr Rgba rgba(int r, int g, int b, int a = 255) { return {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f}; }

struct Canvas {
    int w = 0, h = 0;
    std::vector<float> px;   // (premultiplied, 4 a pixel)
    void reset(int nw, int nh) {
        w = nw, h = nh;
        px.assign((size_t)w * h * 4, 0.0f);
    }
    // colour c over the pixel, with coverage cov (0..1)
    void over(int x, int y, const Rgba& c, float cov) {
        if (x < 0 || y < 0 || x >= w || y >= h || cov <= 0) return;
        const float a = c.a * std::min(cov, 1.0f);
        float* p = &px[((size_t)y * w + x) * 4];
        p[0] = c.r * a + p[0] * (1 - a), p[1] = c.g * a + p[1] * (1 - a), p[2] = c.b * a + p[2] * (1 - a);
        p[3] = a + p[3] * (1 - a);
    }
    // a premultiplied pixel over it (a picture's)
    void over_pm(int x, int y, float r, float g, float b, float a) {
        if (x < 0 || y < 0 || x >= w || y >= h || a <= 0) return;
        float* p = &px[((size_t)y * w + x) * 4];
        p[0] = r + p[0] * (1 - a), p[1] = g + p[1] * (1 - a), p[2] = b + p[2] * (1 - a), p[3] = a + p[3] * (1 - a);
    }
};

// coverage of a pixel whose centre is d past an edge (d < 0 inside): a pixel wide ramp
inline float edge_cov(float d) { return std::clamp(0.5f - d, 0.0f, 1.0f); }

// a pixel box around a shape [x0, x1) x [y0, y1), on the canvas
struct Box {
    int x0, y0, x1, y1;
};
Box box_of(const Canvas& c, float l, float t, float r, float b) {
    return {std::max(0, (int)std::floor(l) - 1), std::max(0, (int)std::floor(t) - 1), std::min(c.w, (int)std::ceil(r) + 2),
            std::min(c.h, (int)std::ceil(b) + 2)};
}

void disc(Canvas& c, float cx, float cy, float r, const Rgba& col) {
    if (r <= 0) return;
    const Box b = box_of(c, cx - r, cy - r, cx + r, cy + r);
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++) c.over(x, y, col, edge_cov(std::hypot(x + 0.5f - cx, y + 0.5f - cy) - r));
}
void ring(Canvas& c, float cx, float cy, float r, float thick, const Rgba& col) {
    const float o = r + thick / 2;
    const Box b = box_of(c, cx - o, cy - o, cx + o, cy + o);
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++)
            c.over(x, y, col, edge_cov(std::fabs(std::hypot(x + 0.5f - cx, y + 0.5f - cy) - r) - thick / 2));
}
// a disc of radius r with an edge of e pixels
void disc_edged(Canvas& c, float cx, float cy, float r, float e, const Rgba& fill, const Rgba& edge) {
    disc(c, cx, cy, r, edge);
    disc(c, cx, cy, std::max(0.0f, r - e), fill);
}
// a bar with round ends: half height hh, from x0 to x1 (the ends' centres hh in), around cy
void stadium(Canvas& c, float x0, float x1, float cy, float hh, const Rgba& col) {
    const float a = x0 + hh, z = std::max(a, x1 - hh);
    const Box b = box_of(c, x0, cy - hh, x1, cy + hh);
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++) {
            const float px = x + 0.5f, py = y + 0.5f, qx = std::clamp(px, a, z);
            c.over(x, y, col, edge_cov(std::hypot(px - qx, py - cy) - hh));
        }
}
void segment(Canvas& c, float ax, float ay, float bx, float by, float thick, const Rgba& col) {
    const Box b = box_of(c, std::min(ax, bx) - thick, std::min(ay, by) - thick, std::max(ax, bx) + thick, std::max(ay, by) + thick);
    const float dx = bx - ax, dy = by - ay, len2 = std::max(1e-6f, dx * dx + dy * dy);
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++) {
            const float px = x + 0.5f - ax, py = y + 0.5f - ay, t = std::clamp((px * dx + py * dy) / len2, 0.0f, 1.0f);
            c.over(x, y, col, edge_cov(std::hypot(px - t * dx, py - t * dy) - thick / 2));
        }
}
void triangle(Canvas& c, float x0, float y0, float x1, float y1, float x2, float y2, const Rgba& col) {
    const float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if (std::fabs(area) < 1e-3f) return;
    const float s = area > 0 ? 1.0f : -1.0f;
    const float px[3] = {x0, x1, x2}, py[3] = {y0, y1, y2};
    const Box b = box_of(c, std::min({x0, x1, x2}), std::min({y0, y1, y2}), std::max({x0, x1, x2}), std::max({y0, y1, y2}));
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++) {
            float d = -1e9f;   // (the distance outside the farthest edge: inside all three, negative)
            for (int k = 0; k < 3; k++) {
                const float ex = px[(k + 1) % 3] - px[k], ey = py[(k + 1) % 3] - py[k], l = std::hypot(ex, ey);
                d = std::max(d, -s * (ex * (y + 0.5f - py[k]) - ey * (x + 0.5f - px[k])) / l);
            }
            c.over(x, y, col, edge_cov(d));
        }
}

// ---- the game's icons (wheel/icons.bin) ----
struct Icon {
    std::string name;
    int size = 0;
    std::vector<float> pm;   // (premultiplied RGBA, size x size)
    float off_x = 0, off_y = 0;   // (where the drawing's middle is from the picture's, in pictures: the bow's and hammer's)
};
std::vector<Icon> icons;
constexpr int kIconPx = 128;   // (each icon scaled to this once: the wheel draws them at most about 110 pixels at 4K)

std::string game_file(const char* rel) {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    return path.substr(0, path.find_last_of("\\/") + 1) + rel;
}

void icons_load() {
    static bool tried = false;
    if (tried) return;
    tried = true;
    const std::string path = game_file("wheel/icons.bin");
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        logf("wheel: no icons (%s): the slots show names", path.c_str());
        return;
    }
    char magic[4];
    uint32_t hd[2];
    if (fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "HWIA", 4) && fread(hd, 4, 2, f) == 2 && hd[0] == 1 && hd[1] <= 1024) {
        for (uint32_t i = 0; i < hd[1]; i++) {
            char name[48];
            uint32_t d[4];   // (format, width, height, bytes)
            if (fread(name, 1, sizeof name, f) != sizeof name || fread(d, 4, 4, f) != 4 || d[3] > (64u << 20)) break;
            // (an icon of a size this does not take is passed over, not the end of the file: the hammer's is 1440 pixels)
            if (!d[1] || !d[2] || d[1] > 4096 || d[2] > 4096 || d[3] != ((d[1] + 3) / 4) * ((d[2] + 3) / 4) * 16) {
                if (fseek(f, (long)d[3], SEEK_CUR)) break;
                continue;
            }
            std::vector<uint8_t> blocks(d[3]);
            if (fread(blocks.data(), 1, blocks.size(), f) != blocks.size()) break;
            name[sizeof name - 1] = 0;
            const uint32_t w = d[1], h = d[2];
            std::vector<uint8_t> rgba((size_t)w * h * 4);
            try {
                Tegra::Texture::ASTC::DecompressSerial(blocks, w, h, 1, 4, 4, rgba);
            } catch (...) {
                continue;
            }
            // (to kIconPx, each output pixel the average of the source pixels under it, premultiplied)
            Icon ic;
            ic.name = name;
            ic.size = kIconPx;
            ic.pm.assign((size_t)kIconPx * kIconPx * 4, 0.0f);
            for (int y = 0; y < kIconPx; y++)
                for (int x = 0; x < kIconPx; x++) {
                    const uint32_t sx0 = x * w / kIconPx, sx1 = std::max(sx0 + 1, (uint32_t)(x + 1) * w / kIconPx);
                    const uint32_t sy0 = y * h / kIconPx, sy1 = std::max(sy0 + 1, (uint32_t)(y + 1) * h / kIconPx);
                    float s[4] = {0, 0, 0, 0};
                    for (uint32_t sy = sy0; sy < sy1; sy++)
                        for (uint32_t sx = sx0; sx < sx1; sx++) {
                            const uint8_t* q = &rgba[((size_t)sy * w + sx) * 4];
                            const float a = q[3] / 255.0f;
                            s[0] += q[0] / 255.0f * a, s[1] += q[1] / 255.0f * a, s[2] += q[2] / 255.0f * a, s[3] += a;
                        }
                    const float k = 1.0f / ((sx1 - sx0) * (sy1 - sy0));
                    for (int c = 0; c < 4; c++) ic.pm[((size_t)y * kIconPx + x) * 4 + c] = s[c] * k;
                }
            // (the drawing's box, for the icons whose drawing sits off the picture's middle: the bow and the hammer)
            if (ic.name == "Weapon_Range_Bow" || ic.name == "Item_Hammer") {
                int x0 = kIconPx, y0 = kIconPx, x1 = -1, y1 = -1;
                for (int y = 0; y < kIconPx; y++)
                    for (int x = 0; x < kIconPx; x++)
                        if (ic.pm[((size_t)y * kIconPx + x) * 4 + 3] > 0.1f)
                            x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
                if (x1 >= x0 && y1 >= y0)
                    ic.off_x = ((x0 + x1 + 1) * 0.5f - kIconPx * 0.5f) / kIconPx, ic.off_y = ((y0 + y1 + 1) * 0.5f - kIconPx * 0.5f) / kIconPx;
            }
            icons.push_back(std::move(ic));
        }
    }
    fclose(f);
    logf("wheel: %d icons from %s", (int)icons.size(), path.c_str());
}
// the game's name of a wheel item: a second bottle of the same content is "<name>#2" (host/oot_wheel.c plain_name)
std::string plain(const char* name) {
    const char* mark = std::strchr(name, '#');
    return mark ? std::string(name, mark - name) : std::string(name);
}
const Icon* icon_of(const char* name) {
    const std::string n = plain(name);
    for (const Icon& ic : icons)
        if (ic.name == n) return &ic;
    return nullptr;
}
// the icon side pixels wide centred on (mx, my), its opacity times alpha (bilinear)
void draw_icon(Canvas& c, const Icon& ic, float mx, float my, float side, float alpha = 1) {
    if (side <= 1) return;
    mx -= ic.off_x * side, my -= ic.off_y * side;
    const float l = mx - side / 2, t = my - side / 2, k = ic.size / side;
    const Box b = box_of(c, l, t, l + side, t + side);
    for (int y = b.y0; y < b.y1; y++)
        for (int x = b.x0; x < b.x1; x++) {
            const float u = (x + 0.5f - l) * k - 0.5f, v = (y + 0.5f - t) * k - 0.5f;
            if (u < -0.5f || v < -0.5f || u > ic.size - 0.5f || v > ic.size - 0.5f) continue;
            const int u0 = (int)std::floor(u), v0 = (int)std::floor(v);
            const float fu = u - u0, fv = v - v0;
            float s[4] = {0, 0, 0, 0};
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++) {
                    const int su = std::clamp(u0 + i, 0, ic.size - 1), sv = std::clamp(v0 + j, 0, ic.size - 1);
                    const float wgt = (i ? fu : 1 - fu) * (j ? fv : 1 - fv);
                    const float* q = &ic.pm[((size_t)sv * ic.size + su) * 4];
                    for (int ch = 0; ch < 4; ch++) s[ch] += q[ch] * wgt;
                }
            c.over_pm(x, y, s[0] * alpha, s[1] * alpha, s[2] * alpha, s[3] * alpha);
        }
}

// ---- the items' names in the game's language (wheel/names.bin) ----
std::vector<std::pair<std::string, std::string>> names;
std::string language_code() {   // (language.txt's code line: the first one not a note)
    std::string code = "en-US";
    if (FILE* f = fopen(game_file("language.txt").c_str(), "r")) {
        char line[128];
        while (fgets(line, sizeof line, f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
            size_t a = 0;
            while (a < s.size() && (s[a] == ' ' || s[a] == '\t')) a++;
            s = s.substr(a);
            if (s.empty() || s[0] == '#') continue;
            code = s;
            break;
        }
        fclose(f);
    }
    return code;
}
bool same_code(const char* a, const std::string& b) {
    if (std::strlen(a) != b.size()) return false;
    for (size_t i = 0; i < b.size(); i++)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    return true;
}
std::string lang;
void names_load() {
    static bool tried = false;
    if (tried) return;
    tried = true;
    lang = language_code();
    const std::string path = game_file("wheel/names.bin");
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        logf("wheel: no names (%s): the items' English names", path.c_str());
        return;
    }
    char magic[4];
    uint32_t hd[2];
    if (fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "HWNM", 4) && fread(hd, 4, 2, f) == 2 && hd[0] == 1 && hd[1] <= 64) {
        for (uint32_t l = 0; l < hd[1]; l++) {
            char code[9] = {};
            uint32_t count = 0;
            if (fread(code, 1, 8, f) != 8 || fread(&count, 4, 1, f) != 1 || count > 4096) break;
            const bool mine = same_code(code, lang);
            for (uint32_t i = 0; i < count; i++) {
                char rec[112];
                if (fread(rec, 1, sizeof rec, f) != sizeof rec) break;
                if (!mine) continue;
                rec[47] = 0, rec[111] = 0;
                names.emplace_back(rec, rec + 48);
            }
            if (mine) break;
        }
    }
    fclose(f);
    logf("wheel: %d names in %s from %s", (int)names.size(), lang.c_str(), path.c_str());
}
const char* name_of(const char* name, const char* label) {
    const std::string p = plain(name);
    for (const auto& n : names)
        if (n.first == p) return n.second.c_str();
    return label;
}
// the prompts' words in the game's language (the languages written with the menu's letters; English otherwise)
const char* word(int which) {   // 0 Move, 1 Confirm, 2 Cancel
    struct W { const char* code; const char* w[3]; };
    static const W t[] = {{"pt-BR", {"Mover", "Confirmar", "Cancelar"}}, {"es", {"Mover", "Confirmar", "Cancelar"}},
                          {"es-419", {"Mover", "Confirmar", "Cancelar"}}, {"fr", {"D\xc3\xa9placer", "Confirmer", "Annuler"}},
                          {"fr-CA", {"D\xc3\xa9placer", "Confirmer", "Annuler"}}, {"de", {"Verschieben", "Best\xc3\xa4tigen", "Abbrechen"}},
                          {"it", {"Sposta", "Conferma", "Annulla"}}, {"nl", {"Verplaatsen", "Bevestigen", "Annuleren"}}};
    for (const W& e : t)
        if (same_code(e.code, lang)) return e.w[which];
    static const char* en[3] = {"Move", "Confirm", "Cancel"};
    return en[which];
}

// ---- text: GDI's letters, white on black, as coverage ----
struct TextMask {
    int w = 0, h = 0;
    std::vector<uint8_t> cov;
};
HFONT text_font(int px) {
    return CreateFontA(-px, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
                       DEFAULT_PITCH, "Segoe UI");
}
#ifdef _WIN32
std::wstring wide(const char* s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}
#else
std::string latin1(const char* s) {   // (the POSIX GDI stand-in draws a byte a letter: Latin-1, '?' past it)
    std::string o;
    for (const unsigned char* p = (const unsigned char*)s; *p;) {
        unsigned cp = *p, n = cp < 0x80 ? 1 : (cp >> 5) == 6 ? 2 : (cp >> 4) == 14 ? 3 : 4;
        if (n == 2 && p[1]) cp = (cp & 31) << 6 | (p[1] & 63);
        for (unsigned k = 0; k < n && *p; k++) p++;
        o += cp < 256 && (n <= 2) ? (char)cp : '?';
    }
    return o;
}
#endif
// the size of text at px pixels
void text_size(const char* s, int px, int* w, int* h) {
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT f = text_font(px);
    HGDIOBJ old = SelectObject(dc, f);
    SIZE sz{};
#ifdef _WIN32
    const std::wstring t = wide(s);
    GetTextExtentPoint32W(dc, t.c_str(), (int)t.size(), &sz);
#else
    const std::string t = latin1(s);
    GetTextExtentPoint32A(dc, t.c_str(), (int)t.size(), &sz);
#endif
    SelectObject(dc, old);
    DeleteObject(f);
    DeleteDC(dc);
    *w = (int)sz.cx, *h = (int)sz.cy;
}
TextMask text_mask(const char* s, int px, int pad) {
    TextMask m;
    int tw = 0, th = 0;
    text_size(s, px, &tw, &th);
    m.w = tw + 2 * pad + 2, m.h = th + 2 * pad + 2;
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = m.w;
    bi.bmiHeader.biHeight = -m.h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC dc = CreateCompatibleDC(nullptr);
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bmp || !bits) {
        if (bmp) DeleteObject(bmp);
        DeleteDC(dc);
        m.w = m.h = 0;
        return m;
    }
    HGDIOBJ old_bmp = SelectObject(dc, bmp);
    std::memset(bits, 0, (size_t)m.w * m.h * 4);
    HFONT f = text_font(px);
    HGDIOBJ old = SelectObject(dc, f);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 255));
#ifdef _WIN32
    const std::wstring t = wide(s);
    TextOutW(dc, pad + 1, pad + 1, t.c_str(), (int)t.size());
#else
    const std::string t = latin1(s);
    TextOutA(dc, pad + 1, pad + 1, t.c_str(), (int)t.size());
#endif
    GdiFlush();
    m.cov.resize((size_t)m.w * m.h);
    const uint32_t* p = (const uint32_t*)bits;
    for (size_t i = 0; i < m.cov.size(); i++) {
        const uint32_t v = p[i];
        m.cov[i] = (uint8_t)std::max({v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF});
    }
    SelectObject(dc, old);
    DeleteObject(f);
    SelectObject(dc, old_bmp);
    DeleteObject(bmp);
    DeleteDC(dc);
    return m;
}
// text with a dark outline o pixels wide, centred on (cx, cy)
void text(Canvas& c, const char* s, int px, float cx, float cy, const Rgba& col, int o) {
    if (!*s) return;
    const TextMask m = text_mask(s, px, o);
    if (!m.w) return;
    const int x0 = (int)std::lround(cx - m.w / 2.0f), y0 = (int)std::lround(cy - m.h / 2.0f);
    const Rgba shade = rgba(0, 0, 0, 230);
    for (int y = 0; y < m.h; y++)
        for (int x = 0; x < m.w; x++) {
            int around = 0;   // (the outline: the letters spread o pixels each way)
            for (int dy = -o; dy <= o && around < 255; dy += std::max(1, o))
                for (int dx = -o; dx <= o; dx += std::max(1, o)) {
                    const int sx = x + dx, sy = y + dy;
                    if (sx >= 0 && sy >= 0 && sx < m.w && sy < m.h) around = std::max(around, (int)m.cov[(size_t)sy * m.w + sx]);
                }
            if (around) c.over(x0 + x, y0 + y, shade, around / 255.0f);
            if (const int v = m.cov[(size_t)y * m.w + x]) c.over(x0 + x, y0 + y, col, v / 255.0f);
        }
}
int text_width(const char* s, int px) {
    int w = 0, h = 0;
    text_size(s, px, &w, &h);
    return w;
}
// a label centred on (cx, cy), split in two lines at the space nearest its middle when it is wider than width
void label(Canvas& c, const char* s, int px, float cx, float cy, float width, const Rgba& col, int o) {
    const int n = (int)std::strlen(s);
    int cut = -1;
    if (text_width(s, px) > width)
        for (int i = 0; i < n; i++)
            if (s[i] == ' ' && (cut < 0 || std::abs(i - n / 2) < std::abs(cut - n / 2))) cut = i;
    if (cut < 0) return text(c, s, px, cx, cy, col, o);
    const std::string a(s, cut), b(s + cut + 1);
    const float line = px * 1.05f;
    text(c, a.c_str(), px, cx, cy - line / 2, col, o);
    text(c, b.c_str(), px, cx, cy + line / 2, col, o);
}

// ---- the wheel ----
MkwVkWheel now;               // (the state drawn: wheel_update)
float spread = 1.0f;          // (the opening's animation: the slots' distance from the centre, 0.35 to 1)
constexpr float kBannerGap = 3, kBannerHalf = 3;   // (units: the gap above the name banner, its half height)

// This frame's wheel from the host (a seqlock: odd while the host writes it); false when it is not shown. *changed: the
// state or the opening's spread is not the last frame's
bool wheel_update(bool* changed) {
    *changed = false;
    if (!opts.item_wheel && !getenv("HOS_ITEM_WHEEL")) return false;
    const MkwVkWheel* src = host && host->wheel ? host->wheel() : nullptr;
    constexpr int64_t kSpreadNs = 160000000;
    static int64_t opened_at = 0;
    static bool shown = false;
    static uint32_t drawn_seq = ~0u;
    if (!src) return shown = false;
    const uint32_t s1 = __atomic_load_n(&src->seq, __ATOMIC_ACQUIRE);
    if (!(s1 & 1)) {
        MkwVkWheel copy;
        std::memcpy(&copy, (const void*)src, sizeof copy);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (__atomic_load_n(&src->seq, __ATOMIC_ACQUIRE) == s1) {   // (else written meanwhile: the last frame's again)
            copy.count = std::min(copy.count, (int32_t)MKWVK_WHEEL_MAX);
            copy.held_name[sizeof copy.held_name - 1] = 0;
            for (int i = 0; i < copy.count; i++) copy.label[i][sizeof copy.label[i] - 1] = 0, copy.name[i][sizeof copy.name[i] - 1] = 0;
            if (s1 != drawn_seq) *changed = true, drawn_seq = s1;
            now = copy;
            shown = copy.open && copy.count > 0;
        }
    }
    if (!shown) {
        opened_at = 0;
        return false;
    }
    const int64_t t_now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (!opened_at) opened_at = t_now, *changed = true;
    const double t = std::min(1.0, (double)(t_now - opened_at) / kSpreadNs);
    const float s = (float)(0.35 + 0.65 * (1 - (1 - t) * (1 - t) * (1 - t)));   // (ease out)
    if (s != spread) *changed = true;
    spread = s;
    return true;
}

void draw(Canvas& c, float mx, float my, float u, float pic_w) {
    icons_load();
    names_load();
    const MkwVkWheel& wh = now;
    auto lbl = [&](int i) { return name_of(wh.name[i], wh.label[i]); };
    const int n = std::min((int)wh.count, (int)MKWVK_WHEEL_SLOTS);
    // ring_r: the slots' centres' radius, slot_r a slot's, centre_r the hub's (the host's slot_under takes the hub as 12 units)
    const float centre_r = 12 * u, ring_r = 27 * u;
    const float slot_r = (float)std::min(8.0 * u, ring_r * std::sin(3.14159265358979 / MKWVK_WHEEL_SLOTS) * 0.9);
    auto slot_at = [&](int i, float* x, float* y) {
        const double a = 2 * 3.14159265358979 * i / MKWVK_WHEEL_SLOTS, r = ring_r * (double)spread;
        *x = mx + (float)(std::sin(a) * r), *y = my - (float)(std::cos(a) * r);
    };
    const Rgba dark = rgba(22, 24, 34), rim = rgba(120, 112, 88), gold = rgba(236, 196, 84), gold_dark = rgba(70, 56, 18),
               white = rgba(244, 244, 236), line = rgba(84, 78, 62), faint = rgba(22, 24, 34, 110), faint_rim = rgba(120, 112, 88, 90),
               move_col = rgba(110, 170, 240);
    const float thick = std::max(2.0f, u / 2), edge = std::max(2.0f, std::floor(u / 3));
    const int o = (int)std::max(1.0f, std::floor(u / 5));
    ring(c, mx, my, ring_r * spread, thick, line);
    disc_edged(c, mx, my, centre_r, edge, dark, gold);
    auto has = [&](int i) { return i >= 0 && i < n && wh.name[i][0] != 0; };
    const bool moving = wh.held_name[0] != 0;   // (an item taken up with Y: in the hub, its slot faint, the ring where it would go)
    const bool on_empty = wh.pick >= 0 && wh.pick < n && !has(wh.pick);
    const int focus = moving || on_empty ? -1 : has(wh.pick) ? wh.pick : has(wh.cursor) ? wh.cursor : -1;
    // (while an item is held and pointed somewhere else, the move as it would come out: the held item in the slot pointed at,
    // and what was there, faint, in the slot it came from; the two slots' names and counts left out)
    const int target = moving && wh.pick >= 0 && wh.pick < n && wh.pick != wh.held ? wh.pick : -1;
    auto in_move = [&](int i) { return moving && (i == target || i == wh.held); };
    for (int i = 0; i < n; i++) {
        float px, py;
        slot_at(i, &px, &py);
        const char* shown = i == target ? wh.held_name : target >= 0 && i == wh.held ? wh.name[target] : wh.name[i];
        if (!shown[0]) {   // (an empty slot: there, so the eight directions read as eight; ringed when pointed at)
            disc_edged(c, px, py, slot_r * 0.8f, edge, faint, moving && i == wh.held ? move_col : faint_rim);
            if (i == wh.pick)
                ring(c, px, py, slot_r * 0.8f + std::max(1.0f, u / 4), std::max(2.0f, u / 2), moving ? move_col : white);
            continue;
        }
        const bool on = i == wh.cursor && !in_move(i), picked = i == wh.pick;
        if (picked) disc(c, px, py, slot_r + std::max(2.0f, u / 2), moving ? move_col : white);
        if (moving && i == wh.held) {   // (the slot it was taken from: faint, with what would go there)
            disc_edged(c, px, py, slot_r, edge, faint, move_col);
            if (const Icon* ic = icon_of(shown)) draw_icon(c, *ic, px, py, slot_r * 1.38f, target >= 0 ? 0.55f : 0.27f);
            continue;
        }
        disc_edged(c, px, py, slot_r, edge, on ? gold_dark : dark, on ? gold : rim);
        if (const Icon* ic = icon_of(shown)) draw_icon(c, *ic, px, py, slot_r * 1.38f);
    }
    const Icon* focus_icon = moving ? icon_of(wh.held_name) : focus >= 0 ? icon_of(wh.name[focus]) : nullptr;
    if (focus_icon) draw_icon(c, *focus_icon, mx, my, centre_r * 1.3f);
    const int small = (int)std::max(10.0f, std::round(1.7f * u)), big = (int)std::max(14.0f, std::round(2.4f * u)),
              name_px = (int)std::max(16.0f, std::round(3.3f * u)), count_px = (int)std::max(12.0f, std::round(slot_r * 0.52f));
    for (int i = 0; i < n; i++) {   // (names only where the game has no icon for the item)
        if (!has(i) || icon_of(wh.name[i]) || in_move(i)) continue;
        float px, py;
        slot_at(i, &px, &py);
        label(c, lbl(i), small, px, py, 2 * slot_r - u, i == wh.cursor ? gold : white, o);
    }
    for (int i = 0; i < n; i++) {   // (the counts, as the game's HUD shows them: at a slot's lower right, grey at none left)
        if (!has(i) || wh.amount[i] < 0 || in_move(i)) continue;
        float px, py;
        slot_at(i, &px, &py);
        char num[12];
        snprintf(num, sizeof num, "%d", (int)wh.amount[i]);
        const float tw = (float)text_width(num, count_px);
        text(c, num, count_px, px + slot_r - tw / 2, py + slot_r - count_px * 0.45f,
             wh.amount[i] ? rgba(250, 250, 240) : rgba(150, 150, 150), o);
    }
    if (focus >= 0 && !focus_icon)   // (an item with no icon: its name in the middle too)
        label(c, lbl(focus), big, mx, my, 2 * centre_r - 2 * u, focus == wh.cursor ? gold : white, o);
    const float banner_y = my + ring_r + slot_r + (kBannerGap + kBannerHalf) * u;
    // the pages: a dot each under the banner, this one gold, with an arrow either side
    if (wh.pages > 1) {
        const float py = banner_y + (kBannerHalf + 1.8f) * u, dot = std::max(3.0f, 0.55f * u), gap = 2.2f * u;
        const float x0 = mx - gap * (wh.pages - 1) * 0.5f;
        for (int p = 0; p < wh.pages; p++) {   // (the player's pages filled, the others' a ring, a new one a ring with a cross)
            const float dx = x0 + gap * p, r = p == wh.page ? dot * 1.3f : dot;
            const Rgba& col = p == wh.page ? gold : rim;
            const int kind = p < (int)sizeof wh.page_kind ? wh.page_kind[p] : 0;
            const float w1 = std::max(1.5f, u / 4);
            if (kind == 0) disc(c, dx, py, r, col);
            else ring(c, dx, py, r, w1, col);
            if (kind == 2) {
                segment(c, dx - r * 0.6f, py, dx + r * 0.6f, py, w1, col);
                segment(c, dx, py - r * 0.6f, dx, py + r * 0.6f, w1, col);
            }
        }
        const float ax = gap * (wh.pages - 1) * 0.5f + 2.2f * u, ah = 0.9f * u;
        triangle(c, mx - ax - ah, py, mx - ax + ah * 0.6f, py - ah, mx - ax + ah * 0.6f, py + ah, white);
        triangle(c, mx + ax + ah, py, mx + ax - ah * 0.6f, py - ah, mx + ax - ah * 0.6f, py + ah, white);
    }
    // the name banner under the ring: the focused item's name on a dark plate with round ends
    const char* nm = moving ? name_of(wh.held_name, wh.held_name) : focus >= 0 ? lbl(focus) : "";
    if (*nm) {
        const float hh = kBannerHalf * u, tw = (float)text_width(nm, name_px);
        const float half_w = std::min(std::min(pic_w / 2 - 1, 47 * u), std::max(16 * u, tw / 2 + 4 * u));   // (within wheel_picture's area)
        const Rgba& edge_col = moving ? move_col : focus == wh.cursor ? gold : rim;
        stadium(c, mx - half_w, mx + half_w, banner_y, hh, edge_col);
        stadium(c, mx - half_w + edge, mx + half_w - edge, banner_y, std::max(0.0f, hh - edge), dark);
        text(c, nm, name_px, mx, banner_y, !moving && focus == wh.cursor ? gold : white, o);
    }
    // the prompts over the wheel: the button bound to the Switch's Y (wherever the layout puts it, its place among the face
    // buttons lit) takes an item up to move it; while one is held, it puts it down and the one bound to B puts it back
    if (n > 0) {
        const int prompt_px = (int)std::max(12.0f, std::round(1.9f * u));
        const float py = my - (ring_r + slot_r) * spread - 4.2f * u;
        auto bound = [&](int sw, int fallback) {
            const int at = host && host->binding ? host->binding(HOS_BIND_PAD, sw, 0) : fallback;
            return at >= HOS_PAD_SOUTH && at <= HOS_PAD_NORTH ? at : fallback;
        };
        const struct { const char* w; int pad; } list[2] = {{word(moving ? 1 : 0), bound(HOS_SW_Y, HOS_PAD_WEST)},
                                                            {word(2), bound(HOS_SW_B, HOS_PAD_EAST)}};
        const int np = moving ? 2 : 1;
        const float icon = 2.6f * u, gap = 0.8f * u, between = 2.4f * u;
        float total = 0;
        for (int k = 0; k < np; k++) total += text_width(list[k].w, prompt_px) + gap + icon + (k ? between : 0);
        float x = mx - total / 2;
        for (int k = 0; k < np; k++) {
            if (k) x += between;
            const float tw = (float)text_width(list[k].w, prompt_px);
            text(c, list[k].w, prompt_px, x + tw / 2, py, white, o);
            x += tw + gap;
            // (the face buttons as four dots, the bound one lit: south, east, west, north)
            const float cx = x + icon / 2, d = icon * 0.27f, r = icon * 0.17f;
            disc(c, cx, py, icon / 2, rgba(232, 232, 224));
            const float off[4][2] = {{0, d}, {d, 0}, {-d, 0}, {0, -d}};
            for (int b = 0; b < 4; b++)
                if (b == list[k].pad) disc(c, cx + off[b][0], py + off[b][1], r, rgba(40, 40, 40));
                else ring(c, cx + off[b][0], py + off[b][1], r * 0.8f, std::max(1.0f, u / 6), rgba(90, 90, 90));
            x += icon;
        }
    }
}

Canvas canvas;
std::vector<uint8_t> out8;
uint64_t version = 0;

}  // namespace

bool wheel_picture(uint32_t pw, uint32_t ph, WheelPicture* out) {
    bool changed = false;
    if (!pw || !ph || !wheel_update(&changed)) return false;
    // (the wheel's size from the picture's height, a hundredth of it a unit, as the host's slot_under takes it)
    const float u = std::max(1.0f, std::floor(ph / 100.0f));
    const float mx = std::floor(pw / 2.0f), my = std::floor(ph / 2.0f);
    // (the area it covers: the slots and the banner across, the prompts above, the pages' dots below)
    const int half = (int)std::min((float)pw / 2, 48 * u);   // (the ring and its slots 35 units; a long name's banner within 48)
    const int top = std::max(0, (int)(my - 46 * u)), bottom = std::min((int)ph, (int)(my + 52 * u));
    const int left = std::max(0, (int)mx - half), right = std::min((int)pw, (int)mx + half);
    const int w = right - left, h = bottom - top;
    if (w <= 0 || h <= 0) return false;
    static uint32_t last_w = 0, last_h = 0;
    if (changed || last_w != pw || last_h != ph || out8.empty()) {
        last_w = pw, last_h = ph;
        canvas.reset(w, h);
        draw(canvas, mx - left, my - top, u, (float)pw);
        out8.resize((size_t)w * h * 4);
        for (size_t i = 0; i < out8.size(); i++) out8[i] = (uint8_t)std::lround(std::clamp(canvas.px[i], 0.0f, 1.0f) * 255);
        version++;
    }
    out->px = out8.data(), out->w = (uint32_t)w, out->h = (uint32_t)h, out->x = left, out->y = top, out->version = version;
    return true;
}

}  // namespace mkwvk
