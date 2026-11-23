"""A start walks the pipeline notes (hookshot_pipelines.bin) once for their references.

notes_lost_by_driver, precompile_warm and precompile_at_create each walked the whole file for the notes' references
(pipe_notes_refs, gpu/vk/vk_draw.cpp): two walks on a normal start that loads at create calls, three when the driver had
lost the notes, about 0.25 s each with the file (1.0 GB, 37,474 notes on one PC) in the OS's cache. Each walk read every note
with three seeks and reads. Now the start's uses go through start_note_refs, which walks once and keeps the references
while the file's size and time stay as they were.

Cut from the source and run on files the test makes (no game data):
- pipe_notes_refs gives the same references, entry for entry, as main 0de25c3e's walk (a copy kept below, base_notes_refs),
  which every answer of start_note_refs is compared with too: whole
  notes, duplicate keys, notes whose targets or formats are refused, a tail cut inside a note's front, inside its middle and
  one byte short, a file of whole notes only, an empty file, a cut header, no file, a wrong stamp and a wrong record size;
- start_note_refs walks once over the three start shapes (a warm start; a start whose driver lost the notes, the warm pass
  between; a --prepare-shaders run), walks again for a file that grew (also with its time put back), was rewritten at the
  same size, or appeared, and after
  the last use took them;
- a walk that failed is not kept, and a use that cannot read the file walks again: the file held by another program with
  reads denied while its size and time still read, during the walk or from after it (Windows: _sopen_s _SH_DENYRW and
  _SH_DENYRD; everywhere: the open failing), a read failing part way (Windows: another handle's LockFileEx on a note;
  elsewhere: the stream's descriptor turned into a folder's, EISDIR), a seek failing part way, and the file's size and time,
  or its time alone, not readable; a file honestly without notes is kept, with main's same empty answer;
- broken variants, each of which must fail its own case's check (named in BROKEN; a crash, or another case failing, is not
  a catch): the kept references used without the file's size and time compared, without its size (a file grown with its
  time put back, as on FAT or exFAT), and without its time (rewritten at the same size); a failed walk kept; a failed read,
  or a failed seek, not seen by the walk; a walk kept while the file's size and time could not be read, or its time alone;
  a file that no longer reads not seen.
And from the source: the three uses call start_note_refs, only precompile_at_create takes, and nothing else walks.

python -I -B tests/test_note_refs_once.py [--cxx g++]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (tests/srccut.py: python -I leaves this folder off the path)
from srccut import block, between   # noqa: E402

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''#define _CRT_SECURE_NO_WARNINGS 1
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>
#ifndef _WIN32
#define _fseeki64 fseeko
#define _ftelli64 ftello
#include <fcntl.h>
#include <unistd.h>
#else
#define NOMINMAX 1
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <share.h>
#endif
#define VK_PRIMITIVE_TOPOLOGY_PATCH_LIST 10u
static int fails;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); fails++; } } while (0)

// (a note shaped as the real one: its first words, then a state of several KB with the vertex formats inside it, so that the
// front, the formats and the note's end are on different 4 KB pages)
struct NvnState {
    uint8_t before[5000];
    uint32_t vattr[16];
    uint8_t after[3000];
};
struct PipeNote {
    uint64_t key, prog, variant;
    uint32_t topo, n, samples, depth_format, has_depth;
    uint32_t color_format[8];
    NvnState s;
};
/* NOTEREF */;
std::vector<NoteRef> pipe_notes_refs(bool* whole = nullptr);   // (vk_pipelines.h)
static std::string dir;
std::string pipe_notes_path() { return dir + "hookshot_pipelines.bin"; }
// (stand-ins, put in place of the cut source's own calls: the opens of the notes, the walk's seeks, the file's size and time)
static bool open_fails, size_fails, time_fails;
static int64_t seek_fails_at = -1, read_fails_from = -1;
FILE* walk_fopen(const char* path, const char* mode) {
    if (open_fails) return nullptr;   // (another program holds the file with reads denied)
    FILE* f = fopen(path, mode);
#ifndef _WIN32
    // (a stream that is to fail part way keeps a 4 KB buffer, whatever the C library or the file system would choose: a buffer
    // holding the whole file from the head's read would serve every note before the descriptor is swapped; a buffer of the
    // fixture's own, as glibc takes the size only with one. The streams opened here are closed before the next one opens)
    static char buffer[4096];
    if (f && read_fails_from >= 0) setvbuf(f, buffer, _IOFBF, sizeof buffer);
#endif
    return f;
}
int walk_fseek(FILE* f, int64_t at, int whence) {
    if (at == seek_fails_at) {   // (one seek fails, its stream's flags left as they are)
        seek_fails_at = -1;
        errno = EINVAL;
        return -1;
    }
#ifndef _WIN32
    if (at == read_fails_from) {   // (from here on the stream's reads fail: its descriptor becomes a folder's, EISDIR)
        read_fails_from = -1;
        const int folder = open(dir.c_str(), O_RDONLY);
        if (folder >= 0) dup2(folder, fileno(f)), close(folder);
    }
#endif
    return _fseeki64(f, at, whence);
}
uintmax_t id_file_size(const std::string& path, std::error_code& ec) {
    if (size_fails) { ec = std::make_error_code(std::errc::permission_denied); return (uintmax_t)-1; }
    return std::filesystem::file_size(path, ec);
}
std::filesystem::file_time_type id_last_write_time(const std::string& path, std::error_code& ec) {
    if (time_fails) { ec = std::make_error_code(std::errc::permission_denied); return std::filesystem::file_time_type::min(); }
    return std::filesystem::last_write_time(path, ec);
}
constexpr uint64_t kStamp = 0x5354414d50ull;
uint64_t program_cache_stamp() { return kStamp; }
/* TARGETS_OK */
bool formats_ok(uint32_t n, const uint32_t color[8], uint32_t has_depth, uint32_t depth) {
    for (uint32_t i = 0; i < n && i < 8; i++) if (color[i] == 0xBAD) return false;
    return !has_depth || depth != 0xBAD;
}
void note_color_types(const PipeNote& n, uint8_t types[8]) { for (int i = 0; i < 8; i++) types[i] = (uint8_t)(n.color_format[i] * 3 + 1); }

/* WALK */

// (the walk before this change, as main 0de25c3e had it: the reference every answer is compared with)
std::vector<NoteRef> base_notes_refs() {
    std::vector<NoteRef> refs;
    FILE* f = fopen(pipe_notes_path().c_str(), "rb");
    if (!f) return refs;
    uint64_t head[2] = {0, 0};
    if (fread(head, 8, 2, f) == 2 && head[0] == program_cache_stamp() && head[1] == sizeof(PipeNote)) {
        constexpr size_t kFront = offsetof(PipeNote, s);
        constexpr size_t kVattr = offsetof(PipeNote, s) + offsetof(NvnState, vattr);
        std::unordered_set<uint64_t> seen;
        uint8_t front[kFront];
        PipeNote probe{};
        for (int64_t at = 16;; at += (int64_t)sizeof(PipeNote)) {
            NoteRef r;
            if (_fseeki64(f, at, SEEK_SET) || fread(front, 1, kFront, f) != kFront) break;
            if (_fseeki64(f, at + (int64_t)kVattr, SEEK_SET) || fread(r.vattr, 4, 16, f) != 16) break;
            if (_fseeki64(f, at + (int64_t)sizeof(PipeNote) - 1, SEEK_SET) || fgetc(f) == EOF) break;
            std::memcpy((void*)&probe, front, kFront);
            if (!targets_ok(probe.n, probe.samples, probe.topo) ||
                !formats_ok(probe.n, probe.color_format, probe.has_depth, probe.depth_format))
                continue;
            r.key = probe.key, r.prog = probe.prog, r.variant = probe.variant, r.at = at;
            note_color_types(probe, r.color_types);
            if (seen.insert(r.key).second) refs.push_back(r);
        }
    }
    fclose(f);
    return refs;
}

/* START */

static bool same(const std::vector<NoteRef>& a, const std::vector<NoteRef>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].key != b[i].key || a[i].prog != b[i].prog || a[i].variant != b[i].variant || a[i].at != b[i].at ||
            std::memcmp(a[i].vattr, b[i].vattr, sizeof a[i].vattr) || std::memcmp(a[i].color_types, b[i].color_types, 8))
            return false;
    }
    return true;
}
static PipeNote make(uint64_t key, uint32_t n = 2, uint32_t color0 = 37) {
    static PipeNote note;
    std::memset((void*)&note, 0xA5, sizeof note);   // (every byte something: a read from the wrong place shows)
    note.key = key, note.prog = key * 7 + 1, note.variant = key ^ 0x55, note.topo = 3, note.n = n, note.samples = 1;
    note.depth_format = 126, note.has_depth = (uint32_t)(key & 1);
    for (int i = 0; i < 8; i++) note.color_format[i] = i ? 0 : color0;
    for (int i = 0; i < 16; i++) note.s.vattr[i] = (uint32_t)(key * 100 + i);
    return note;
}
static void write(const std::vector<PipeNote>& notes, uint64_t stamp = kStamp, uint64_t rec = sizeof(PipeNote), long cut = 0) {
    std::string bytes;
    const uint64_t head[2] = {stamp, rec};
    bytes.append((const char*)head, sizeof head);
    for (const PipeNote& n : notes) bytes.append((const char*)&n, sizeof n);
    bytes.resize(bytes.size() - (size_t)cut);
    FILE* f = fopen(pipe_notes_path().c_str(), "wb");
    fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
}
static void append(const PipeNote& n) {
    FILE* f = fopen(pipe_notes_path().c_str(), "ab");
    fwrite(&n, sizeof n, 1, f);
    fclose(f);
}
static std::vector<PipeNote> some(size_t count, uint64_t first = 1) {
    std::vector<PipeNote> v;
    for (size_t i = 0; i < count; i++) v.push_back(make(first + i));
    return v;
}

int main(int argc, char** argv) {
    dir = std::string(argv[1]) + "/";   // (".": the test runs the fixture in its folder)
    const std::string part = argc > 2 ? argv[2] : "all";
    const long rec = (long)sizeof(PipeNote), front = (long)offsetof(PipeNote, s);
    if (part == "all" || part == "walk") {
        int cases = 0;
        const auto agree = [&](const char* what, size_t want_n) {
            const std::vector<NoteRef> now = pipe_notes_refs(), was = base_notes_refs();
            cases++;
            if (!same(now, was) || now.size() != want_n) {
                std::printf("FAIL %s: %zu references, the old walk %zu, %zu expected\n", what, now.size(), was.size(), want_n);
                fails++;
            }
        };
        // (whole notes; a duplicate key, the first kept; a note refused by its targets and one by its formats)
        std::vector<PipeNote> notes = some(6);
        notes.push_back(make(3));
        notes.push_back(make(40, 9));
        notes.push_back(make(41, 2, 0xBAD));
        notes.push_back(make(42));
        write(notes);
        agree("whole notes, a duplicate key, refused targets and formats", 7);
        // (a tail cut inside the next note's front, inside its middle, one byte short; and a whole note past a cut one: the
        // walk stops at the cut, as before)
        for (const long keep : {10L, front + 4, (long)(offsetof(PipeNote, s) + offsetof(NvnState, vattr)) + 8, rec / 2, rec - 1}) {
            std::vector<PipeNote> v = some(4);
            v.push_back(make(9));
            write(v, kStamp, sizeof(PipeNote), rec - keep);
            agree("a tail cut inside a note", 4);
        }
        write(some(5));
        agree("whole notes only, the last ending at the file's end", 5);
        write({});
        agree("the head alone", 0);
        write({}, kStamp, sizeof(PipeNote), 16);
        agree("an empty file", 0);
        write({}, kStamp, sizeof(PipeNote), 5);
        agree("a cut head", 0);
        write(some(3), kStamp + 1);
        agree("a wrong stamp", 0);
        write(some(3), kStamp, sizeof(PipeNote) + 8);
        agree("a wrong record size", 0);
        std::filesystem::remove(pipe_notes_path());
        agree("no file", 0);
        CHECK(cases == 13);
    }
    if (part == "all" || part == "once") {
        StartNoteRefs& k = start_note_refs_kept;
        // (a warm start: the driver's check, then the load at create, which takes them)
        k = StartNoteRefs{};
        write(some(5));
        const std::vector<NoteRef> check = start_note_refs(false);
        const std::vector<NoteRef> create = start_note_refs(true);
        CHECK(k.walks == 1);
        CHECK(same(check, base_notes_refs()) && same(create, check));
        // (the driver lost the notes: the check, the warm pass, then the load at create)
        k = StartNoteRefs{};
        const std::vector<NoteRef> a = start_note_refs(false), b = start_note_refs(false), c = start_note_refs(true);
        CHECK(k.walks == 1);
        CHECK(same(a, b) && same(b, c) && c.size() == 5);
        // (a --prepare-shaders run: the warm pass alone, or the load at create alone)
        k = StartNoteRefs{};
        CHECK(start_note_refs(false).size() == 5 && k.walks == 1);
        k = StartNoteRefs{};
        CHECK(start_note_refs(true).size() == 5 && k.walks == 1);
        // (a file that grew between two uses: walked again, the new note in it)
        k = StartNoteRefs{};
        CHECK(start_note_refs(false).size() == 5);
        append(make(77));
        const std::vector<NoteRef> grown = start_note_refs(false);
        CHECK(k.walks == 2);
        CHECK(grown.size() == 6 && same(grown, base_notes_refs()));
        // (rewritten at the same size, its time moved on: walked again)
        std::vector<PipeNote> v = some(6);
        v[2].key = 1000;
        write(v);
        std::filesystem::last_write_time(pipe_notes_path(), std::filesystem::last_write_time(pipe_notes_path()) +
                                                            std::chrono::seconds(5));
        const std::vector<NoteRef> rewritten = start_note_refs(false);
        CHECK(k.walks == 3);
        CHECK(same(rewritten, base_notes_refs()) && rewritten[2].key == 1000);
        // (taken by the last use: the next use walks again)
        CHECK(same(start_note_refs(true), rewritten) && k.walks == 3);
        CHECK(same(start_note_refs(false), rewritten) && k.walks == 4);
        // (grown with its time put back, as a file system whose times are coarse (FAT 2 s, exFAT 10 ms) can show an append:
        // walked again by its size alone)
        k = StartNoteRefs{};
        write(some(3));
        const auto when = std::filesystem::last_write_time(pipe_notes_path());
        CHECK(start_note_refs(false).size() == 3 && k.walks == 1);
        append(make(88));
        std::filesystem::last_write_time(pipe_notes_path(), when);
        const std::vector<NoteRef> coarse = start_note_refs(false);
        CHECK(k.walks == 2 && coarse.size() == 4 && same(coarse, base_notes_refs()));
        // (no file, then one: nothing kept from no file, so each use walks; then the new file's notes)
        k = StartNoteRefs{};
        std::filesystem::remove(pipe_notes_path());
        CHECK(start_note_refs(false).empty() && start_note_refs(false).empty() && k.walks == 2);
        write(some(2));
        CHECK(start_note_refs(false).size() == 2 && k.walks == 3);
        // (a file honestly without notes, its head alone: read whole, kept, the same empty answer as main's walks)
        k = StartNoteRefs{};
        write({});
        CHECK(start_note_refs(false).empty() && start_note_refs(true).empty() && base_notes_refs().empty() && k.walks == 1);
    }
    if (part == "all" || part == "fail") {
        StartNoteRefs& k = start_note_refs_kept;
        // (the driver's check walks while another program holds the file with reads denied, its size and time still read;
        // the program lets go before the load at create: that walk is not kept, the load at create reads every note)
        k = StartNoteRefs{};
        write(some(5));
        open_fails = true;
        const std::vector<NoteRef> held = start_note_refs(false);
        open_fails = false;
        const std::vector<NoteRef> freed = start_note_refs(true);
        CHECK(held.empty() && k.walks == 2);
        CHECK(freed.size() == 5 && same(freed, base_notes_refs()));
        // (a whole walk, then another program takes the file with reads denied before the load at create: that use walks
        // again and reads nothing, as main's second walk did; the file's head would not read for notes_open_locked either)
        k = StartNoteRefs{};
        CHECK(start_note_refs(false).size() == 5 && k.walks == 1);
        open_fails = true;
        const std::vector<NoteRef> relocked = start_note_refs(true);
        open_fails = false;
        CHECK(relocked.empty() && k.walks == 2);
        // (a seek failing at the fourth note: the walk stops there with three, not kept; the next use reads all five)
        k = StartNoteRefs{};
        seek_fails_at = 16 + 3 * (int64_t)sizeof(PipeNote);
        const std::vector<NoteRef> seeked = start_note_refs(false);
        seek_fails_at = -1;
        const std::vector<NoteRef> reseeked = start_note_refs(true);
        CHECK(seeked.size() == 3 && seeked.size() < 5 && k.walks == 2);
        CHECK(reseeked.size() == 5 && same(reseeked, base_notes_refs()));
#ifndef _WIN32
        {   // (reads failing from the fourth note on: the walk stops there, not kept; Windows has its own case below)
            k = StartNoteRefs{};
            read_fails_from = 16 + 3 * (int64_t)sizeof(PipeNote);
            const std::vector<NoteRef> cut = start_note_refs(false);
            read_fails_from = -1;
            const std::vector<NoteRef> after = start_note_refs(true);
            CHECK(!cut.empty() && cut.size() < 5 && k.walks == 2);
            CHECK(after.size() == 5 && same(after, base_notes_refs()));
        }
#endif
        // (the file's size and time cannot be read, the walk can: not kept, so a note appended is seen)
        k = StartNoteRefs{};
        size_fails = time_fails = true;
        CHECK(start_note_refs(false).size() == 5 && k.walks == 1);
        append(make(77));
        const std::vector<NoteRef> unknown = start_note_refs(false);
        size_fails = time_fails = false;
        CHECK(k.walks == 2 && unknown.size() == 6 && same(unknown, base_notes_refs()));
        // (its time alone cannot be read: not kept either, so a rewrite at the same size is seen)
        k = StartNoteRefs{};
        write(some(5));
        time_fails = true;
        CHECK(start_note_refs(false).size() == 5 && k.walks == 1);
        std::vector<PipeNote> v = some(5);
        v[1].key = 2000;
        write(v);
        const std::vector<NoteRef> no_time = start_note_refs(false);
        time_fails = false;
        CHECK(k.walks == 2 && same(no_time, base_notes_refs()) && no_time.size() == 5 && no_time[1].key == 2000);
#ifdef _WIN32
        {   // (the real thing on Windows: another handle opened with reads denied; the size and time still read)
            k = StartNoteRefs{};
            write(some(5));
            int fd = -1;
            CHECK(_sopen_s(&fd, pipe_notes_path().c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRW, 0) == 0);
            const std::vector<NoteRef> denied = start_note_refs(false);
            if (fd >= 0) _close(fd);
            const std::vector<NoteRef> after = start_note_refs(true);
            CHECK(denied.empty() && k.walks == 2);
            CHECK(after.size() == 5 && same(after, base_notes_refs()));
        }
        {   // (the real thing after a whole walk: another handle opened with reads denied, _SH_DENYRD, before the take)
            k = StartNoteRefs{};
            write(some(5));
            CHECK(start_note_refs(false).size() == 5 && k.walks == 1);
            int fd = -1;
            CHECK(_sopen_s(&fd, pipe_notes_path().c_str(), _O_RDONLY | _O_BINARY, _SH_DENYRD, 0) == 0);
            const std::vector<NoteRef> relocked_win = start_note_refs(true);
            if (fd >= 0) _close(fd);
            CHECK(relocked_win.empty() && k.walks == 2);
        }
        {   // (a read failing part way: another handle locks the fourth note's bytes, the walk stops there)
            k = StartNoteRefs{};
            write(some(5));
            HANDLE h = CreateFileA(pipe_notes_path().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                   OPEN_EXISTING, 0, nullptr);
            CHECK(h != INVALID_HANDLE_VALUE);
            OVERLAPPED o{};
            const uint64_t from = 16 + 3 * (uint64_t)sizeof(PipeNote);
            o.Offset = (DWORD)from, o.OffsetHigh = (DWORD)(from >> 32);
            CHECK(LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, (DWORD)sizeof(PipeNote), 0, &o));
            const std::vector<NoteRef> cut = start_note_refs(false);
            UnlockFileEx(h, 0, (DWORD)sizeof(PipeNote), 0, &o);
            CloseHandle(h);
            const std::vector<NoteRef> after = start_note_refs(true);
            CHECK(!cut.empty() && cut.size() < 5 && k.walks == 2);
            CHECK(after.size() == 5 && same(after, base_notes_refs()));
        }
#endif
    }
    if (fails) return 1;
    std::printf("ok\n");
    return 0;
}
'''

KEPT_TEST = "if (!k.kept || k.size != size || k.time != time) {"
KEEP = "k.kept = whole && size >= 0 && after == size && after_time == time;"
SIZE_OF_ID = "*size = ec || ec_time || !reads ? -1 : (int64_t)n;"
WHOLE = "if (whole) *whole = feof(f) && !ferror(f);"
# (the cut source's calls the driver's stand-ins take the place of)
WALK_OPEN = 'fopen(pipe_notes_path().c_str(), "rb")'
ID_OPEN = 'fopen(path.c_str(), "rb")'
WALK_SEEK = "_fseeki64("
ID_CALLS = ("std::filesystem::file_size(path, ec)", "std::filesystem::last_write_time(path, ec_time)")
# (each broken variant: the part it runs, the replacement, and the text of the check of its own case that must fail)
BROKEN = (("kept-unchecked", "once", KEPT_TEST, "if (!k.kept) {", "grown.size() == 6"),
          ("size-unchecked", "once", KEPT_TEST, "if (!k.kept || k.time != time) {", "coarse.size() == 4"),
          ("time-unchecked", "once", KEPT_TEST, "if (!k.kept || k.size != size) {", "rewritten[2].key == 1000"),
          ("failed-walk-kept", "fail", KEEP, "k.kept = size >= 0 && after == size && after_time == time;",
           "seeked.size() == 3"),
          ("read-failure-unseen", "fail", WHOLE, "if (whole) *whole = true;", "!cut.empty() && cut.size() < 5"),
          ("seek-failure-unseen", "fail", WHOLE, "if (whole) *whole = !ferror(f);", "seeked.size() == 3"),
          ("unread-id-kept", "fail", KEEP, "k.kept = whole && after == size && after_time == time;", "unknown.size() == 6"),
          ("unread-time-kept", "fail", SIZE_OF_ID, "*size = ec || !reads ? -1 : (int64_t)n;", "no_time[1].key == 2000"),
          ("unreadable-unseen", "fail", SIZE_OF_ID, "*size = ec || ec_time ? -1 : (int64_t)n;", "relocked.empty()"))


def fixture(draw, pipelines):
    walk = block(draw, "std::vector<NoteRef> pipe_notes_refs(bool* whole) {", where="gpu/vk/vk_draw.cpp")
    start = between(draw, "struct StartNoteRefs {", "/// (hookshot, RX_VK_LOAD=create) The first start after a preparation",
                    where="gpu/vk/vk_draw.cpp")
    for anchor, text, n in ((KEPT_TEST, start, 1), (KEEP, start, 1), (SIZE_OF_ID, start, 1), (WHOLE, walk, 1),
                            (WALK_OPEN, walk, 1), (WALK_SEEK, walk, 3), (ID_OPEN, start, 1), (ID_CALLS[0], start, 1),
                            (ID_CALLS[1], start, 1)):
        if text.count(anchor) != n:
            raise SystemExit("FAIL: gpu/vk/vk_draw.cpp does not have %r %d times (the check and its broken variants cut it)"
                             % (anchor, n))
    walk = walk.replace(WALK_OPEN, "walk_" + WALK_OPEN).replace(WALK_SEEK, "walk_fseek(")
    start = (start.replace(ID_OPEN, "walk_" + ID_OPEN).replace(ID_CALLS[0], "id_file_size(path, ec)")
             .replace(ID_CALLS[1], "id_last_write_time(path, ec_time)"))
    return (DRIVER.replace("/* NOTEREF */", block(pipelines, "struct NoteRef {", where="gpu/vk/vk_pipelines.h"))
            .replace("/* TARGETS_OK */", block(draw, "bool targets_ok(uint32_t n, uint32_t samples, uint32_t topo) {",
                                               where="gpu/vk/vk_draw.cpp"))
            .replace("/* WALK */", walk).replace("/* START */", start))


def run(cxx, code, tmp, name, part="all"):
    """The fixture's exit code and output; the build's own failure stops the check"""
    cpp, exe = Path(tmp) / (name + ".cpp"), Path(tmp) / (name + ".exe")
    cpp.write_text(code, encoding="utf-8")
    r = subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", "-Wno-unused-variable",
                        str(cpp), "-o", str(exe)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("FAIL: the %s fixture does not build:\n%s%s" % (name, r.stdout, r.stderr))
    files = Path(tmp) / name
    files.mkdir()
    r = subprocess.run([str(exe), ".", part], cwd=str(files), capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def source_checks(draw):
    """The three uses of a start take the references from start_note_refs, only the last takes them, and nothing else walks"""
    uses = {"bool notes_lost_by_driver() {": "start_note_refs(false)",
            "void precompile_warm(void (*progress)(size_t done, size_t total)) {": "start_note_refs(false)",
            "void precompile_at_create(void (*progress)(size_t done, size_t total)) {": "start_note_refs(true)"}
    for marker, call in uses.items():
        body = block(draw, marker, where="gpu/vk/vk_draw.cpp")
        if body.count("start_note_refs(") != 1 or call not in body or "pipe_notes_refs(" in body:
            raise SystemExit("FAIL: %s does not take the notes' references once through %s" % (marker.split("(")[0], call))
    # (start_note_refs' walk and the definition; and Hookshot's precompile_index_only, the one walk of a start that skips
    # the startup compilation: it runs in the precompile's place, never beside it)
    index_only = block(draw, "void precompile_index_only() {", where="gpu/vk/vk_draw.cpp") \
        if "void precompile_index_only() {" in draw else ""
    if draw.count("pipe_notes_refs(") - index_only.count("pipe_notes_refs(") != 2 or index_only.count("pipe_notes_refs(") > 1:
        raise SystemExit("FAIL: pipe_notes_refs() is called somewhere other than start_note_refs: a second walk a start")
    # (a start that leaves the load-at-create pass without taking them, the warm pass's marker unwritten, lets them go before the
    # precompile and play: kept, they were held for the session)
    run = block(draw, "void precompile_run(void (*progress)(size_t done, size_t total)) {", where="gpu/vk/vk_draw.cpp")
    took, drop, leave = "return start_steps::end();", "start_note_refs_kept = StartNoteRefs{};", "if (prepare_stopped() ||"
    for anchor in (took, drop, leave):
        if run.count(anchor) != 1:
            raise SystemExit("FAIL: precompile_run does not have %r once (the check of the kept references' release cuts it)"
                             % anchor)
    if not run.index(took) < run.index(drop) < run.index(leave):
        raise SystemExit("FAIL: precompile_run does not let the kept references go where it leaves the load at create untaken")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cxx", default="g++")
    a = p.parse_args()
    draw = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
    pipelines = (ROOT / "gpu/vk/vk_pipelines.h").read_text(encoding="utf-8")
    source_checks(draw)
    code = fixture(draw, pipelines)
    with tempfile.TemporaryDirectory(prefix="note-refs-once-") as tmp:
        rc, out = run(a.cxx, code, tmp, "refs")
        if rc:
            raise SystemExit("FAIL: the notes' references:\n" + out)
        # (broken variants: each must fail)
        for name, part, anchor, broken, own in BROKEN:
            variant = code.replace(anchor, broken)
            if variant == code:
                raise SystemExit("FAIL: the broken variant %s changed nothing" % name)
            rc, out = run(a.cxx, variant, tmp, name, part)
            caught = [line for line in out.splitlines() if line.startswith("FAIL line") and own in line]
            if rc != 1 or not caught:   # (its own case's check failing: not a pass, a crash, or another case)
                raise SystemExit("FAIL: the broken variant %s gave exit %d without failing its own check (%r): the check "
                                 "does not see it\n%s" % (name, rc, own, out))
    print("PASS: one walk of the notes a start, a changed file walked again")


if __name__ == "__main__":
    main()
