/* (hookshot) What the Discord status says (discord_activity.h).

   Discord shows the application's name in bold ("Playing Hookshot": the Discord application of discord_app_id=, its
   name and its art; host/discord_rpc.c), and under it the status's two lines. Those are the player's to write in
   hookshot_options.ini (the menu's saves keep these lines):
     discord_text=        the two lines, a template (discord_activity.h); default:
                            {location|{location} - {age}{break}{hearts|❤️ {hearts}/{max_hearts} · }
                            {skulltulas|🕷️ {skulltulas} · }{heart_pieces|💗 {heart_pieces}}}
                            {!location|{version} {build}{!version| {commit}}}   (one line)
                          so in a map "Kakariko Village - Adult" over "❤️ 2.75/3 · 🕷️ 12 · 💗 2", the version on the picture's
                          hover text; with no location (the title screen) the version, "v3.0.0 RC7": the bold name above
                          already says Hookshot. The commit only on a build with no version number (a nightly, "N486
                          09a20cd6"; a checkout with no tag), where it is what tells two builds apart
     discord_image=       the large picture: an art asset's key of the Discord application, or an https:// link to a
                          picture (default: hookshot, the application's art)
     discord_image_text=  the picture's hover text, a template (default: Hookshot {version} {build}{!version| {commit}})
   An empty part takes with it the brackets around it, its extra space and a dash it leaves at a line's end ("Hookshot
   {version} {build} [{commit}]" on a release: "Hookshot v3.0.0"). Discord takes 2 to 128 letters a line; a shorter line
   is left out, a longer one cut. (Not discord_status=: that is the Insert menu's own row, Discord status on or off.) */
#include "discord_activity.h"
#include <ctype.h>
#include <stdio.h>
#include <string.h>

#define DEFAULT_STATUS "{location|{location} - {age}{break}{hearts|\xe2\x9d\xa4\xef\xb8\x8f {hearts}/{max_hearts} \xc2\xb7 }{skulltulas|\xf0\x9f\x95\xb7\xef\xb8\x8f {skulltulas} \xc2\xb7 }{heart_pieces|\xf0\x9f\x92\x97 {heart_pieces}}}{!location|{version} {build}{!version| {commit}}}"
#define BREAK '\x1e'   /* ({break} in an expanded template: no text has it, the options file's lines being text) */

void discord_settings_default(DiscordSettings *s) {
    memset(s, 0, sizeof *s);
    snprintf(s->status, sizeof s->status, "%s", DEFAULT_STATUS);
    snprintf(s->image, sizeof s->image, "hookshot");
    snprintf(s->image_text, sizeof s->image_text, "Hookshot {version} {build}{!version| {commit}}");
}

int discord_settings_line(DiscordSettings *s, const char *line) {
    static const struct { const char *key; size_t at; size_t size; } keys[] = {
        {"discord_text=", offsetof(DiscordSettings, status), sizeof s->status},
        {"discord_image=", offsetof(DiscordSettings, image), sizeof s->image},
        {"discord_image_text=", offsetof(DiscordSettings, image_text), sizeof s->image_text},
    };
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        const size_t n = strlen(keys[i].key);
        if (strncmp(line, keys[i].key, n)) continue;
        char *dst = (char *)s + keys[i].at;
        snprintf(dst, keys[i].size, "%s", line + n);
        dst[strcspn(dst, "\r\n")] = 0;
        for (char *p = dst; *p; p++)
            if (*p == BREAK) *p = ' ';
        return 1;
    }
    return 0;
}

/* ---- the template ---- */

/* the field called name (n bytes): its value ("" when unknown), NULL when no field has that name; break is BREAK's */
static const char *field(const DiscordFields *f, const char *name, size_t n) {
    static const char brk[2] = {BREAK, 0};
    static const char *const names[] = {"location", "map", "age", "version", "build", "commit", "release", "break",
                                        "hearts", "max_hearts", "rupees", "skulltulas", "heart_pieces"};
    for (size_t k = 0; k < sizeof names / sizeof *names; k++) {
        if (strlen(names[k]) != n || strncmp(names[k], name, n)) continue;
        const char *const values[] = {f->location, f->map, f->age, f->version, f->build, f->commit, f->release, brk,
                                      f->hearts, f->max_hearts, f->rupees, f->skulltulas, f->heart_pieces};
        return values[k] ? values[k] : "";
    }
    return NULL;
}

/* where the "}" that closes a {name|...} whose text starts at t[i] is, or n. Read by its parts: {{ a brace, {name} a
   field (its "}" is its own: "{age}}" is the field, then the close), {name|...} a nested one; any other "}" closes (a
   brace inside one is "{{" only; "}}" is a brace outside them) */
static size_t closing(const char *t, size_t i, size_t n, int depth) {
    while (i < n) {
        if (t[i] == '{') {
            if (i + 1 < n && t[i + 1] == '{') {
                i += 2;
                continue;
            }
            const size_t at = i + 1 + (i + 1 < n && t[i + 1] == '!');   /* ({!name|...}: its name after the !) */
            size_t e = at;
            while (e < n && (isalnum((unsigned char)t[e]) || t[e] == '_')) e++;
            if (e > at && e < n && t[e] == '}' && at == i + 1) {
                i = e + 1;
                continue;
            }
            if (e > at && e < n && t[e] == '|') {   /* (no limit here: the template's 512 bytes bound it; fill's is the limit) */
                const size_t end = closing(t, e + 1, n, depth + 1);
                if (end >= n) return n;
                i = end + 1;
                continue;
            }
            i++;
            continue;
        }
        if (t[i] == '}') return i;   /* (inside one, every "}" closes: "{age| as {age}}}" is two closes) */
        i++;
    }
    return n;
}

static void add(char *buf, size_t cap, size_t *o, const char *s, size_t n) {
    for (size_t k = 0; k < n && *o + 1 < cap; k++) buf[(*o)++] = s[k];
}

/* t (n bytes) filled into buf at *o: {name} its value, {name|text} text filled in when name's value is not empty, {{ and
   }} braces; a { that starts none of these is text. depth bounds the nesting */
static void fill(const char *t, size_t n, const DiscordFields *f, char *buf, size_t cap, size_t *o, int depth) {
    for (size_t i = 0; i < n;) {
        if ((t[i] == '{' || t[i] == '}') && i + 1 < n && t[i + 1] == t[i]) {
            add(buf, cap, o, t + i, 1), i += 2;
            continue;
        }
        if (t[i] == '{') {
            const int negate = i + 1 < n && t[i + 1] == '!';   /* ({!name|text}: text when name is empty) */
            const size_t at = i + 1 + negate;
            size_t e = at;
            while (e < n && (isalnum((unsigned char)t[e]) || t[e] == '_')) e++;
            const char *value = e < n && (t[e] == '}' || t[e] == '|') ? field(f, t + at, e - at) : NULL;
            if (value && t[e] == '}' && !negate) {
                add(buf, cap, o, value, strlen(value)), i = e + 1;
                continue;
            }
            if (value && t[e] == '|') {
                const size_t end = closing(t, e + 1, n, depth + 1);
                if (end < n) {
                    if ((*value != 0) != negate && depth < 8) fill(t + e + 1, end - e - 1, f, buf, cap, o, depth + 1);
                    i = end + 1;
                    continue;
                }
            }
        }
        add(buf, cap, o, t + i, 1), i++;
    }
}

/* (a cut that does not split a UTF-8 letter: back to the start of the letter at n) */
static size_t utf8_cut(const char *s, size_t n) {
    while (n && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
    return n;
}

/* one line, s (n bytes), cleaned and cut into out */
static void tidy(const char *s, size_t n, char *out, size_t cap) {
    if (!cap) return;
    char buf[600];
    size_t o = 0;
    for (size_t r = 0; r < n && o + 1 < sizeof buf;) {   /* ("[]", "()" and "[ ]" go, runs of spaces become one) */
        const char open = s[r];
        if (open == '[' || open == '(') {
            size_t q = r + 1;
            while (q < n && s[q] == ' ') q++;
            if (q < n && s[q] == (open == '[' ? ']' : ')')) {
                r = q + 1;
                continue;
            }
        }
        if ((s[r] == ' ' || s[r] == '\t') && o && buf[o - 1] == ' ') {
            r++;
            continue;
        }
        buf[o++] = s[r] == '\t' ? ' ' : s[r];
        r++;
    }
    size_t a = 0, b = o;
    for (int again = 1; again;) {   /* (outer spaces, and a dash or a middle dot an empty part left at either end) */
        again = 0;
        while (a < b && buf[a] == ' ') a++;
        while (b > a && buf[b - 1] == ' ') b--;
        if (b - a >= 2 && buf[b - 1] == '-' && buf[b - 2] == ' ') b -= 2, again = 1;
        if (b - a >= 2 && buf[a] == '-' && buf[a + 1] == ' ') a += 2, again = 1;
        if (b - a == 1 && buf[a] == '-') b = a;
        if (b - a >= 3 && (unsigned char)buf[b - 2] == 0xc2 && (unsigned char)buf[b - 1] == 0xb7 && buf[b - 3] == ' ') b -= 3, again = 1;
        if (b - a >= 3 && (unsigned char)buf[a] == 0xc2 && (unsigned char)buf[a + 1] == 0xb7 && buf[a + 2] == ' ') a += 3, again = 1;
        if (b - a == 2 && (unsigned char)buf[a] == 0xc2 && (unsigned char)buf[a + 1] == 0xb7) b = a;
    }
    size_t len = b - a;
    if (len > 128) len = utf8_cut(buf + a, 128);
    if (len > cap - 1) len = utf8_cut(buf + a, cap - 1);
    memcpy(out, buf + a, len);
    out[len] = 0;
    while (len && out[len - 1] == ' ') out[--len] = 0;
}

void discord_lines(const char *tmpl, const DiscordFields *f, char *first, size_t first_cap, char *second, size_t second_cap) {
    char buf[1200];
    size_t o = 0;
    fill(tmpl ? tmpl : "", tmpl ? strlen(tmpl) : 0, f, buf, sizeof buf, &o, 0);
    buf[o] = 0;
    const char *brk = memchr(buf, BREAK, o);
    const size_t n1 = brk ? (size_t)(brk - buf) : o;
    tidy(buf, n1, first, first_cap);
    if (brk) {   /* (a second {break}: a space, the status having two lines) */
        for (char *p = buf + n1 + 1; p < buf + o; p++)
            if (*p == BREAK) *p = ' ';
        tidy(buf + n1 + 1, o - n1 - 1, second, second_cap);
    } else if (second_cap) {
        second[0] = 0;
    }
}

/* ---- the activity ---- */

/* s as a JSON string's inside (quotes, backslashes and control characters escaped) after out[*o]: 0 when it does not fit */
static int json_text(char *out, size_t cap, size_t *o, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char esc[8];
        const char *put_s = esc;
        if (*p == '"') put_s = "\\\"";
        else if (*p == '\\') put_s = "\\\\";
        else if (*p < 0x20) snprintf(esc, sizeof esc, "\\u%04x", *p);
        else esc[0] = (char)*p, esc[1] = 0;
        const size_t n = strlen(put_s);
        if (*o + n + 1 > cap) return 0;
        memcpy(out + *o, put_s, n), *o += n;
    }
    out[*o] = 0;
    return 1;
}

static int put(char *out, size_t cap, size_t *o, const char *s) {
    const size_t n = strlen(s);
    if (*o + n + 1 > cap) return 0;
    memcpy(out + *o, s, n + 1), *o += n;
    return 1;
}

/* (a picture: an art asset's key, letters, digits, _ and -, or an https:// link with no space or quote) */
static int valid_image(const char *s) {
    const size_t n = strlen(s);
    if (!n || n > 256) return 0;
    if (!strncmp(s, "https://", 8)) {
        for (const char *p = s; *p; p++)
            if ((unsigned char)*p <= ' ' || *p == '"' || *p == '\\') return 0;
        return n > 8;
    }
    if (n > 32) return 0;
    for (const char *p = s; *p; p++)
        if (!(isalnum((unsigned char)*p) || *p == '_' || *p == '-')) return 0;
    return 1;
}

size_t discord_activity_json(const DiscordSettings *s, const DiscordFields *f, unsigned nonce, unsigned long pid,
                             long long started, char *out, size_t cap) {
    char details[160], state[160], image_text[160], text_two[8], head[160];
    discord_lines(s->status, f, details, sizeof details, state, sizeof state);
    discord_lines(s->image_text, f, image_text, sizeof image_text, text_two, sizeof text_two);
    const char *image = valid_image(s->image) ? s->image : "hookshot";
    size_t o = 0;
    snprintf(head, sizeof head, "{\"cmd\":\"SET_ACTIVITY\",\"nonce\":\"%u\",\"args\":{\"pid\":%lu,\"activity\":{", nonce, pid);
    int ok = put(out, cap, &o, head);
    if (strlen(details) >= 2) ok = ok && put(out, cap, &o, "\"details\":\"") && json_text(out, cap, &o, details) && put(out, cap, &o, "\",");
    if (strlen(state) >= 2) ok = ok && put(out, cap, &o, "\"state\":\"") && json_text(out, cap, &o, state) && put(out, cap, &o, "\",");
    char stamp[64];
    snprintf(stamp, sizeof stamp, "\"timestamps\":{\"start\":%lld},", started);
    ok = ok && put(out, cap, &o, stamp) && put(out, cap, &o, "\"assets\":{\"large_image\":\"") && json_text(out, cap, &o, image) &&
         put(out, cap, &o, "\"");
    if (strlen(image_text) >= 2) ok = ok && put(out, cap, &o, ",\"large_text\":\"") && json_text(out, cap, &o, image_text) && put(out, cap, &o, "\"");
    ok = ok && put(out, cap, &o, "}}}}");
    if (!ok) {
        if (cap) out[0] = 0;
        return 0;
    }
    return o;
}

void discord_hearts_text(int quarters, char *out, size_t cap) {
    static const char *const parts[4] = {"", ".25", ".5", ".75"};   /* (in plain digits: ¼ is too small to read there) */
    if (!cap) return;
    if (quarters < 0) {
        out[0] = 0;
        return;
    }
    snprintf(out, cap, "%d%s", quarters / 4, parts[quarters % 4]);
}

int discord_in_map(int map_known, int64_t link_seen_ns, int64_t now_ns) {
    return map_known && link_seen_ns && now_ns - link_seen_ns < DISCORD_IN_MAP_NS;
}

/* ---- the version ---- */

/* (a pre-release's name made readable: "rc.7" "RC7", "beta.2" "Beta 2", "nightly-70" "N70") */
static void pretty_build(const char *s, size_t n, char *out, size_t cap) {
    char word[32] = "", num[32] = "";
    size_t i = 0, w = 0, d = 0;
    while (i < n && isalpha((unsigned char)s[i]) && w + 1 < sizeof word) word[w++] = s[i++];
    word[w] = 0;
    while (i < n && (s[i] == '.' || s[i] == '-' || s[i] == '_')) i++;
    while (i < n && isdigit((unsigned char)s[i]) && d + 1 < sizeof num) num[d++] = s[i++];
    num[d] = 0;
    if (!w || i != n) {   /* (not a word and a number: as it is) */
        snprintf(out, cap, "%.*s", (int)n, s);
        return;
    }
    for (size_t k = 0; k < w; k++)
        word[k] = (char)(w <= 2 ? toupper((unsigned char)word[k]) : k ? tolower((unsigned char)word[k]) : toupper((unsigned char)word[k]));
    if (d && !strcmp(word, "Nightly")) snprintf(out, cap, "N%s", num);
    else snprintf(out, cap, "%s%s%s", word, d && w > 2 ? " " : "", num);
}

static int is_hex(const char *s, size_t n) {
    if (!n) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isxdigit((unsigned char)s[i])) return 0;
    return 1;
}

void discord_version_parts(const char *raw, char *version, char *build, char *commit, char *release, size_t cap) {
    if (!cap) return;
    version[0] = build[0] = commit[0] = release[0] = 0;
    if (!raw) return;
    size_t n = strcspn(raw, " +");   /* ("v3.0.0-rc.7 (source ...)": the release alone) */
    snprintf(release, cap, "%.*s", (int)n, raw);
    const char *s = release;
    n = strlen(s);
    /* (from the end: -dirty, then git describe's -<count>-g<hash>) */
    int dirty = 0;
    if (n >= 6 && !strcmp(s + n - 6, "-dirty")) dirty = 1, n -= 6;
    size_t tail = n;
    {
        size_t g = n;
        while (g && s[g - 1] != '-') g--;   /* (the last part) */
        if (g >= 2 && s[g] == 'g' && is_hex(s + g + 1, n - g - 1)) {
            size_t c = g - 1;
            while (c && s[c - 1] != '-') c--;   /* (the count before it) */
            int digits = c < g - 1;
            for (size_t k = c; k < g - 1; k++) digits &= isdigit((unsigned char)s[k]) != 0;
            if (c >= 2 && digits) tail = c - 1;
        }
    }
    if (tail < n) snprintf(commit, cap, "%.*s%s", (int)(n - tail - 1), s + tail + 1, dirty ? "-dirty" : "");
    else if (dirty) snprintf(commit, cap, "dirty");
    n = tail;
    /* (v<numbers>.<numbers>... then an optional -<pre-release>) */
    size_t v = 0;
    if (v < n && s[v] == 'v') v++;
    const size_t digits_at = v;
    while (v < n && (isdigit((unsigned char)s[v]) || s[v] == '.')) v++;
    if (v > digits_at && isdigit((unsigned char)s[digits_at]) && memchr(s + digits_at, '.', v - digits_at)) {
        snprintf(version, cap, "%s%.*s", s[0] == 'v' ? "" : "v", (int)v, s);
        if (v < n && s[v] == '-') pretty_build(s + v + 1, n - v - 1, build, cap);
        else if (v < n) snprintf(build, cap, "%.*s", (int)(n - v), s + v);
    } else if (!commit[0] && is_hex(s, n) && n >= 7) {
        snprintf(commit, cap, "%.*s%s", (int)n, s, dirty ? "-dirty" : "");   /* (a bare hash: a checkout with no tag) */
    } else {
        /* ("nightly-70": a name and a number, made readable, as the build, and the nightly's own name,
           "nightly-486-09a20cd6", has its commit after the number; anything else as it is, the version) */
        size_t dash = 0;
        while (dash < n && s[dash] != '-') dash++;
        size_t end = n;
        if (dash < n) {
            size_t d2 = dash + 1;
            while (d2 < n && s[d2] != '-') d2++;
            if (!commit[0] && d2 + 8 <= n && is_hex(s + d2 + 1, n - d2 - 1)) end = d2;
        }
        int named = dash && dash + 1 < end;
        for (size_t k = 0; k < dash && named; k++) named &= isalpha((unsigned char)s[k]) != 0;
        for (size_t k = dash + 1; k < end && named; k++) named &= isdigit((unsigned char)s[k]) != 0;
        if (named) {
            pretty_build(s, end, build, cap);
            if (end < n) snprintf(commit, cap, "%.*s%s", (int)(n - end - 1), s + end + 1, dirty ? "-dirty" : "");
        } else {
            snprintf(version, cap, "%.*s", (int)n, s);
        }
    }
}
