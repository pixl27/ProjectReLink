/* The game's window (Windows): the renderer's surface (gpu/render/, WebGPU). The native NVN layer (nvn_native.c)
   presents the game's window textures into it (rx_nvn_present). The title bar carries live numbers.
   TOTKPort's hos/display.c, without its emulated-GPU fallback (the window buffer read from guest memory): MKW has
   no emulated GPU, so the renderer is the only way to a picture.
   HOS_HEADLESS=1: no window (scripted runs; the renderer draws offscreen). Closing the window ends the run.
   HOS_WINDOW_NOACTIVATE=1: the window opens behind the one in front (test runs do not take the keyboard).
   HOS_WINDOW_SIZE=<w>x<h>: the client area (default 1280x720 at 96 dpi). F11 or Alt+Enter: fullscreen.
   Mouse: a click in the window captures the mouse (hidden, kept in the window), Esc or leaving the window releases
   it (nn_hid.c reads its motion). While the Insert menu is open the mouse is free (the cursor shown, for the menu's
   clicks: hos_mouse_pointer) and the game gets none of it; the capture it had comes back when the menu closes. */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../gpu/render/rx.h"
void rx_close_asked(void);   /* host/rx_backend.c */
#include "../gpu/vk/vk_api.h"

void hos_relaunch_ask(void);        /* (host/hos_relaunch.c) */
void hos_window_closed_exit(void);
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0
#endif

volatile LONG64 hos_present_count;   /* (nvn_native.c: the game's presents) */
static HWND hwnd;
static int renderer;   /* the renderer presents into the window */

/* mouse capture: raw motion accumulated here, taken by nn_hid.c */
static volatile LONG mouse_dx, mouse_dy;
static volatile LONG mouse_captured;
static int cursor_hidden;

/* (relink) HOS_MOUSE_TEST=<counts a second>: a test run has no mouse and its window is behind the others. With this the mouse
   counts as captured and moves to the right at that rate, so a run can see whether the camera goes on turning (the mouse look
   that stopped after a scene change, issue 2). 0 or unset: the real mouse alone. */
static int mouse_test_rate(void) {
    static int rate = -1;
    if (rate < 0) {
        const char *e = getenv("HOS_MOUSE_TEST");
        rate = e ? atoi(e) : 0;
        if (rate < 0) rate = 0;
        if (rate) hos_log("mouse: a test's mouse, captured, %d counts a second to the right (HOS_MOUSE_TEST)", rate);
    }
    return rate;
}
int hos_mouse_test_on(void) { return mouse_test_rate() > 0; }   /* (nn_hid.c: read even while a script is the only input) */
static int mouse_test(int32_t *dx) {
    static int64_t last;
    const int rate = mouse_test_rate();
    if (rate <= 0) return 0;
    const int64_t now = hos_now_ns();
    if (!last) last = now;
    const int64_t n = (now - last) * rate / 1000000000;
    if (n > 0) {
        last += n * 1000000000 / rate;
        *dx += (int32_t)n;
    }
    return 1;
}

int hos_mouse_take(int32_t *dx, int32_t *dy) {
    *dx = InterlockedExchange(&mouse_dx, 0);
    *dy = InterlockedExchange(&mouse_dy, 0);
    if (mouse_test(dx)) return 1;
    return mouse_captured;
}

/* (mouse_native.c) motion taken and not used, put back for the next taker (the stick's, while no camera is found) */
void hos_mouse_give_back(int32_t dx, int32_t dy) {
    InterlockedExchangeAdd(&mouse_dx, dx);
    InterlockedExchangeAdd(&mouse_dy, dy);
}

/* the free mouse, for the Insert menu (gpu/vk/vk_api.h MkwVkPointer): set on the window's thread, read on the renderer's */
static volatile LONG ptr_x, ptr_y, ptr_w, ptr_h, ptr_in, ptr_pressed, ptr_wheel;   /* (client pixels; wheel in WHEEL_DELTA units) */
static int menu_open, captured_before_menu;

int hos_key_down(int vk);
int hos_mouse_pointer(MkwVkPointer *out) {
    memset(out, 0, sizeof *out);
    if (!hwnd || mouse_captured) return 0;
    out->x = ptr_x, out->y = ptr_y, out->width = ptr_w, out->height = ptr_h;
    out->held = (hos_key_down(VK_LBUTTON) ? HOS_MOUSE_LEFT : 0) | (hos_key_down(VK_RBUTTON) ? HOS_MOUSE_RIGHT : 0) |
                (hos_key_down(VK_MBUTTON) ? HOS_MOUSE_MIDDLE : 0);
    out->pressed = (uint32_t)InterlockedExchange(&ptr_pressed, 0);
    const LONG wheel = InterlockedExchange(&ptr_wheel, 0);
    out->wheel = wheel / WHEEL_DELTA;
    if (wheel % WHEEL_DELTA) InterlockedExchangeAdd(&ptr_wheel, wheel % WHEEL_DELTA);   /* (a part of a notch: kept) */
    return ptr_in && GetForegroundWindow() == hwnd && out->width > 0 && out->height > 0;
}

static HANDLE renderer_up;
static int fullscreen;
static WINDOWPLACEMENT windowed_place;

static void set_fullscreen(HWND w, int on) {
    if (on == fullscreen) return;
    const LONG style = GetWindowLongA(w, GWL_STYLE);
    if (on) {
        MONITORINFO mi;
        mi.cbSize = sizeof mi;
        windowed_place.length = sizeof windowed_place;
        if (!GetWindowPlacement(w, &windowed_place) ||
            !GetMonitorInfoA(MonitorFromWindow(w, MONITOR_DEFAULTTONEAREST), &mi))
            return;
        fullscreen = 1;
        SetWindowLongA(w, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(w, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    } else {
        fullscreen = 0;
        SetWindowLongA(w, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(w, &windowed_place);
        SetWindowPos(w, NULL, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    hos_log("display: %s", on ? "fullscreen (F11 for a window)" : "windowed");
}

static void dpi_aware_thread(void) {
    if (getenv("HOS_DPI_UNAWARE")) return;
    typedef void *(WINAPI * SetCtx)(void *);
    SetCtx set = (SetCtx)(void *)GetProcAddress(GetModuleHandleA("user32.dll"), "SetThreadDpiAwarenessContext");
    if (set) set((void *)(intptr_t)-4);
}

static UINT window_dpi(HWND w) {
    typedef UINT(WINAPI * GetDpi)(HWND);
    GetDpi get = (GetDpi)(void *)GetProcAddress(GetModuleHandleA("user32.dll"), "GetDpiForWindow");
    const UINT dpi = get ? get(w) : 96;
    return dpi ? dpi : 96;
}

static void mouse_capture(HWND w, int on) {
    if (on == (int)mouse_captured) return;
    mouse_captured = on;
    InterlockedExchange(&mouse_dx, 0);
    InterlockedExchange(&mouse_dy, 0);
    if (on) {
        RECT rc;
        GetClientRect(w, &rc);
        MapWindowPoints(w, NULL, (POINT *)&rc, 2);
        ClipCursor(&rc);
        if (!cursor_hidden) { ShowCursor(FALSE); cursor_hidden = 1; }
    } else {
        ClipCursor(NULL);
        if (cursor_hidden) { ShowCursor(TRUE); cursor_hidden = 0; }
    }
}

/* the menu opened: the mouse let go (its capture kept for later); closed: the capture back, if the window still has the focus */
static void menu_mouse(HWND w, int open) {
    if (open == menu_open) return;
    menu_open = open;
    if (open) {
        captured_before_menu = (int)mouse_captured;
        mouse_capture(w, 0);
        InterlockedExchange(&ptr_pressed, 0), InterlockedExchange(&ptr_wheel, 0);
    } else {
        if (captured_before_menu && GetForegroundWindow() == w) mouse_capture(w, 1);
        captured_before_menu = 0;
    }
}
static void menu_press(HWND w, LPARAM lp, LONG bit) {   /* (a click while the menu is open: the menu's) */
    ptr_x = (short)LOWORD(lp), ptr_y = (short)HIWORD(lp), ptr_in = 1;
    InterlockedOr(&ptr_pressed, bit);
    SetCapture(w);   /* (a drag past the window's edge still reaches it) */
}

/* (relink, issue 3: "not even the keyboard is recognized in the game ... I can press F1 to bring up the settings ... I can
   press ESC to cancel changing a control, but I can't press any keyboard key")
   The game's keys and the menu's key bindings were read by asking Windows for the keyboard's state (GetAsyncKeyState);
   F1, Esc and the menu's own keys come as the window's messages. On that player's PC the messages arrived and the state
   answered nothing (what a program that hides the keyboard's state from the others does, a key-logger guard for one; which
   it was there is not known), and the game could not be played at all. The window now keeps the keys its own messages say are down, mouse buttons too, cleared when
   it loses the focus; hos_key_down answers from them once Windows' state has been seen to answer nothing: three keys
   pressed in the window that it called up, and none it called down. Where it answers, nothing changes. */
static volatile unsigned char key_msg[256];
static volatile LONG key_state_said, key_state_missed;
static int key_state_deaf(void) { return key_state_missed >= 3 && !key_state_said; }
static void key_note(unsigned vk, int down, int first) {
    if (vk >= 256) return;
    key_msg[vk] = (unsigned char)down;
    if (!down || !first || vk < 8) return;   /* (the mouse's buttons are not asked about: a click is over by now more often) */
    const int was_deaf = key_state_deaf();
    if (GetAsyncKeyState((int)vk) & 0x8000) InterlockedIncrement(&key_state_said);
    else InterlockedIncrement(&key_state_missed);
    if (!was_deaf && key_state_deaf())
        hos_log("hid: Windows says no key is down while keys reach the window (something hides the keyboard's state from "
                "the game): the window's own key messages are used for the game's keys");
}
static void key_messages(UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN: key_note((unsigned)wp, 1, !(lp & (1 << 30))); break;
    case WM_KEYUP:
    case WM_SYSKEYUP: key_note((unsigned)wp, 0, 0); break;
    case WM_LBUTTONDOWN: key_note(VK_LBUTTON, 1, 0); break;
    case WM_LBUTTONUP: key_note(VK_LBUTTON, 0, 0); break;
    case WM_RBUTTONDOWN: key_note(VK_RBUTTON, 1, 0); break;
    case WM_RBUTTONUP: key_note(VK_RBUTTON, 0, 0); break;
    case WM_MBUTTONDOWN: key_note(VK_MBUTTON, 1, 0); break;
    case WM_MBUTTONUP: key_note(VK_MBUTTON, 0, 0); break;
    case WM_XBUTTONDOWN: key_note(HIWORD(wp) == 1 ? VK_XBUTTON1 : VK_XBUTTON2, 1, 0); break;
    case WM_XBUTTONUP: key_note(HIWORD(wp) == 1 ? VK_XBUTTON1 : VK_XBUTTON2, 0, 0); break;
    case WM_KILLFOCUS: memset((void *)key_msg, 0, sizeof key_msg); break;
    }
}
/* (nn_hid.c, the Insert menu through rx_backend.c) whether a key or a mouse button is down. HOS_KEY_MESSAGES=1: as on a PC
   whose keyboard state answers nothing, from the window's messages alone (a test) */
int hos_key_down(int vk) {
    static int only = -1;
    if (only < 0) only = getenv("HOS_KEY_MESSAGES") != NULL;
    if (!only && (GetAsyncKeyState(vk) & 0x8000)) return 1;
    return (only || key_state_deaf()) && key_msg[vk & 255];
}

static LRESULT CALLBACK wndproc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    key_messages(msg, wp, lp);
    switch (msg) {
    case WM_MOUSEMOVE:
        ptr_x = (short)LOWORD(lp), ptr_y = (short)HIWORD(lp);
        if (!ptr_in) {
            TRACKMOUSEEVENT t = {sizeof t, TME_LEAVE, w, 0};
            TrackMouseEvent(&t);
            ptr_in = 1;
        }
        return 0;
    case WM_MOUSELEAVE:
        ptr_in = 0;
        return 0;
    case WM_MOUSEWHEEL:
        if (menu_open) InterlockedExchangeAdd(&ptr_wheel, GET_WHEEL_DELTA_WPARAM(wp));
        return 0;
    case WM_RBUTTONDOWN:
        if (menu_open) menu_press(w, lp, HOS_MOUSE_RIGHT);
        return 0;
    case WM_MBUTTONDOWN:
        if (menu_open) menu_press(w, lp, HOS_MOUSE_MIDDLE);
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
        if (GetCapture() == w && !(wp & (MK_LBUTTON | MK_RBUTTON | MK_MBUTTON))) ReleaseCapture();
        return 0;
    case WM_APP + 8:   /* (the Insert menu opened or closed: gpu/vk/vk_menu.cpp) */
        menu_mouse(w, (int)wp);
        return 0;
    case WM_TIMER:
        return 0;
    case WM_SYSCOMMAND:
        if ((wp & 0xFFF0) == SC_SCREENSAVE && !(wp & SCF_ISSECURE) &&
            renderer && GetForegroundWindow() == w && !IsIconic(w)) {
            BOOL secure = TRUE;   /* unknown or protected policy keeps Windows' normal handling */
            if (SystemParametersInfoA(SPI_GETSCREENSAVESECURE, 0, &secure, 0) && !secure) return 0;
        }
        break;
    case WM_INPUT: {
        RAWINPUT ri;
        UINT size = sizeof ri;
        if (mouse_captured && GetRawInputData((HRAWINPUT)lp, RID_INPUT, &ri, &size, sizeof(RAWINPUTHEADER)) != (UINT)-1 &&
            ri.header.dwType == RIM_TYPEMOUSE && !(ri.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)) {
            InterlockedExchangeAdd(&mouse_dx, ri.data.mouse.lLastX);
            InterlockedExchangeAdd(&mouse_dy, ri.data.mouse.lLastY);
        }
        return DefWindowProcA(w, msg, wp, lp);
    }
    case WM_LBUTTONDOWN:
        if (menu_open) menu_press(w, lp, HOS_MOUSE_LEFT);
        else if (renderer && !mouse_captured) mouse_capture(w, 1);
        return 0;
    case WM_KEYDOWN: {
        extern void hos_vk_note_key_press(unsigned key);
        if (!(lp & (1 << 30))) hos_vk_note_key_press((unsigned)wp);
        if (wp == VK_ESCAPE) mouse_capture(w, 0);
        if (wp == VK_F11 && !(lp & (1 << 30))) set_fullscreen(w, !fullscreen);
        return 0;
    }
    case WM_SYSKEYDOWN:   /* Alt+Enter, as on Linux (display_sdl.c); other Alt keys (Alt+F4) go on to Windows */
        if (wp == VK_RETURN && (lp & (1 << 29))) {
            if (!(lp & (1 << 30))) set_fullscreen(w, !fullscreen);
            return 0;
        }
        break;
    case WM_SYSCHAR:   /* (its character: without this Windows beeps, as for an Alt key with no menu) */
        if (wp == '\r') return 0;
        break;
    case WM_APP + 7:   /* (the Insert menu's row) */
        set_fullscreen(w, (int)wp);
        return 0;
    case WM_DPICHANGED:
        if (!fullscreen) {
            const RECT *s = (const RECT *)lp;
            SetWindowPos(w, NULL, s->left, s->top, s->right - s->left, s->bottom - s->top,
                         SWP_NOZORDER | SWP_NOACTIVATE);
        }
        return 0;
    case WM_KILLFOCUS:
    case WM_CANCELMODE:
        mouse_capture(w, 0);
        break;
    case WM_SETCURSOR:
        if (mouse_captured && LOWORD(lp) == HTCLIENT) {
            SetCursor(NULL);
            return TRUE;
        }
        break;
    case WM_MOVE:
        if (mouse_captured) { mouse_capture(w, 0); mouse_capture(w, 1); }   /* (the clip follows the window) */
        break;
    case WM_SIZE:
        ptr_w = LOWORD(lp), ptr_h = HIWORD(lp);
        if (mouse_captured) { mouse_capture(w, 0); mouse_capture(w, 1); }
        if (renderer) rx_resize(LOWORD(lp), HIWORD(lp));
        return 0;
    case WM_PAINT:
        if (renderer) {   /* the renderer's surface: nothing to paint */
            ValidateRect(w, NULL);
            return 0;
        }
        break;
    case WM_ERASEBKGND:
        return renderer ? 1 : DefWindowProcA(w, msg, wp, lp);
    case WM_APP + 9:   /* (the Insert menu's Restart now: closed as by the player, then started again: host/hos_relaunch.c) */
        hos_relaunch_ask();
        /* fall through */
    case WM_CLOSE:
        rx_close_asked();   /* (a start still preparing the shaders keeps what it made) */
        hos_window_closed_exit();
        return 0;
    }
    return DefWindowProcA(w, msg, wp, lp);
}

static DWORD WINAPI display_main(void *p) {
    (void)p;
    {   /* (nothing is typed here. Windows' text services, set up under DefWindowProc as the window was sized, ended the process: a
           fail-fast in CoreMessaging, called from textinputframework) */
        HMODULE imm = LoadLibraryA("imm32.dll");
        BOOL (WINAPI *ime_off)(DWORD) = imm ? (BOOL (WINAPI *)(DWORD))GetProcAddress(imm, "ImmDisableIME") : NULL;
        if (ime_off) ime_off((DWORD)-1);
    }
    WNDCLASSA wc;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleA(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hIcon = LoadIconA(wc.hInstance, MAKEINTRESOURCEA(1));   /* (nx2/make_icon.py; none in a build without the picture) */
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = "ProjectReLink";
    RegisterClassA(&wc);
    dpi_aware_thread();
    extern float hos_aspect_ratio;
    int win_w = hos_aspect_ratio > 1.8f ? (int)(720 * hos_aspect_ratio + 0.5f) : 1280, win_h = 720;
    if (getenv("HOS_WINDOW_SIZE")) {
        int w = 0, h = 0;
        if (sscanf(getenv("HOS_WINDOW_SIZE"), "%dx%d", &w, &h) == 2 && w >= 64 && h >= 64) win_w = w, win_h = h;
    }
    RECT r = {0, 0, win_w, win_h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int noactivate = getenv("HOS_WINDOW_NOACTIVATE") != NULL;
    hwnd = CreateWindowA("ProjectReLink", HOS_GAME_TITLE, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                         CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, NULL, NULL, wc.hInstance, NULL);
    if (window_dpi(hwnd) != 96) {
        const UINT dpi = window_dpi(hwnd);
        RECT s = {0, 0, (LONG)(win_w * dpi / 96), (LONG)(win_h * dpi / 96)};
        AdjustWindowRect(&s, WS_OVERLAPPEDWINDOW, FALSE);
        SetWindowPos(hwnd, NULL, 0, 0, s.right - s.left, s.bottom - s.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
    /* (no activate: shown at the bottom of the z-order, so a game in front keeps its focus) */
    if (noactivate)
        SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    else
        ShowWindow(hwnd, SW_SHOW);
    if (getenv("HOS_FULLSCREEN") && atoi(getenv("HOS_FULLSCREEN"))) set_fullscreen(hwnd, 1);
    RAWINPUTDEVICE rid = {1, 2, 0, hwnd};   /* generic desktop mouse: WM_INPUT while the window is in front */
    RegisterRawInputDevices(&rid, 1, sizeof rid);
    renderer = rx_start(hwnd);
    if (!renderer) hos_log("display: no WebGPU device: the window stays black");
    SetEvent(renderer_up);
    SetTimer(hwnd, 1, 250, NULL);
    MSG m;
    while (GetMessageA(&m, NULL, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageA(&m);
    }
    return 0;
}

/* (relink) The game's first work for the GPU waits for the renderer. It ran beside the renderer's start, and what it submitted while the
   shaders were being prepared (80 s on a first start) was lost: that run had no ambient light, black foliage and dark characters. The
   game itself goes on starting as it did: held at the start instead, it left holes in the ground */
void hos_renderer_wait(void) {
    { const char *e = getenv("HOS_NO_RENDERER_WAIT"); if (e && atoi(e) != 0) return; }   /* (=0: the wait, as without it) */   /* (a test: the game's first GPU work no longer waits for the renderer, as before lighting-fix) */
    if (renderer_up) WaitForSingleObject(renderer_up, INFINITE);
}

void hos_display_start(void) {
    if (getenv("HOS_HEADLESS")) {   /* no window: the renderer still draws, offscreen (screenshots, frame dumps) */
        rx_start(NULL);
        return;
    }
    renderer_up = CreateEventA(NULL, TRUE, FALSE, NULL);
    CreateThread(NULL, 0, display_main, NULL, 0, NULL);
}
