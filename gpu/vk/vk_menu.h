// The Insert menu (vk_menu.cpp): options that apply while the game runs, kept in relink_options.ini beside the executable.
#pragma once

#include "vk_internal.h"

namespace mkwvk {

// (relink) Option numbers the host asks for (rx_option, host/rx_backend.c), kept here and not in vk_api.h: that file is part
// of the shader cache's stamp, and a line added there makes every player prepare their shaders again at the update.
// The Controls tab's Left / Right stick dead zone, in percent of the stick's travel (host/hos_stick.h)
enum { HOS_OPT_DEADZONE_L = 10, HOS_OPT_DEADZONE_R = 11 };
// the game frames the display has taken, modulo 2^30 (host/nn_vi.c paces HOS_FPS_CAP's frames by it)
enum { HOS_OPT_PRESENTS_SHOWN = 12 };
// the Graphics tab's Field of view, in percent of the game's camera angle (host/nn_vi.c, nx2/oot_fov.py)
enum { HOS_OPT_FOV = 13 };
// HOS_FPS_CAP's frames as the display shows them, in thousandths of a frame a second (100 Hz holding 3 refreshes: 33333);
// 0 when the display does not hold them (host/nn_vi.c paces to it)
enum { HOS_OPT_LOW_CAP_MFPS = 14 };
// (relink) HDR output (vk_main.cpp hdr_output): whether the window's images are HDR10, so the game is told its display is HDR
// (host/nn_vi.c QueryDisplayHdrStatus) and draws its own HDR picture; the display's peak and the paper white, in nits
enum { HOS_OPT_HDR = 70, HOS_OPT_HDR_PEAK = 71, HOS_OPT_HDR_PAPER_WHITE = 72 };
// (relink) the fullscreen key's virtual-key code, 0 none (host/display.c; the Gameplay tab's Fullscreen key row)
enum { HOS_OPT_KEY_FULLSCREEN = 73 };

// (out of the box every feature is off: what the game does on its own. The ini beside the executable keeps a player's choices)
struct Options {
    int framegen = 0;     // made frames: 0 off, or 2, 3, 4 times the game's frames
    int pace = 0;         // frames shown on an even line (vk_framegen.cpp)
    int vsync = 0;        // (relink) V-Sync: 0 off (mailbox where the driver has it, else immediate, which can tear), 1 on
                          // (FIFO: every frame waits for the display's refresh; vk_main.cpp make_swapchain)
    // (relink: 1 by default, the 3D at the picture's size. Auto ran at 1536 x 864 and below for a 2048 x 1152 picture)
    int resolution = 1;   // the 3D's (Render resolution): 0 Auto (the game's dynamic resolution), else a share of the output,
                          // 1..6 = 100%, 75%, 67%, 50%, 58%, 33% (upscaler_rows.h; nx2/oot_patches.py)
    int output_res = 0;   // the game's whole frame, docked: 0 its 2048x1152, 1 2560x1440, 2 3840x2160, 3 1920x1080; read once, at the start (host/nn_applet.c)
    int upscale = 0;      // to the window (Window scaling filter, saved as upscaling): 0 off (the plain copy), 1 Catmull-Rom, 2 Lanczos
    int reflex = 0;       // NVIDIA Reflex: 0 off, 1 on, 2 on + boost (vk_reflex.cpp; NVIDIA GPUs with VK_NV_low_latency2)
    int dlss = 0;         // the Upscaler: 0 the game's own network, 1 NVIDIA DLSS, 2 NVIDIA DLAA (vk_dlss.cpp; a build with NVIDIA's
                          // SDK only). Saved as dlss
    int dlss_model = 0;   // DLSS's network, by its preset: 0 K (transformer), 1 M, 2 L (second-generation transformer), 3 NVIDIA's
                          // default (upscaler_rows.h model_preset; vk_dlss.cpp). RX_VK_DLSS_PRESET over it
    int sharpness = 0;    // 0 to 100
    int realtime = 1;     // game time follows the clock (host/nn_vi.c); 0: per frame (slower frames, slower game)
    int handheld = 0;     // the game's handheld profile: ~4 fps, softer; read once, at the start (host/nn_applet.c)
    int ultrawide = 0;    // the camera at the monitor's shape when it is wider than 16:9; read once, at the start (host/hos_main.c)
    int show_fps = 0;
    int fullscreen = 0;   // borderless, at the monitor's size: F11 is the same setting
    int hdr = 0;          // (relink) HDR output: the game's own HDR picture in HDR10, when Windows' HDR is on for the display;
                          // read once, at the start (vk_main.cpp hdr_output; RX_VK_HDR over it)
    // (relink) the menu's, the screenshot's and the fullscreen's keys, as indexes of vk_menu.cpp's kHotkeys (F1..F12, Insert, Pause,
    // Scroll Lock, None): Insert, F12 and F11 by default. F1 still opens the menu while no other of these keys is F1
    int key_menu = 12, key_shot = 11, key_full = 10;
    int hdr_paper_white = 203;   // (relink) the nits of the game's interface and SDR white in HDR (relink_options.ini only:
                                 // hdr_paper_white, 80 to 500; HOS_HDR_PAPER_WHITE over it)
    int gpu_device = 0;   // graphics card index: 0 Auto, 1..N device 0..N-1 (gpu/vk/vk_main.cpp; RX_VK_DEVICE)
    int pad_background = 0;   // the controllers count while another window has the focus (host/nn_hid.c; HOS_INPUT_BACKGROUND)
    int mouse_native = 0; // the mouse turns the camera's own angles (host/mouse_native.c); 0: as the right stick
    int mouse_sens = 100; // the mouse's sensitivity in percent: 0.1 degrees a count at 100 with native look (host/nn_hid.c)
    int mouse_invert = 0; // the mouse's up and down swapped
    int text_speed = 0;   // how fast dialogue prints: 0 the game's, 1 2x, 2 4x, 3 instant (host/nn_vi.c, nx2/oot_patches.py; HOS_TEXT_SPEED)
    int invert_cam_x = 0; // right-stick camera horizontal inversion: 0 normal, 1 inverted
    int invert_cam_y = 0; // right-stick camera vertical inversion: 0 normal, 1 inverted
    int rumble = 100;     // the controllers' rumble in percent of the game's: 0 Off, 25 to 150 (host/pad_sdl.c)
    int deadzone_l = 24;  // (relink) the left stick's round dead zone in percent, 0 to 50 (host/hos_stick.h); 24: XInput's
    int deadzone_r = 24;  // (relink) the right stick's
    int fov = 100;        // (relink) the camera's field of view in percent of the game's, 70 to 150 (host/nn_vi.c, nx2/oot_fov.py)
    int pad_layout = 0;   // 0 Nintendo (default), 1 Xbox / PC, 2 Swap Face Buttons, 3 Custom
    int button_prompts = 0;   // (relink) the game's button prompts: 0 its own (Nintendo), 1 Xbox, 2 PlayStation; read once, at
                              // the start (host/prompts.c; HOS_BUTTON_PROMPTS over it)
    int gyro = 1;         // motion controls from a pad's gyroscope (host/pad_sdl.c, host/hos_motion.h); on by default
    int gyro_sens_x = 100;   // the motion aim's horizontal sensitivity in percent
    int gyro_sens_y = 100;   // the motion aim's vertical sensitivity in percent
    int gyro_inv_x = 0;   // the motion aim's horizontal turned around
    int gyro_inv_y = 0;   // the motion aim's vertical turned around
    int robust = 0;       // bounds-checked buffer access: 0 Auto (on), 1 Off, 2 On (gpu/vk/vk_main.cpp; RX_VK_ROBUST)
    int preset = 0;       // quality preset: 0 Custom, 1 Minimum (4GB), 2 Low (6GB), 3 Balanced (8GB), 4 High (12GB), 5 Ultra (16GB+)
    int vram_mb = 0;      // dynamic texture cache budget limit in MB
    int mirror_mb = 0;    // staging mirror limit in MB
    int audio_output = 0; // (relink) the speakers the game mixes for: 0 Auto (the default output's), 1 Stereo, 2 5.1; read once, at the start (host/nn_audio.c; HOS_AUDIO_CHANNELS)
    int volume = 100;     // (relink) the game's whole sound in percent, 0 to 100, heard at once (host/nn_audio.c, hos_volume.h)
    int mic = 1;          // (relink) the ocarina's microphone: 0 Off, 1 the system's default, 2.. a device as host->mic_name lists it (host/nn_audio_in.c)
    int framerate = 0;    // 0: the game's 60 fps; 1: unlocked, up to the display's refresh, game time following the clock (host/nn_vi.c)
    int item_wheel = 0;   // (relink) R or L held shows the item row as a wheel (host/oot_wheel.c, vk_wheel.cpp); HOS_ITEM_WHEEL over it
    int menu_dark = 0;    // (relink) the menu's look (Menu look, saved as menu_look): 0 light, 1 dark (vk_menu.cpp cart::set_look;
                          // the launcher's look row writes it too: launcher/Settings.cs)
    int cycle_items = 0;  // (relink) a tap of R or L changes the ZR item: 0 off (the default: R and L as the game has them),
                          // 1 Last used (the item on ZR before this one), 2 Basic, 3 +Bombs, 4 +Tools (within the ZR item's
                          // group; host/oot_wheel.c cycle_level)
};
extern Options opts;
/// (vk_memory.cpp) true when the quality preset's memory budgets (vram_mb, mirror_mb) are to be used: the card has no more
/// video memory than the preset is for. A 4GB preset on an 8 GB card kept 256 MB of mirrors and 2.2 GB of textures: the
/// game's memory was read over PCIe and the textures made again every frame, 5 fps with the CPU idle (Discord, 11-08)
bool preset_caps_memory();

void menu_init();             // (the options from the file, once)
/// (relink, vk_main.cpp) HDR output: the player asked for it (the HDR row, RX_VK_HDR), and whether the window's images are HDR10
/// now (the swapchain made with the ST 2084 colour space); the display's peak brightness in nits (DXGI's, 1000 when unknown)
bool hdr_wanted();
bool hdr_output();
int hdr_peak_nits();
/// (relink, vk_menu.cpp) the paper white in nits: HOS_HDR_PAPER_WHITE, else hdr_paper_white of relink_options.ini (80 to 500)
int hdr_paper_white_nits();
/// (relink, vk_menu.cpp) a hotkey row's value as a Windows virtual-key code (0: None)
int hotkey_vk(int index);
void menu_set_gpu_count(int count);
void menu_tick();             // (each frame shown: the keys, while the game's window has the focus)
bool menu_is_open();
bool resolution_auto();       // (the game's dynamic resolution: Resolution Auto)
int resolution_scale();       // (the fixed path's scale in 1/10000: the Resolution row's, as a twelfth of the output)
void menu_option_set(int id, int value);   // (from the game's side: 7 the mouse's sensitivity, the - and = keys)
int menu_heart_protection();
int menu_player_flight();
int menu_free_camera();
int menu_take_cheat_action();
int menu_item_count(int i);   // (relink) the game side's count of oot_cheat_items.h's item i, -1 unknown: the item wheel's numbers
int menu_link_age();          // (relink) Link's age as the game side said it: 0 child, 1 adult, -1 unknown
int menu_take_capture_request();   // (relink, vk_replay.cpp: a place whose picture the Travel page wants now, -1 none)
void menu_store_capture(int place, const uint8_t* rgba, uint32_t w, uint32_t h);   // (that picture, RGBA: to the picture worker)
bool menu_teleport_captures_on();   // (relink: whether the Travel page's pictures are taken: the game side watches only then)
bool menu_take_screenshot_request();   // (relink, vk_replay.cpp: F12 was pressed since the last picture taken)
/// (relink) F12's picture (RGBA, the game's picture as presented): a worker writes it as a PNG in the game folder's
/// screenshots folder and says so over the game (vk_screenshot.h)
void menu_store_screenshot(std::vector<uint8_t>&& rgba, uint32_t w, uint32_t h);
void menu_note_frame(bool game_frame);   // (the frame rates the FPS counter shows)
/// The game's picture's size (the Output resolution, its crop) and the window's, at each present: the Window scaling filter
/// row says when it is not needed
void menu_note_present_sizes(uint32_t picture_w, uint32_t picture_h, uint32_t window_w, uint32_t window_h);
/// Over the window's image (TRANSFER_DST_OPTIMAL), w x h: the menu and the FPS counter. slot: the frame in flight, 0 to 2.
void menu_record(VkCommandBuffer cmd, VkImage dst, int slot, uint32_t w, uint32_t h);
/// (relink) The start-up screen while the shaders are prepared (vk_main.cpp, vk_startup_paint.h), drawn by menu_record under the
/// counter and the menu. Null takes it away.
struct StartupView;
void startup_screen(const StartupView* view);

} // namespace mkwvk
