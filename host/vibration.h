/* (hookshot) What plays each controller's vibration, and what the Controls page says about it. Pure: no SDL, no state;
   host/pad_sdl.c decides with it and tests/test_vibration_pick.py compiles it.

   The game sends a value per side: a low and a high band, an amplitude and a frequency each. Each pad plays it one way:
     VIB_MOTORS     SDL's rumble, the bands mapped onto two motors (host/nn_hid.c's rumble_map): every pad SDL opens
     VIB_HD         the Switch pads' own HD rumble packets (host/hd_rumble.h): the bands as they are
     VIB_RH_DS_USB  RenHDrum (host/rumble_renhdrum.c): a DualSense on a cable, its voice coils through its own sound card
     VIB_RH_DS_BT   RenHDrum: a DualSense over Bluetooth, its voice coils through HID report 0x32
     VIB_RH_XBOX    RenHDrum: an Xbox One or Series pad's four motors, its triggers' too
     VIB_RH_SWITCH  RenHDrum: a Switch pad's HD rumble packets, made by RenHDrum's engine instead of ours

   The menu's Vibration row: Auto plays the best a pad can do (vibration_pick below), Basic the motors on every pad, Off
   nothing. Auto takes RenHDrum only where people have played it on the pad and found it better: a cabled DualSense. The
   others are there for testers, with HOS_RUMBLE_BACKEND=renhdrum, until they have been. */
#pragma once
#include <stdlib.h>
#include <string.h>

enum { VIB_MODE_AUTO = 0, VIB_MODE_BASIC = 1, VIB_MODE_OFF = 2 };   /* the Vibration row (HOS_OPT_VIBRATION) */
enum { VIB_FORCE_AUTO = 0, VIB_FORCE_MOTORS, VIB_FORCE_HD, VIB_FORCE_RENHDRUM };   /* HOS_RUMBLE_BACKEND */
enum { VIB_PAD_OTHER = 0, VIB_PAD_DUALSENSE, VIB_PAD_XBOX, VIB_PAD_SWITCH };
enum { VIB_NONE = 0, VIB_MOTORS, VIB_HD, VIB_RH_DS_USB, VIB_RH_DS_BT, VIB_RH_XBOX, VIB_RH_SWITCH };

/* HOS_RUMBLE_BACKEND: unset or auto, motors, hd, renhdrum (anything else: auto) */
static inline int vibration_force(const char *env) {
    if (!env || !*env) return VIB_FORCE_AUTO;
    if (!strcmp(env, "motors")) return VIB_FORCE_MOTORS;
    if (!strcmp(env, "hd")) return VIB_FORCE_HD;
    if (!strcmp(env, "renhdrum")) return VIB_FORCE_RENHDRUM;
    return VIB_FORCE_AUTO;
}

/* A pad, from SDL's type for it (sdl: 2 or 3, whose numbers differ) and its name: a DualSense or DualSense Edge, an Xbox
   One or Series pad (an Xbox 360 pad has no trigger motors: it is any other pad here), a Switch Pro Controller or
   Joy-Con. A DualShock 4 is Sony's too, but its sound card is two channels and it has no voice coils: another pad. */
static inline int vibration_pad_kind(int sdl, int type, const char *name) {
    if (sdl == 3) {   /* SDL_GamepadType: 3 XBOXONE, 6 PS5, 7 to 10 Switch Pro and Joy-Cons */
        if (type == 6) return VIB_PAD_DUALSENSE;
        if (type == 3) return VIB_PAD_XBOX;
        if (type >= 7 && type <= 10) return VIB_PAD_SWITCH;
    } else if (sdl == 2) {   /* SDL_GameControllerType: 2 XBOXONE, 7 PS5, 5 Switch Pro, 11 to 13 Joy-Cons */
        if (type == 7) return VIB_PAD_DUALSENSE;
        if (type == 2) return VIB_PAD_XBOX;
        if (type == 5 || (type >= 11 && type <= 13)) return VIB_PAD_SWITCH;
    }
    if (name && strstr(name, "DualSense")) return VIB_PAD_DUALSENSE;   /* (an SDL without the type) */
    return VIB_PAD_OTHER;
}

/* What plays a pad. mode: the Vibration row; force: HOS_RUMBLE_BACKEND; kind: vibration_pad_kind; wireless: 1 Bluetooth,
   0 a cable, -1 SDL cannot say; hd: the pad takes our HD rumble (a Switch pad, an SDL with the effect call); rh: RenHDrum
   is loaded; ds_usb: a DualSense's four-channel sound card is there; ds_hid: the pad's HID device can be written. */
static inline int vibration_pick(int mode, int force, int kind, int wireless, int hd, int rh, int ds_usb, int ds_hid) {
    if (mode == VIB_MODE_OFF) return VIB_NONE;
    if (mode == VIB_MODE_BASIC || force == VIB_FORCE_MOTORS) return VIB_MOTORS;
    const int tried = rh && force == VIB_FORCE_RENHDRUM;   /* (RenHDrum where testers asked for it) */
    switch (kind) {
    case VIB_PAD_DUALSENSE:
        if (rh && wireless != 1 && ds_usb) return VIB_RH_DS_USB;   /* (its sound card is there: it is on a cable) */
        if (tried && wireless != 0 && ds_hid) return VIB_RH_DS_BT;
        return VIB_MOTORS;
    case VIB_PAD_XBOX: return tried ? VIB_RH_XBOX : VIB_MOTORS;
    case VIB_PAD_SWITCH:
        if (tried) return VIB_RH_SWITCH;
        return hd ? VIB_HD : VIB_MOTORS;
    default: return VIB_MOTORS;
    }
}

/* What the Controls page says, as one number (HOS_OPT_VIBRATION_STATUS): the pad's kind times 16, plus one of these. */
enum {
    VIB_SAY_NO_PAD = 0,    /* no controller */
    VIB_SAY_OFF,           /* the Vibration row's Off */
    VIB_SAY_MOTORS,        /* its motors */
    VIB_SAY_HD,            /* its own HD rumble */
    VIB_SAY_FINE_USB,      /* fine haptics, on its cable (RenHDrum) */
    VIB_SAY_FINE_BT,       /* fine haptics over Bluetooth (RenHDrum) */
    VIB_SAY_FOUR,          /* four motors, the triggers' too (RenHDrum) */
    VIB_SAY_HD_RH,         /* HD rumble, RenHDrum's */
    VIB_SAY_WAITING,       /* a DualSense on a cable whose sound card is not there (yet): its motors meanwhile */
    VIB_SAY_NO_LIB,        /* a pad RenHDrum would play, without renhdrum.dll / .so: its motors */
    VIB_SAY_BLUETOOTH,     /* a DualSense over Bluetooth: its motors (fine haptics there are for testers yet) */
};
static inline int vibration_status(int kind, int say) { return kind * 16 + say; }
static inline int vibration_status_kind(int status) { return status < 0 ? 0 : status / 16; }
static inline int vibration_status_say(int status) { return status < 0 ? 0 : status % 16; }

/* What to say for a pad, from what plays it and why */
static inline int vibration_say(int backend, int kind, int wireless, int rh, int mode) {
    switch (backend) {
    case VIB_NONE: return mode == VIB_MODE_OFF ? VIB_SAY_OFF : VIB_SAY_MOTORS;
    case VIB_HD: return VIB_SAY_HD;
    case VIB_RH_DS_USB: return VIB_SAY_FINE_USB;
    case VIB_RH_DS_BT: return VIB_SAY_FINE_BT;
    case VIB_RH_XBOX: return VIB_SAY_FOUR;
    case VIB_RH_SWITCH: return VIB_SAY_HD_RH;
    default:
        if (mode == VIB_MODE_AUTO && kind == VIB_PAD_DUALSENSE) {
            if (wireless == 1) return VIB_SAY_BLUETOOTH;
            return rh ? VIB_SAY_WAITING : VIB_SAY_NO_LIB;
        }
        return VIB_SAY_MOTORS;
    }
}

/* The Vibration strength (percent, 100 the game's own) as a gain on samples or motor speeds: 0 to 2 */
static inline float vibration_gain(int percent) {
    if (percent <= 0) return 0.0f;
    return percent >= 200 ? 2.0f : (float)percent / 100.0f;
}
