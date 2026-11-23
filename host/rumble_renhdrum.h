/* (hookshot) RenHDrum's ways of playing the game's vibration (third_party/renhdrum, MIT OR Apache-2.0), for the pads
   host/vibration.h gives them: a DualSense's voice coils over its cable (its own four-channel sound card) or over
   Bluetooth (HID report 0x32), an Xbox One or Series pad's four motors, a Switch pad's HD rumble packets.

   RenHDrum is loaded at run time from beside the game (renhdrum.dll, renhdrum.so on Linux: scripts/renhdrum.py builds
   them, the Builder carries them); without it every pad keeps what it had, and rh_load says why.

   Threads: host/pad_sdl.c's (the pads' poll) opens, closes and ticks everything here; the game's (host/nn_hid.c)
   pushes the values; the sound card's callback and the Bluetooth sender read the DSPs. One lock holds them all, never
   across a call that waits for one of those threads (a stream's close, the sender's join). */
#pragma once
#include <stdint.h>

int rh_load(void);                 /* 1 when RenHDrum is loaded (tried once) */
const char *rh_why(void);          /* why it is not, when it is not */

void rh_push(int side, const float value[4]);    /* (the game's thread) {amp_low, freq_low, amp_high, freq_high} */
void rh_tick(int64_t now, int testing, int percent);   /* each poll: a test's values, a hand gone quiet, the strength */

/* A DualSense on a cable: its four-channel sound card, found by name (rate-limited by the caller), opened for one pad */
int rh_usb_find(void);             /* 1 when it is there */
int rh_usb_open(void *pad);        /* 1 when it plays */
int rh_usb_alive(void);
void rh_usb_close(void);
void *rh_usb_owner(void);

/* A DualSense over Bluetooth: report 0x32 every 10.7 ms through a second handle on its HID device */
int rh_bt_open(void *pad, const char *hid_path);
int rh_bt_alive(void);
void rh_bt_close(void);
void *rh_bt_owner(void);

/* An Xbox pad's four motors, a Switch pad's packets: what to send now, when there is something (1) */
int rh_pad_open(void *pad, int backend, int bluetooth, int pro);   /* backend: VIB_RH_XBOX or VIB_RH_SWITCH */
void rh_pad_close(void *pad);
int rh_xbox_poll(void *pad, int64_t now, uint16_t motors[4]);      /* left grip, right grip, left trigger, right */
int rh_switch_poll(void *pad, int64_t now, uint8_t packet[10]);

void rh_shutdown(void);            /* the game is over: everything closed, nothing opens again */
