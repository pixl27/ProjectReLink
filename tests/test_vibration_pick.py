"""host/vibration.h: what plays each controller's vibration, and what the Controls page says about it. Compiled and run.

The rule, case by case: Off plays nothing and Basic the motors, on every pad. Auto plays a DualSense on a cable through
RenHDrum when its sound card is there, and otherwise its motors, saying why (no sound card yet, no RenHDrum, Bluetooth);
a DualShock 4 is not a DualSense (Sony's, but two channels and no voice coils); a Switch pad plays our HD rumble when it
takes it. RenHDrum for an Xbox pad, a DualSense over Bluetooth and a Switch pad only with HOS_RUMBLE_BACKEND=renhdrum, for
testers, and never without RenHDrum loaded; =motors plays the motors everywhere, =hd uses no RenHDrum. SDL2's and SDL3's
type numbers are told apart, and the status packs the pad's kind and what it says into one number and back.

    python -I -B tests/test_vibration_pick.py [--cxx g++]
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]

TEST = r"""
#include <stdio.h>
#include "vibration.h"
static int failed;
#define CHECK(x) do { if (!(x)) printf("FAIL %d: %s\n", __LINE__, #x), failed = 1; } while (0)
/* (pick for one pad: mode, force, kind, wireless, hd, rh, ds_usb, ds_hid) */
#define PICK vibration_pick
int main(void) {
    enum { A = VIB_MODE_AUTO, B = VIB_MODE_BASIC, O = VIB_MODE_OFF, FA = VIB_FORCE_AUTO, FR = VIB_FORCE_RENHDRUM };
    /* the switch */
    CHECK(vibration_force(NULL) == FA && vibration_force("") == FA && vibration_force("auto") == FA);
    CHECK(vibration_force("motors") == VIB_FORCE_MOTORS && vibration_force("hd") == VIB_FORCE_HD && vibration_force("renhdrum") == FR);
    CHECK(vibration_force("RenHDrum") == FA);   /* (anything else: Auto) */
    /* the pads, from SDL3's and SDL2's numbers */
    CHECK(vibration_pad_kind(3, 6, "x") == VIB_PAD_DUALSENSE && vibration_pad_kind(2, 7, "x") == VIB_PAD_DUALSENSE);
    CHECK(vibration_pad_kind(3, 3, "x") == VIB_PAD_XBOX && vibration_pad_kind(2, 2, "x") == VIB_PAD_XBOX);
    CHECK(vibration_pad_kind(3, 2, "Xbox 360 Controller") == VIB_PAD_OTHER);   /* (no trigger motors) */
    CHECK(vibration_pad_kind(3, 7, "x") == VIB_PAD_SWITCH && vibration_pad_kind(3, 10, "x") == VIB_PAD_SWITCH);
    CHECK(vibration_pad_kind(2, 5, "x") == VIB_PAD_SWITCH && vibration_pad_kind(2, 13, "x") == VIB_PAD_SWITCH);
    CHECK(vibration_pad_kind(3, 5, "PS4 Controller") == VIB_PAD_OTHER && vibration_pad_kind(2, 4, "PS4 Controller") == VIB_PAD_OTHER);
    CHECK(vibration_pad_kind(3, 7, "x") != vibration_pad_kind(2, 7, "x"));   /* (7: SDL3's Switch Pro, SDL2's PS5) */
    CHECK(vibration_pad_kind(0, 0, "DualSense Edge Wireless Controller") == VIB_PAD_DUALSENSE);   /* (an SDL with no type) */
    CHECK(vibration_pad_kind(0, 0, "Generic pad") == VIB_PAD_OTHER && vibration_pad_kind(0, 0, NULL) == VIB_PAD_OTHER);
    /* Off and Basic, on every pad */
    for (int k = VIB_PAD_OTHER; k <= VIB_PAD_SWITCH; k++) {
        CHECK(PICK(O, FR, k, 0, 1, 1, 1, 1) == VIB_NONE);
        CHECK(PICK(B, FR, k, 0, 1, 1, 1, 1) == VIB_MOTORS);
        CHECK(PICK(A, VIB_FORCE_MOTORS, k, 0, 1, 1, 1, 1) == VIB_MOTORS);
    }
    /* a DualSense, Auto */
    CHECK(PICK(A, FA, VIB_PAD_DUALSENSE, 0, 0, 1, 1, 1) == VIB_RH_DS_USB);
    CHECK(PICK(A, FA, VIB_PAD_DUALSENSE, -1, 0, 1, 1, 1) == VIB_RH_DS_USB);   /* (SDL2: placed by its sound card) */
    CHECK(PICK(A, FA, VIB_PAD_DUALSENSE, 0, 0, 1, 0, 1) == VIB_MOTORS);       /* (no sound card yet) */
    CHECK(PICK(A, FA, VIB_PAD_DUALSENSE, 0, 0, 0, 1, 1) == VIB_MOTORS);       /* (no RenHDrum) */
    CHECK(PICK(A, FA, VIB_PAD_DUALSENSE, 1, 0, 1, 1, 1) == VIB_MOTORS);       /* (Bluetooth: for testers yet) */
    CHECK(PICK(A, FR, VIB_PAD_DUALSENSE, 1, 0, 1, 0, 1) == VIB_RH_DS_BT);
    CHECK(PICK(A, FR, VIB_PAD_DUALSENSE, 1, 0, 1, 0, 0) == VIB_MOTORS);       /* (no HID device to write) */
    CHECK(PICK(A, FR, VIB_PAD_DUALSENSE, 0, 0, 1, 0, 1) == VIB_MOTORS);       /* (on a cable: its sound card, not HID) */
    CHECK(PICK(A, FR, VIB_PAD_DUALSENSE, 1, 0, 0, 0, 1) == VIB_MOTORS);       /* (no RenHDrum) */
    /* an Xbox pad, a Switch pad, another pad */
    CHECK(PICK(A, FA, VIB_PAD_XBOX, 0, 0, 1, 0, 0) == VIB_MOTORS);
    CHECK(PICK(A, FR, VIB_PAD_XBOX, 1, 0, 1, 0, 0) == VIB_RH_XBOX);
    CHECK(PICK(A, FR, VIB_PAD_XBOX, 1, 0, 0, 0, 0) == VIB_MOTORS);
    CHECK(PICK(A, FA, VIB_PAD_SWITCH, 0, 1, 1, 0, 0) == VIB_HD);
    CHECK(PICK(A, FA, VIB_PAD_SWITCH, 0, 0, 1, 0, 0) == VIB_MOTORS);          /* (SDL refused HD rumble) */
    CHECK(PICK(A, VIB_FORCE_HD, VIB_PAD_SWITCH, 0, 1, 1, 0, 0) == VIB_HD);
    CHECK(PICK(A, FR, VIB_PAD_SWITCH, 1, 1, 1, 0, 0) == VIB_RH_SWITCH);
    CHECK(PICK(A, FR, VIB_PAD_OTHER, 0, 0, 1, 1, 1) == VIB_MOTORS);
    /* what the page says */
    CHECK(vibration_say(VIB_NONE, VIB_PAD_XBOX, 0, 1, O) == VIB_SAY_OFF);
    CHECK(vibration_say(VIB_RH_DS_USB, VIB_PAD_DUALSENSE, 0, 1, A) == VIB_SAY_FINE_USB);
    CHECK(vibration_say(VIB_MOTORS, VIB_PAD_DUALSENSE, 0, 1, A) == VIB_SAY_WAITING);
    CHECK(vibration_say(VIB_MOTORS, VIB_PAD_DUALSENSE, -1, 0, A) == VIB_SAY_NO_LIB);
    CHECK(vibration_say(VIB_MOTORS, VIB_PAD_DUALSENSE, 1, 1, A) == VIB_SAY_BLUETOOTH);
    CHECK(vibration_say(VIB_MOTORS, VIB_PAD_DUALSENSE, 0, 1, B) == VIB_SAY_MOTORS);   /* (Basic asked for the motors) */
    CHECK(vibration_say(VIB_HD, VIB_PAD_SWITCH, 0, 0, A) == VIB_SAY_HD);
    CHECK(vibration_say(VIB_RH_XBOX, VIB_PAD_XBOX, 0, 1, A) == VIB_SAY_FOUR);
    CHECK(vibration_say(VIB_MOTORS, VIB_PAD_OTHER, 0, 0, A) == VIB_SAY_MOTORS);
    for (int k = VIB_PAD_OTHER; k <= VIB_PAD_SWITCH; k++)
        for (int s = VIB_SAY_NO_PAD; s <= VIB_SAY_BLUETOOTH; s++) {
            const int st = vibration_status(k, s);
            CHECK(vibration_status_kind(st) == k && vibration_status_say(st) == s);
        }
    CHECK(VIB_SAY_BLUETOOTH < 16);   /* (the say fits beside the kind) */
    /* the strength as a gain */
    CHECK(vibration_gain(0) == 0.0f && vibration_gain(-5) == 0.0f && vibration_gain(100) == 1.0f);
    CHECK(vibration_gain(25) == 0.25f && vibration_gain(200) == 2.0f && vibration_gain(999) == 2.0f);
    if (!failed) printf("vibration pick: ok\n");
    return failed;
}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as d:
        c = Path(d) / "vibration_pick.c"
        c.write_text(TEST, encoding="utf-8")
        exe = Path(d) / ("vibration_pick.exe" if os.name == "nt" else "vibration_pick")
        subprocess.run([a.cxx, "-I", str(ROOT / "host"), "-x", "c", "-std=c11", "-Wall", "-Wextra", "-Werror", str(c), "-o",
                        str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
