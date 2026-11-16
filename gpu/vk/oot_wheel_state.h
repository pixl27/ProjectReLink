/* (relink) The item wheel's state, shared by the game side (host/oot_wheel.c fills it) and the Vulkan back end
   (gpu/vk/vk_wheel.cpp draws it), and the Gameplay tab's two rows for it. Kept out of vk_api.h, which is part of the
   program cache's stamp (nx2/build.py): a change there makes every player prepare their shaders again. The back end asks
   for the state through the host table's mic_name, with MKWVK_WHEEL_QUERY for the index (host/rx_backend.c), as the
   Insert menu asks whether a key is down through take_key_press. */
#pragma once

#include <stdint.h>

/* the Gameplay tab's Item wheel row: 1 the wheel, 0 the game's own item row; and its Cycle items row: 0 off, 1 Last used
   (a tap of R or L puts back the item on ZR before this one), 2 the arrows, bottles and spells, 3 and the bombs, 4 and the
   hookshot and hammer (a tap changes the ZR item within its group) */
enum { HOS_OPT_ITEM_WHEEL = 40, HOS_OPT_CYCLE_ITEMS };

/* the index of mic_name (MkwVkHost) that asks for the item wheel's state instead of a microphone's name: the answer is a
   const MkwVkWheel *, never null with the host of this build */
#define MKWVK_WHEEL_QUERY 0x57484c00

/* The R button's item row shown as a wheel: one page of MKWVK_WHEEL_SLOTS fixed slots (slot 0 at the top, the rest
   clockwise, 45 degrees apart; an empty name is an empty slot), the equipped item's slot and the stick's pick. seq changes
   whenever anything else does; the back end copies the whole and checks seq again (a copy made while the game side wrote
   is taken the next frame) */
#define MKWVK_WHEEL_MAX 32
#define MKWVK_WHEEL_SLOTS 8
typedef struct MkwVkWheel {
    volatile uint32_t seq;
    int32_t open;     /* 1 while the item row is open */
    int32_t count;    /* slots on this page (MKWVK_WHEEL_SLOTS), 0 with no item known yet */
    int32_t cursor;   /* the equipped item's slot on this page, -1 when it is on another page */
    int32_t pick;     /* the slot picked (the stick's, else the item chosen this opening), -1 none on this page */
    int32_t page, pages;    /* this page, of pages */
    uint8_t page_kind[16];  /* each page's kind: 0 the player's, 1 the items no page names, 2 a new page (while one is moved) */
    int32_t held;           /* the slot on this page of the item being moved (Y), -1 none or on another page */
    char held_name[48];     /* the item being moved, "" none */
    int32_t amount[MKWVK_WHEEL_MAX];   /* the item's count, -1 for an item without one */
    char name[MKWVK_WHEEL_MAX][48];    /* the game's name (Item_DekuNut) */
    char label[MKWVK_WHEEL_MAX][32];   /* what the player reads (Deku Nuts) */
} MkwVkWheel;
