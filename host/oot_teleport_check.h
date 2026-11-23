/* (hookshot) What the debug teleport's follow-up (host/hos_main.c oot_teleport_followup, 1.5 s after a request) does with
   what it reads back. A request is sometimes answered 0 and then dropped by the game: nothing is reserved
   (Sequence_ReservedNextBanc ""), no event plays and Link is still where he was, so the map never changes and the control
   bridge's `teleport --arrive` times out. In scripted runs on an RTX 3080 that was 6 of 159 requests, 4 of them the first
   one a few frames after a save had loaded. Such a request is sent once more; one the game answered other than 0, one that is under way (a map reserved, an
   event playing, the map loading) or that has arrived is left alone, and a resent one is not sent a third time. */
#pragma once
#include <string.h>

/* 1: send the request again. reserved and current: the game's Sequence_ReservedNextBanc and Sequence_CurrentBanc; wanted:
   the map the request was for (NULL when none was sent); answer: the event manager's answer to it (every drop seen was
   answered 0; one it took may only be slow to reserve); resent: whether this request was already sent again */
static inline int oot_teleport_dropped(const char *reserved, const char *current, const char *wanted, int event_playing,
                                       int loading, unsigned long long answer, int resent) {
    if (!wanted || !*wanted || answer || resent || event_playing || loading) return 0;
    if (reserved && *reserved) return 0;              /* (the map change is reserved: under way) */
    return !current || strcmp(current, wanted) != 0;  /* (still elsewhere: dropped) */
}
