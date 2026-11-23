market-entrance: Castle Town Market, child Link in the town's gate facing the Market, by day

Made fresh for the Test Farm from slot_02 of a new profile; scrubbed with hookshot_saves.py scrub.

How it was made (2026-11-19, CachyOS Linux, RTX 3090, NVIDIA 620.70.06; Hookshot v3.0.0-rc.7-111-gda91e7702, main da91e770): a new
profile with only `kokiri-start` in it (`hookshot_ctl.py launch <game> --profile test --repo-save kokiri-start
--headless`, the options file empty, so every setting at its default and the Autosave row On), Continue on the title,
then through the control bridge `teleport Market_Child` (the teleport's arrival writes no save), the left stick held
back for 4 seconds (out of the gate into Hyrule Field: an autosave, slot_01, the `castle-gate` save), then back for 2
seconds (through the gate into the Market again). The Market's first-visit scene played by itself, and the game's own
autosave written after it (scene Market, map Market/Market_Child, 3 hearts, the world clock by day) is this save:
loading it puts Link in the gate with the Market ahead and its first-visit scene already seen. Nothing else was done in
the game. The story is a new game's, as in `kokiri-start`: no sword, no shield, the Kokiri Emerald not yet won.
