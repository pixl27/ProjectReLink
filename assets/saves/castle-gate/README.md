castle-gate: Hyrule Field at Castle Town's gate, child Link on the drawbridge's town end facing the Field, by day

Made fresh for testing from slot_01 of a new profile; scrubbed with hookshot_saves.py scrub.

How it was made (2026-11-19, CachyOS Linux, RTX 3090, NVIDIA 620.70.06; Hookshot v3.0.0-rc.7-111-gda91e7702, main da91e770): a new
profile with only `kokiri-start` in it (`hookshot_ctl.py launch <game> --profile test --repo-save kokiri-start
--headless`, the options file empty, so every setting at its default and the Autosave row On), Continue on the title,
then through the control bridge `teleport Market_Child` (Link arrives in the Market's gate, facing the town; the
teleport's arrival writes no save) and the left stick held back for 4 seconds, which walks him out of the gate into
Hyrule Field. The game's own transition autosave, written as he arrived (scene HyruleField, map
HyruleField/HyruleField_Child, 3 hearts, the world clock by day), is this save. Nothing else was done in the game.
The story is a new game's, as in `kokiri-start`: no sword, no shield, the Kokiri Emerald not yet won.
