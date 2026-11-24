# Red Tunic 4.2.1

- CPU ASTC decoding uses integer rounding in the LDR texel loop. The parallel decoder batches block rows across depth slices and waits once per image.
- The Windows Vega preset continues to use CPU ASTC decoding for material compatibility and keeps 4.2.0's texture retention and replay core reservation.

The new decoder matched the previous decoder byte for byte in 42 synthetic cases covering all 14 ASTC block sizes and three depths. A synthetic 2D texture decoded about 14% faster. Four Hyrule Field camera runs showed mixed frame times, so there is no confirmed gameplay speedup from these changes. See the [test records](docs/tested/) for the camera measurements.

The Windows x64 download contains the Red Tunic Builder and a SHA-256 checksum is provided separately. The Builder creates or updates the game locally from your own game files.
