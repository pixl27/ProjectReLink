"""Summarize a local RTMESH01 fallback capture without exporting the game's data."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import struct

MAGIC = b'RTMESH01'
RECORD = struct.Struct('<4Q12I')
MAX_BYTES = 8 + 48 * (RECORD.size + 4 * (128 * 64 + 64))
FLAGS = {1: 'task', 2: 'indirect', 4: 'storage buffers', 8: 'images or textures',
         16: 'global memory', 32: 'memory writes', 128: 'eligible for exact reuse'}


def summarize(path):
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or data[:8] != MAGIC:
        raise ValueError('invalid mesh capture')
    source_flags, samples, visibility_counts = Counter(), defaultdict(list), defaultdict(set)
    records = first = last = nonempty = complete = task_lists = 0
    vertices = task_slots = 0
    at = len(MAGIC)
    while at < len(data):
        if records == 48 or len(data) - at < RECORD.size:
            raise ValueError('invalid or truncated mesh capture record')
        frame, key, variant, _submit, flags, words, kept_vertices, kept_slots, *header = RECORD.unpack_from(data, at)
        at += RECORD.size
        if not (1 <= words <= 128 and kept_vertices == min(64, header[0]) and
                kept_slots == (min(64, header[5]) if flags & 1 else 0)):
            raise ValueError('invalid captured mesh counts')
        end = at + 4 * (words * kept_vertices + kept_slots)
        if end > len(data):
            raise ValueError('truncated captured mesh bytes')
        if not records:
            first = frame
        last = frame
        records += 1
        vertices += kept_vertices
        task_slots += kept_slots
        nonempty += kept_vertices != 0
        task_lists += bool(flags & 1)
        full = header[0] == kept_vertices and (not flags & 1 or header[5] == kept_slots)
        complete += full
        for flag, name in FLAGS.items():
            if flags & flag:
                source_flags[name] += 1
        if flags & 1:
            visibility_counts[key, variant].add(header[5])
        if kept_vertices or kept_slots:
            sample = (flags & 64, words, header[0], header[5], data[at:end])
            samples[key, variant].append(sample)
        at = end
    repeated = sum(len(items) > 1 for items in samples.values())
    changing = sum(len(set(items)) > 1 for items in samples.values())
    return {'records': records, 'first': first, 'last': last, 'nonempty': nonempty, 'complete': complete,
            'task_lists': task_lists, 'vertices': vertices, 'task_slots': task_slots, 'variant_count': len(samples),
            'repeated_variants': repeated, 'changing_variants': changing,
            'changing_visibility_counts': sum(len(counts) > 1 for counts in visibility_counts.values()),
            'dependencies': dict(source_flags)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.capture)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for name, value in report.items():
        print(f'{name}: {value}')


if __name__ == '__main__':
    main()
