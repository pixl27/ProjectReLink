"""Check the bounded capture reader and eligibility for exact input matching with synthetic data."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

RUN_ALL_ARGS = ['--cxx', '{cxx}']
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from mesh_capture_report import summarize

CPP = r'''
#include "mesh_capture.h"
#include <cassert>
#include <atomic>
#include <vector>

constexpr int VK_SHADER_STAGE_COMPUTE_BIT = 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 3;
namespace Shader::Backend::SPIRV { bool MeshFallbackCounters() { return false; } }
struct Program {
    struct Stage {
        int vk_stage = 1, nvn_stage = 7;
        bool lane_free = true;
        struct Info {
            struct Binding { uint32_t count; };
            std::vector<Binding> constant_buffer_descriptors;
            struct Storage { uint32_t count; bool is_written; uint32_t cbuf_index, cbuf_offset; };
            std::vector<Storage> storage_buffers_descriptors;
        } info;
    };
    bool uses_render_area = false;
    std::vector<Stage> stages;
};
struct DrawArgs { uint32_t first_task = 0; int groups[3]{1, 1, 1}; };
struct Buffer { int buffer = 1; size_t offset = 0, range = 4; };
using VkDescriptorBufferInfo = Buffer;
constexpr int zero_buffer = 3;
uint64_t storage_binding_align() { return 16; }
struct Writer {
    struct Write { int descriptorType = 2; uint32_t descriptorCount = 1; const Buffer* pBufferInfo; };
    std::vector<Write> writes;
};
Writer& dw() { static Writer writer; return writer; }
std::atomic<bool> par_active{false};
struct Stream { uint8_t* mapped; int stream = 1; size_t used = 4; };
Stream* cur = nullptr;

// GEOMETRY_INPUTS

int main() {
    const std::array<uint8_t, 4> a{1, 2, 3, 4}, b{1, 2, 3, 5};
    mesh_capture::Inputs first{}, changed{}, split{}, exact{};
    assert(first.append(a) && changed.append(b) && exact.append(a));
    assert(split.append(std::span(a).first(2)) && split.append(std::span(a).last(2)));
    assert(first.same(exact) && !first.same(changed) && !first.same(split));
    assert(!exact.append(std::span(first.bytes)));
    assert(mesh_capture::stored_vertices(30, 29) == 0);
    assert(mesh_capture::stored_vertices(100, 1000) == 64);
    assert(mesh_capture::stored_slots(42, 50) == 42);
    assert(mesh_capture::stored_slots(51, 50) == 0);
    Program program;
    program.stages.emplace_back().info.constant_buffer_descriptors.push_back({1});
    DrawArgs draw;
    Buffer buffer;
    dw().writes.push_back({2, 1, &buffer});
    std::array<uint8_t, 4> bytes{1, 2, 3, 4};
    Stream stream{bytes.data()};
    cur = &stream;
    mesh_capture::Inputs live{}, different{};
    GeometryStorage storage;
    assert(geometry_inputs(program, draw, 0, live, storage) && live.same(first));
    bytes.back() = 5;
    assert(geometry_inputs(program, draw, 0, different, storage) && different.same(changed));
    assert(!geometry_inputs(program, draw, mesh_capture::task, different, storage));
    assert(!geometry_inputs(program, draw, mesh_capture::indirect, different, storage));
    buffer.buffer = 2;
    assert(!geometry_inputs(program, draw, 0, different, storage));
    buffer.buffer = 1;
    buffer.offset = 1;
    assert(!geometry_inputs(program, draw, 0, different, storage));
    buffer.offset = 0;
    draw.groups[0] = 2;
    assert(!geometry_inputs(program, draw, 0, different, storage));
    draw.groups[0] = 1;
    const auto storage_flag = mesh_capture::storage;
    auto& descriptors = program.stages.front().info.storage_buffers_descriptors;
    descriptors.push_back({2, false, 1, 0});
    std::array<Buffer, 2> ranges{{{2, 4, 4}, {2, 32, 8}}};
    assert(!geometry_inputs(program, draw, storage_flag, different, storage));
    dw().writes.push_back({3, 2, ranges.data()});
    auto eligible = [&](mesh_capture::Inputs& inputs, uint32_t flags = mesh_capture::storage) {
        inputs.size = 0;
        storage = {};
        return geometry_inputs(program, draw, flags, inputs, storage);
    };
    assert(eligible(live));
    assert(storage.count == 2 && storage.used == 24);
    assert(storage.ranges[0].original.offset == 4 && storage.ranges[1].offset == 16);
    ranges[1].range = 12;
    assert(eligible(different) && !live.same(different));
    ranges[1].offset = 33;
    assert(!eligible(different));
    ranges[1].offset = 32;
    ranges[1].buffer = zero_buffer;
    assert(!eligible(different));
    ranges[1].buffer = 2;
    ranges[1].range = GeometryStorage::capacity;
    assert(!eligible(different));
    ranges[1].range = 8;
    descriptors[0].is_written = true;
    assert(!eligible(different));
    descriptors[0].is_written = false;
    assert(!eligible(different, storage_flag | mesh_capture::global));
    par_active = true;
    assert(!geometry_inputs(program, draw, 0, different, storage));
}
'''


def row(frame, value):
    return struct.pack('<4Q17I', frame, 1, 2, frame, 1 | 64 | 4, 2, 2, 1,
                       2, 1, 0, 0, 2, 1, 1, 1, value, 10, 30, 40, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='clang++')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mesh-capture-') as directory:
        temp = Path(directory)
        fixture = temp / 'fixture.rtm'
        data = b'RTMESH01' + row(1, 10) + row(2, 20)
        fixture.write_bytes(data)
        report = summarize(fixture)
        assert report['records'] == report['complete'] == report['task_lists'] == 2
        assert report['changing_variants'] == 1 and report['vertices'] == 4 and report['task_slots'] == 2
        fixture.write_bytes(data[:-1])
        try:
            summarize(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError('a truncated capture was accepted')
        source, binary = temp / 'check.cpp', temp / 'check.exe'
        renderer = (ROOT / 'gpu' / 'vk' / 'vk_draw.cpp').read_text(encoding='utf-8')
        storage = renderer[renderer.index('struct GeometryStorage {'):renderer.index('struct GeometryStorageGpu {')]
        inputs = storage + renderer[renderer.index('bool geometry_inputs('):renderer.index('struct RecentGeometry {')]
        source.write_text(CPP.replace('// GEOMETRY_INPUTS', inputs), encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++20', '-I', str(ROOT / 'gpu' / 'vk'), str(source), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
