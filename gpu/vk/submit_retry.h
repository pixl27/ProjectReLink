// (hookshot) A queue submit the driver refused for want of memory, tried again after memory is given back, not dropped.
// vkQueueSubmit2 that returns VK_ERROR_OUT_OF_DEVICE_MEMORY or VK_ERROR_OUT_OF_HOST_MEMORY leaves the command buffers, the fence
// and the semaphores as they were (the Vulkan spec's rule for a failed submit), so the same submit may be made again. Before
// this, the replay logged it and went on without it: the frame's work never ran, its timeline value came with the next
// submit's, and on a Legion Go (Radeon 780M, 8 GB UMA) every submit after a save load or the Deku Tree's entrance was refused
// that way, a black screen with the sound going on. RX_VK_SUBMIT_RETRY=0: dropped, as before.
// Standard library only: tests/test_submit_oom.py
#pragma once

namespace submit_retry {

constexpr int kTries = 3;   // (the submit and two more: each after give_back, the GPU idle and the freed memory returned)

/// VK_ERROR_OUT_OF_HOST_MEMORY (-1) or VK_ERROR_OUT_OF_DEVICE_MEMORY (-2): the refusals a submit may be made again after
inline bool refused_for_memory(int result) { return result == -1 || result == -2; }

/// submit() once, and while the driver refuses it for want of memory and on() says so, give_back(tries so far) and submit()
/// again, kTries in all. Returns the last result: VK_SUCCESS once one was taken, else what the last refusal said
template <class Submit, class GiveBack, class On>
auto submit(Submit&& submit, GiveBack&& give_back, On&& on) -> decltype(submit()) {
    auto result = submit();
    for (int tried = 1; tried < kTries && refused_for_memory((int)result) && on(); tried++) {
        give_back(tried);
        result = submit();
    }
    return result;
}

}  // namespace submit_retry
