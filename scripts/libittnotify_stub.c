/*
 * Minimal no-op ITT/JIT profiling shim for offline PyTorch runtime overlays.
 * It satisfies the three symbols libtorch_cpu.so needs when Intel's
 * profiling runtime is not available in the local environment.
 */

#include <stdint.h>

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int next_id = 1;
    return next_id++;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

int iJIT_NotifyEvent(int event_type, void *event_data) {
    (void)event_type;
    (void)event_data;
    return 0;
}
