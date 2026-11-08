#include "torch_npu/csrc/core/npu/native_submission.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Producer {
    unsigned references = 1;
    int started = 1;
    int acquire_rc = 0;
    int cancel_rc = 0;
    int release_failures = 0;
    int release_calls = 0;
    int submit_calls = 0;
    void* stream = this;
    static Producer& get(void* context) { return *static_cast<Producer*>(context); }
    static void retain(void* context) noexcept { ++get(context).references; }
    static void release(void* context) noexcept { --get(context).references; }
    static int submit(void* context, SimplerNativePreflight preflight, void* arg, char* message, size_t capacity) noexcept {
        auto& p = get(context);
        ++p.submit_calls;
        return preflight ? preflight(arg, p.stream, message, capacity) : 0;
    }
    static int acquire(void* context, int* started, char* message, size_t capacity) noexcept {
        auto& p = get(context);
        *started = p.started;
        if (p.acquire_rc && message && capacity) std::snprintf(message, capacity, "acquire error");
        return p.acquire_rc;
    }
    static int cancel(void* context, NativeSubmissionCleanup* out, char* message, size_t capacity) noexcept {
        auto& p = get(context);
        *out = {NativeSubmissionDisposition::Retired, true};
        if (p.cancel_rc && message && capacity) std::snprintf(message, capacity, "cancel error");
        return p.cancel_rc;
    }
    static int release_resources(void* context, char* message, size_t capacity) noexcept {
        auto& p = get(context);
        ++p.release_calls;
        if (p.release_failures > 0) {
            --p.release_failures;
            if (message && capacity) std::snprintf(message, capacity, "release error");
            return -1;
        }
        return 0;
    }
    SimplerNativeSubmission api() { return {this, retain, release, submit, acquire, cancel, release_resources}; }
};
template<class Function> void must_throw(Function&& function, const char* text) {
    try { function(); } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(text) != std::string::npos, "original callback error lost");
        return;
    }
    throw std::runtime_error("callback failure was reported as success");
}
}

int main() {
    try {
        Producer producer;
        {
            NativeSubmissionRef original(producer.api());
            require(producer.references == 2, "reference was not retained");
            {
                NativeSubmissionRef copied(original);
                NativeSubmissionRef moved(std::move(copied));
                require(producer.references == 3, "copy/move ownership count changed");
            }
            require(producer.references == 2, "temporary owner did not release");
            bool called = false;
            struct Argument { bool* called; void* expected; } argument{&called, producer.stream};
            original.submit([](void* raw, void* stream, char*, size_t) noexcept -> int {
                auto* value = static_cast<Argument*>(raw);
                *value->called = stream == value->expected;
                return 0;
            }, &argument);
            require(called && producer.submit_calls == 1, "ephemeral preflight/stream not forwarded");
            producer.acquire_rc = -73;
            bool started = false;
            must_throw([&] { original.retain_resources(started); }, "acquire error");
            require(started, "started output was lost on failed acquire");
            producer.started = 0;
            started = true;
            must_throw([&] { original.retain_resources(started); }, "acquire error");
            require(!started, "duplicate refused acquisition acquired old registration");
            producer.cancel_rc = -81;
            NativeSubmissionCleanup cleanup;
            must_throw([&] { original.cancel_pending(cleanup); }, "cancel error");
            require(cleanup.disposition == NativeSubmissionDisposition::Retired && cleanup.carrier_retired,
                    "cleanup facts were lost on historical failure");
            producer.release_failures = 1;
            must_throw([&] { original.release_resources(); }, "release error");
            require(producer.references == 2, "failed checked release dropped native holder");
            original.release_resources();
            require(producer.release_calls == 2, "explicit checked retry did not use original producer");
        }
        require(producer.references == 1, "last reference did not release exactly once");
        auto invalid = producer.api();
        invalid.cancel_pending = nullptr;
        bool refused = false;
        try { NativeSubmissionRef bad(invalid); } catch (const std::invalid_argument&) { refused = true; }
        require(refused && producer.references == 1, "invalid callback table changed ownership");
        std::puts("NATIVE_SUBMISSION_REF_PASS: production header ownership, outputs, preflight and checked retry");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
