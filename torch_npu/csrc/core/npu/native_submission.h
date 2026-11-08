/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

// The named Python capsule is inspected only on the caller thread, with GIL.
// Retain a NativeSubmissionRef before handing it to a native queue callback.
inline constexpr const char *SIMPLER_NATIVE_SUBMISSION_CAPSULE = "simpler.NativeSubmission";

using SimplerNativePreflight = int (*)(void *, void *, char *, size_t) noexcept;

enum class NativeSubmissionDisposition : int { Pending, Submitting, Submitted, Cancelled, Retired };

struct NativeSubmissionCleanup {
    NativeSubmissionDisposition disposition{NativeSubmissionDisposition::Pending};
    bool carrier_retired{false};
};

struct SimplerNativeSubmission {
    void *context;
    void (*retain)(void *) noexcept;
    void (*release)(void *) noexcept;
    int (*submit)(void *, SimplerNativePreflight, void *, char *, size_t) noexcept;
    int (*retain_resources)(void *, int *, char *, size_t) noexcept;
    int (*cancel_pending)(void *, NativeSubmissionCleanup *, char *, size_t) noexcept;
    int (*release_resources)(void *, char *, size_t) noexcept;
};

// This value, including its final destructor, never accesses Python.
class NativeSubmissionRef {
public:
    explicit NativeSubmissionRef(const SimplerNativeSubmission &api) :
        api_(api) {
        if (!api_.context || !api_.retain || !api_.release || !api_.submit || !api_.retain_resources ||
            !api_.cancel_pending || !api_.release_resources)
            throw std::invalid_argument("invalid Simpler native submission capsule");
        api_.retain(api_.context);
    }
    NativeSubmissionRef(const NativeSubmissionRef &other) :
        api_(other.api_) {
        if (api_.context) api_.retain(api_.context);
    }
    NativeSubmissionRef(NativeSubmissionRef &&other) noexcept :
        api_(other.api_) {
        other.api_.context = nullptr;
    }
    NativeSubmissionRef &operator=(NativeSubmissionRef other) noexcept {
        std::swap(api_, other.api_);
        return *this;
    }
    ~NativeSubmissionRef() {
        if (api_.context) api_.release(api_.context);
    }
    void submit(SimplerNativePreflight preflight = nullptr, void *argument = nullptr) const {
        require_context();
        char message[1024]{};
        if (api_.submit(api_.context, preflight, argument, message, sizeof(message)) != 0)
            throw std::runtime_error(message);
    }
    void retain_resources(bool &registration_started) const {
        registration_started = false;
        require_context();
        int started = 0;
        char message[1024]{};
        const int rc = api_.retain_resources(api_.context, &started, message, sizeof(message));
        registration_started = started != 0;
        if (rc != 0) throw std::runtime_error(message);
    }
    void cancel_pending(NativeSubmissionCleanup &outcome) const {
        outcome = {};
        require_context();
        char message[1024]{};
        if (api_.cancel_pending(api_.context, &outcome, message, sizeof(message)) != 0)
            throw std::runtime_error(message);
    }
    void release_resources() const {
        require_context();
        char message[1024]{};
        if (api_.release_resources(api_.context, message, sizeof(message)) != 0) throw std::runtime_error(message);
    }

private:
    void require_context() const {
        if (!api_.context) throw std::logic_error("empty Simpler native submission reference");
    }
    SimplerNativeSubmission api_;
};
