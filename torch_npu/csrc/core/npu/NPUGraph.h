#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <exception>
#include <memory>
#include <vector>

struct SimplerNativeSubmission;

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "third_party/acl/inc/acl/super_kernel.h"
#include "torch_npu/csrc/core/npu/interface/SkInterface.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace at {
    struct Generator;
}

namespace at_npu {
    struct NPUGeneratorImpl;
    struct NPUGeneratorState;
}
namespace c10_npu {
// Standalone way to get a unique mempool id usable as a pool=... argument
// to CUDAGraph::capture_begin
TORCH_NPU_API MempoolId_t graph_pool_handle();

struct TORCH_NPU_API NPUTaskGroupHandle {
    aclrtTaskGrp task_group;
};

typedef TORCH_NPU_API void (*NPUCallbackFunc)(void *fnData);

TORCH_NPU_API void graph_task_group_begin(c10_npu::NPUStream stream);
TORCH_NPU_API NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream);
TORCH_NPU_API void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle);
TORCH_NPU_API void graph_task_update_end(c10_npu::NPUStream stream);
TORCH_NPU_API void super_kernel_scope_begin(const char* scope_name);
TORCH_NPU_API void super_kernel_scope_end(const char* scope_name);

TORCH_NPU_API void launch_callback(c10_npu::NPUStream stream, NPUCallbackFunc func, void *fnData);
TORCH_NPU_API void subscribe_report(uint64_t threadId, c10_npu::NPUStream stream);
TORCH_NPU_API void unsubscribe_report(uint64_t threadId, c10_npu::NPUStream stream);

struct TORCH_NPU_API NPUGraph {
    NPUGraph();
    ~NPUGraph() noexcept;
    // Queue frames may pin State; graph wrappers never share cleanup authority.
    NPUGraph(const NPUGraph&) = delete;
    NPUGraph& operator=(const NPUGraph&) = delete;

    static void inc_pending_event_queries();
    static void dec_pending_event_queries();
    static int num_pending_event_queries();

    void register_generator_state(c10::intrusive_ptr<at_npu::NPUGeneratorState> state);
    void register_generator_state(const at::Generator& generator);
    void capture_begin(
        MempoolId_t pool = {0, 0},
        aclmdlRICaptureMode capture_mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_GLOBAL,
		bool report_shape = true);
    void capture_end();
    void replay();
    void reset();
    MempoolId_t pool();
    void enable_debug_mode();
    void debug_dump(const std::string& debug_path);
    void super_kernel_optimize(const aclskOptions *options);

protected:
    static std::atomic<int> pending_event_queries;

private:
    struct State;
    struct NativeEntry;
    struct NativeQueueContext;
    std::shared_ptr<State> state_;
    SimplerNativeSubmission register_native_submission(
        const SimplerNativeSubmission& submission, std::vector<at::Tensor> tensors);
    void reset_impl();
    void close_native_admission();
    void cancel_native_pending();
    std::exception_ptr native_submission_error() const;
    void release_native_resources();
    friend struct NPUGraphNativeAccess;
    friend struct at_npu::NPUGeneratorState;

};

} // namespace c10_npu
