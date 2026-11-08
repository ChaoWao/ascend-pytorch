#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/GetCANNInfo.h"
#include "third_party/acl/inc/acl/error_codes/rt_error_codes.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

#include "torch_npu/csrc/core/npu/native_submission.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <thread>
#include <vector>

#include <ATen/Functions.h>

namespace c10_npu {

static bool _npu_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

namespace {
void apply_cache_op_info(aclrtStream stream, bool enabled)
{
    if (!IsGteCANNVersion("8.5.0", "CANN")) {
		return;
	}
    aclrtStreamAttrValue val;
    val.cacheOpInfoSwitch = static_cast<uint32_t>(enabled ? 1u : 0u);
    int32_t ret = c10_npu::acl::AclrtSetStreamAttribute(stream, aclrtStreamAttr::ACL_STREAM_ATTR_CACHE_OP_IFNO,
        &val);
    if (ret == ACL_ERROR_RT_PARAM_INVALID) {
        ASCEND_LOGW("Report shape function is disabled due to incompatible CANN version.");
    } else {
        TORCH_CHECK(ret == ACL_RT_SUCCESS, "AclrtSetStreamAttribute failed with error code: ", ret);
    }
}
}

MempoolId_t graph_pool_handle()
{
    // Sets just the second value, to distinguish it from MempoolId_ts created from
    // aclmdlRICaptureGetInfo id_s in capture_begin.
    auto new_pool = c10_npu::MemPool();
    return new_pool.id();
}

void graph_task_group_begin(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpBegin(stream));
}

NPUTaskGroupHandle graph_task_group_end(c10_npu::NPUStream stream)
{
    aclrtTaskGrp group;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskGrpEnd(stream, &group));
    NPUTaskGroupHandle handle;
    handle.task_group = group;
    return handle;
}

void graph_task_update_begin(c10_npu::NPUStream stream, NPUTaskGroupHandle handle)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group));
}

void graph_task_update_end(c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureTaskUpdateEnd(stream));
}

void super_kernel_scope_begin(const char* scope_name)
{
    auto stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(c10_npu::skapi::AclskScopeBegin(scope_name, stream));
}

void super_kernel_scope_end(const char* scope_name)
{
    auto stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(c10_npu::skapi::AclskScopeEnd(scope_name, stream));
}

void launch_callback(c10_npu::NPUStream stream, NPUCallbackFunc func, void *fnData)
{
    aclrtCallbackBlockType type = aclrtCallbackBlockType::ACL_CALLBACK_BLOCK;
    NPU_CHECK_ERROR(c10_npu::acl::AclrtLaunchCallback(func, fnData, type, stream));
}

void subscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclrtSubscribeReport(threadId, stream));
}

void unsubscribe_report(uint64_t threadId, c10_npu::NPUStream stream)
{
    NPU_CHECK_ERROR(c10_npu::acl::AclrtUnSubscribeReport(threadId, stream));
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

struct NPUGraph::NativeEntry {
    NativeEntry(const SimplerNativeSubmission& api, std::vector<at::Tensor> owners)
        : submission(api), tensors(std::move(owners)) {}
    NativeSubmissionRef submission;
    std::vector<at::Tensor> tensors;
    bool registration_started = false;
    bool callback_entered = false;
    std::exception_ptr submit_error;
};

struct NPUGraph::State {
    // NPUStreams may not be default-constructed. This allocation is stable even
    // when a stack-owned NPUGraph wrapper is destroyed after failed cleanup.
    State() : capture_stream_(c10_npu::getCurrentNPUStream()) {}

    // One resource owner. Queue contexts pin this same allocation, not a wrapper.
    std::mutex native_mutex_;
    bool abandoned_ = false;
    bool native_admission_closed_ = true;
    bool native_execution_closed_ = true;
    bool native_lifecycle_running_ = false;
    unsigned native_registrations_ = 0;
    unsigned native_callbacks_ = 0;
    aclrtStream native_stream_ = nullptr;
    std::vector<std::shared_ptr<NativeEntry>> native_entries_;
    std::exception_ptr native_cleanup_error_;
    struct Lifecycle {
        State& state;
        explicit Lifecycle(State& value) : state(value) {
            std::lock_guard<std::mutex> lock(state.native_mutex_);
            TORCH_CHECK(!state.abandoned_ && !state.native_lifecycle_running_,
                        "Graph lifecycle operation is already in progress.");
            state.native_lifecycle_running_ = true;
        }
        ~Lifecycle() {
            std::lock_guard<std::mutex> lock(state.native_mutex_);
            state.native_lifecycle_running_ = false;
        }
    };
    aclmdlRI model_ri_ = nullptr;
    bool has_graph_exec_ = false;
    bool capture_pending_ = false;
    bool capture_begun_ = false;
    bool model_owned_ = false;
    bool pool_held_ = false;
    bool pool_capture_active_ = false;
    CaptureId_t capture_id_ = -1;
    MempoolId_t mempool_id_{0, 0};
    NPUStream capture_stream_;
    int capture_dev_ = -1;
    ska::flat_hash_map<c10::intrusive_ptr<at_npu::NPUGeneratorState>, uint64_t> captured_generator_states_;
    std::vector<at_npu::NPUGeneratorState*> prepared_generator_states_;
};


// Only this queue frame pins the State. State -> Entry has no back-reference.
struct NPUGraph::NativeQueueContext {
    std::atomic<unsigned> references{1};
    std::shared_ptr<State> state;
    std::shared_ptr<NativeEntry> entry;
    const aclmdlRI model;
    const aclrtStream stream;

    NativeQueueContext(std::shared_ptr<State> owner, std::shared_ptr<NativeEntry> call,
                       aclmdlRI model_identity, aclrtStream caller_stream)
        : state(std::move(owner)), entry(std::move(call)), model(model_identity), stream(caller_stream) {}

    template<class Function>
    static int checked(char* message, size_t capacity, Function&& function) noexcept
    {
        try {
            function();
            return 0;
        } catch (const std::exception& error) {
            if (capacity) std::snprintf(message, capacity, "%s", error.what());
        } catch (...) {
            if (capacity) std::snprintf(message, capacity, "native graph submission failed");
        }
        return -1;
    }

    struct Invocation {
        NativeQueueContext* context;
        SimplerNativePreflight outer;
        void* argument;
    };

    static int preflight(void* opaque, void* caller_stream, char* message, size_t capacity) noexcept
    {
        auto& invocation = *static_cast<Invocation*>(opaque);
        auto& context = *invocation.context;
        const int rc = checked(message, capacity, [&] {
            aclmdlRI expected;
            {
                std::lock_guard<std::mutex> lock(context.state->native_mutex_);
                TORCH_CHECK(!context.state->abandoned_ && context.state->capture_begun_ &&
                            !context.state->native_execution_closed_,
                            "The original graph is no longer accepting captured execution.");
                TORCH_CHECK(caller_stream == context.stream && context.state->native_stream_ == context.stream &&
                            context.state->model_ri_ == context.model,
                            "Native submission belongs to a different capture identity.");
                expected = context.model;
            }
            aclmdlRICaptureStatus status;
            aclmdlRI model = nullptr;
            NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(
                static_cast<aclrtStream>(caller_stream), &status, &model));
            TORCH_CHECK(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE &&
                        expected && model == expected, "Native submission is outside its original graph.");
        });
        if (rc != 0) return rc;
        return invocation.outer ? invocation.outer(invocation.argument, caller_stream, message, capacity) : 0;
    }

    static SimplerNativeSubmission api(NativeQueueContext* context)
    {
        return {
            context,
            [](void* opaque) noexcept { ++static_cast<NativeQueueContext*>(opaque)->references; },
            [](void* opaque) noexcept {
                auto* context = static_cast<NativeQueueContext*>(opaque);
                if (--context->references == 0) delete context;
            },
            [](void* opaque, SimplerNativePreflight outer, void* argument,
               char* message, size_t capacity) noexcept {
                auto& context = *static_cast<NativeQueueContext*>(opaque);
                return checked(message, capacity, [&] {
                    bool counted = false;
                    try {
                        {
                            std::lock_guard<std::mutex> lock(context.state->native_mutex_);
                            // Actual entry, not registration, acknowledges queue publication.
                            context.entry->callback_entered = true;
                            TORCH_CHECK(!context.state->abandoned_ && context.state->capture_begun_ &&
                                        !context.state->native_execution_closed_ &&
                                        context.state->model_ri_ == context.model &&
                                        context.state->native_stream_ == context.stream &&
                                        context.entry->registration_started &&
                                        std::find(context.state->native_entries_.begin(),
                                                  context.state->native_entries_.end(), context.entry) !=
                                            context.state->native_entries_.end(),
                                        "The original graph cannot submit this native entry.");
                            // closing only forbids NEW entries: End drains accepted callbacks.
                            ++context.state->native_callbacks_;
                            counted = true;
                        }
                        Invocation invocation{&context, outer, argument};
                        context.entry->submission.submit(&preflight, &invocation);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(context.state->native_mutex_);
                        if (!context.entry->submit_error) context.entry->submit_error = std::current_exception();
                        if (counted) --context.state->native_callbacks_;
                        throw;
                    }
                    std::lock_guard<std::mutex> lock(context.state->native_mutex_);
                    --context.state->native_callbacks_;
                });
            },
            [](void*, int* started, char* message, size_t capacity) noexcept {
                *started = 0;
                return checked(message, capacity, [] {
                    TORCH_CHECK(false, "A graph-owned entry cannot be registered again.");
                });
            },
            [](void* opaque, NativeSubmissionCleanup* outcome, char* message, size_t capacity) noexcept {
                return checked(message, capacity, [&] {
                    static_cast<NativeQueueContext*>(opaque)->entry->submission.cancel_pending(*outcome);
                });
            },
            [](void* opaque, char* message, size_t capacity) noexcept {
                auto& context = *static_cast<NativeQueueContext*>(opaque);
                return checked(message, capacity, [&] {
                    {
                        std::lock_guard<std::mutex> lock(context.state->native_mutex_);
                        TORCH_CHECK(!context.state->abandoned_ && !context.state->capture_pending_ &&
                                    !context.state->model_owned_ && !context.state->native_registrations_ &&
                                    !context.state->native_callbacks_ && !context.state->native_lifecycle_running_,
                                    "The original graph still owns native resources.");
                    }
                    context.entry->submission.release_resources();
                });
            }
        };
    }
};

SimplerNativeSubmission NPUGraph::register_native_submission(
    const SimplerNativeSubmission& submission, std::vector<at::Tensor> tensors)
{
    // Raw-stream conversion drains accepted Host callbacks, not device work.
    // No State or submission lock is held while the original queue makes progress.
    const aclrtStream raw_stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(raw_stream != nullptr, "Could not drain the capture queue before native registration.");
    // All allocation and native ref ownership precedes acquisition side effects.
    auto entry = std::make_shared<NativeEntry>(submission, std::move(tensors));
    std::unique_ptr<NativeQueueContext> context;
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        TORCH_CHECK(!state_->abandoned_ && state_->capture_begun_ && !state_->native_admission_closed_ &&
                    !state_->native_lifecycle_running_ && state_->model_ri_ && state_->native_stream_ &&
                    raw_stream == state_->native_stream_,
                    "The original graph is not accepting native entries.");
        context = std::make_unique<NativeQueueContext>(state_, entry, state_->model_ri_, state_->native_stream_);
        state_->native_entries_.push_back(entry);
        ++state_->native_registrations_;
    }
    bool started = false;
    try {
        entry->submission.retain_resources(started);
    } catch (...) {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        entry->registration_started = started;
        if (started) entry->submit_error = std::current_exception();
        --state_->native_registrations_;
        if (!started) {
            // Duplicate ownership belongs to the OLD entry, never this new one.
            auto& entries = state_->native_entries_;
            entries.erase(std::find(entries.begin(), entries.end(), entry));
        }
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        entry->registration_started = started;
        --state_->native_registrations_;
        if (!started) {
            auto& entries = state_->native_entries_;
            entries.erase(std::find(entries.begin(), entries.end(), entry));
        }
    }
    TORCH_CHECK(started, "Native acquisition succeeded without an ownership claim.");
    return NativeQueueContext::api(context.release());
}

void NPUGraph::close_native_admission()
{
    std::lock_guard<std::mutex> lock(state_->native_mutex_);
    state_->native_admission_closed_ = true;
    TORCH_CHECK(!state_->abandoned_ && !state_->native_registrations_,
                "Native entry registration is still in flight.");
}

void NPUGraph::cancel_native_pending()
{
    // Admission is closed. Never hold this mutex across native cleanup or queue drain.
    std::vector<std::shared_ptr<NativeEntry>> entries;
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        TORCH_CHECK(!state_->native_registrations_ && !state_->native_callbacks_,
                    "Native callbacks are still in flight.");
        state_->native_execution_closed_ = true;
        for (const auto& entry : state_->native_entries_) {
            if (!entry->callback_entered && !entry->submit_error) {
                // Registration is inside enqueue_native, so a missing callback
                // is a missing operation, never a successful partial graph.
                entry->submit_error = std::make_exception_ptr(std::runtime_error(
                    "Native submission was registered but not dispatched"));
            }
        }
        entries = state_->native_entries_;
    }
    for (const auto& entry : entries) {
        NativeSubmissionCleanup outcome;
        entry->submission.cancel_pending(outcome);
        TORCH_CHECK(outcome.carrier_retired, "Native submission cleanup is unresolved.");
    }
}

std::exception_ptr NPUGraph::native_submission_error() const
{
    std::lock_guard<std::mutex> lock(state_->native_mutex_);
    for (const auto& entry : state_->native_entries_) {
        if (entry->submit_error) return entry->submit_error;
    }
    return {};
}

void NPUGraph::release_native_resources()
{
    // Called only after checked Destroy (or a known no-model state).
    while (true) {
        std::shared_ptr<NativeEntry> entry;
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            if (state_->native_entries_.empty()) return;
            entry = state_->native_entries_.front();
        }
        entry->submission.release_resources();
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->native_entries_.erase(state_->native_entries_.begin());
        }
        // Tensor finalizers may enter the allocator or other locks.
        entry->tensors.clear();
    }
}

std::atomic<int> NPUGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void NPUGraph::inc_pending_event_queries()
{
    pending_event_queries++;
}

void NPUGraph::dec_pending_event_queries()
{
    TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
                          "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
    pending_event_queries--;
}

int NPUGraph::num_pending_event_queries()
{
    return pending_event_queries;
}

NPUGraph::NPUGraph() : state_(new State(), [](State* state) noexcept {
    // This final deleter performs no SDK calls or allocations. An unresolved
    // graph deliberately keeps the SAME allocation, including RNG/tensor roots.
    if (!state->abandoned_) delete state;
})
{
}

void NPUGraph::register_generator_state(c10::intrusive_ptr<at_npu::NPUGeneratorState> state)
{
    auto inserted = state_->captured_generator_states_.emplace(state, 0);
    if (inserted.second) {
        try {
            state->register_graph(this);
        } catch (...) {
            state_->captured_generator_states_.erase(inserted.first);
            throw;
        }
    }
}

void NPUGraph::register_generator_state(const at::Generator& generator)
{
    c10::intrusive_ptr<at_npu::NPUGeneratorImpl> npu_gen =
        c10::dynamic_intrusive_pointer_cast<at_npu::NPUGeneratorImpl>(generator.getIntrusivePtr());
    npu_gen->register_graph(this);
}

void NPUGraph::capture_begin(MempoolId_t pool, aclmdlRICaptureMode capture_mode, bool report_shape)
{
    State::Lifecycle lifecycle(*state_);
    static const auto _task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    TORCH_CHECK(_task_queue_enable != 2,
        "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please "
        "export TASK_QUEUE_ENABLE=1/0.",
        PTA_ERROR(ErrCode::NOT_SUPPORT));

    TORCH_CHECK(!state_->has_graph_exec_ && !state_->model_owned_ && !state_->capture_pending_ &&
                !state_->pool_held_ && state_->prepared_generator_states_.empty(),
                "This NPUGraph instance still owns capture resources. "
                "To capture a new graph, create a new instance.");

    auto stream = c10_npu::getCurrentNPUStream();

    TORCH_CHECK(stream != c10_npu::getDefaultNPUStream(),
                "NPU graphs must be captured on a non-default stream. "
                "(However, after capture, it's ok to replay them on the "
                "default stream.)");

    apply_cache_op_info(stream, report_shape);

    // default generator is always registered
    auto* gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(c10::nullopt, at_npu::detail::getDefaultNPUGenerator());
    gen->register_graph(this);
    gen->set_secondary_stream_capture_state(false);

    try {
        state_->prepared_generator_states_.reserve(state_->captured_generator_states_.size());
        for (auto& [generator_state, wholegraph_increments] : state_->captured_generator_states_) {
            generator_state->capture_prologue();
            state_->prepared_generator_states_.push_back(generator_state.get());
        }

        state_->capture_stream_ = stream;
        state_->capture_dev_ = c10_npu::current_device();

        if (pool.first != 0 || pool.second != 0) {
            // Either value being nonzero means the user supplied a pool to share.
            // But only one should be nonzero.
            // If pool was created by another graph's capture_begin, first should be nonzero.
            // If pool was created by graph_pool_handle, second should be nonzero.
            TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
            state_->mempool_id_ = pool;
        } else {
            // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
            // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
            auto mempool = c10_npu::MemPool({}, false);
            state_->mempool_id_ = mempool.id();
            TORCH_INTERNAL_ASSERT(state_->mempool_id_.first > 0);
        }

        // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
        // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
        // due to the capture status being updated _after_ a capture had already started.
        c10_npu::NPUCachingAllocator::beginAllocateToPool(
            state_->capture_dev_, state_->mempool_id_, [state = state_.get()](aclrtStream stream) {
                aclmdlRICaptureStatus status;
                aclmdlRI model_ri;
                NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
                return status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE &&
                       model_ri == state->model_ri_;
            });

        state_->pool_held_ = true;
        state_->pool_capture_active_ = true;
    } catch (...) {
        auto primary = std::current_exception();
        // Keep this transaction's lifecycle claim throughout known pre-Begin
        // rollback; public reset() would try to acquire that same claim again.
        try {
            reset_impl();
        } catch (...) {
            // A completed acquisition remains recorded for explicit reset retry.
            TORCH_WARN("NPUGraph pre-capture cleanup failed; reset() must be retried.");
        }
        std::rethrow_exception(primary);
    }

    // At this point, any NCCL watchdogs should be aware that we are in capture mode
    // and therefore should not enqueue any additional work that could be event-queried.
    // We still must wait on any existing work that has not been cleaned up.
    while (num_pending_event_queries()) {
        TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }

    // cudaStreamCaptureModeGlobal is the most conservative option to
    // prevent potentially unsafe CUDA API calls during capture.
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        state_->capture_pending_ = true;
    }
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureBegin(state_->capture_stream_, capture_mode));
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        state_->capture_begun_ = true;
    }

    aclmdlRICaptureStatus status;
    aclmdlRI model_ri = nullptr;
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureGetInfo(stream, &status, &model_ri));
    TORCH_INTERNAL_ASSERT(status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && model_ri);
    {
        std::lock_guard<std::mutex> lock(state_->native_mutex_);
        state_->model_ri_ = model_ri;
        state_->native_stream_ = stream.stream(false);
        state_->native_admission_closed_ = false;
        state_->native_execution_closed_ = false;
    }
}

void NPUGraph::capture_end()
{
    State::Lifecycle lifecycle(*state_);
    try {
        close_native_admission();
        TORCH_CHECK(state_->capture_begun_,
                    "CaptureEnd requires this graph's successful Begin and no prior End attempt.");
        auto stream = c10_npu::getCurrentNPUStream();

        TORCH_CHECK(stream == state_->capture_stream_,
                    "Capture must end on the same stream it began on.");

        // Conversion drains accepted queue callbacks without holding the State lock.
        const aclrtStream raw_stream = stream;
        TORCH_CHECK(raw_stream != nullptr, "Could not drain the capture queue.");
        cancel_native_pending();
        apply_cache_op_info(raw_stream, false);

        aclmdlRI model_ri = nullptr;
        // A failed End does not prove that another End or Destroy is safe.
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->capture_begun_ = false;
        }
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRICaptureEnd(raw_stream, &model_ri));
        TORCH_CHECK(model_ri != nullptr, "CaptureEnd succeeded without a valid model.");

        // End's successful output is owned even if subsequent bookkeeping throws.
        aclmdlRI expected_model;
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            expected_model = state_->model_ri_;
            state_->model_ri_ = model_ri;
            state_->model_owned_ = true;
            state_->capture_pending_ = false;
        }
        TORCH_CHECK(expected_model == nullptr || model_ri == expected_model,
                    "Invalid end capture model id: ", model_ri);

        c10_npu::NPUCachingAllocator::endAllocateToPool(state_->capture_dev_, state_->mempool_id_);
        state_->pool_capture_active_ = false;

        // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
        // between replays.
        // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
        // is cudaMallocAsync.
        // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
        // the graph's internal bookkeeping requires that we instantiate with
        // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
        // cudaGraphLaunch
        // cudaGraphInstantiateWithFlags
        for (auto& [generator_state, wholegraph_increments] : state_->captured_generator_states_) {
            wholegraph_increments = generator_state->capture_epilogue();
        }
        state_->prepared_generator_states_.clear();
    } catch (...) {
        const auto secondary = std::current_exception();
        const auto primary = native_submission_error();
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->native_cleanup_error_ = secondary;
        }
        if (primary) {
            // Keep the exact primary in the entry/origin. The reported failure
            // names it and nests the actual unknown cleanup error, not success.
            std::string reason = "unknown native submission exception";
            try { std::rethrow_exception(primary); }
            catch (const std::exception& error) { reason = error.what(); }
            catch (...) {}
            std::throw_with_nested(std::runtime_error(
                "Native submission failed: " + reason + "; graph cleanup remains unresolved"));
        }
        throw;
    }
    if (auto error = native_submission_error()) std::rethrow_exception(error);
    state_->has_graph_exec_ = true;

    uint32_t num_graph_nodes = 0;
}

void NPUGraph::replay()
{
    TORCH_CHECK(state_->has_graph_exec_,
                "Called NPUGraph::replay without a preceding successful capture.");

    c10::OptionalDeviceGuard device_guard{state_->capture_stream_.device()};

    for (auto& [generator_state, wholegraph_increments] : state_->captured_generator_states_) {
        generator_state->replay_prologue(wholegraph_increments);
    }

    // state_->model_ri_ may be replayed in any stream.
    NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIExecuteAsync(state_->model_ri_, c10_npu::getCurrentNPUStream()));
}

void NPUGraph::enable_debug_mode()
{
    _npu_graphs_debug = true;
}

void NPUGraph::debug_dump(const std::string& debug_path)
{
    if (state_->has_graph_exec_) {
        TORCH_WARN("calling NPUGraph::debug_dump() for model id ", state_->model_ri_);
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDebugJsonPrint(state_->model_ri_, debug_path.c_str()));
    } else {
        TORCH_WARN("Called NPUGraph::debug_dump without a preceding successful capture.");
    }
}

void NPUGraph::super_kernel_optimize(const aclskOptions *options)
{
    TORCH_CHECK(state_->has_graph_exec_,
                "Called NPUGraph::super_kernel_optimize without a preceding successful capture.");
    NPU_CHECK_ERROR(c10_npu::skapi::AclskOptimize(state_->model_ri_, options));
}

void NPUGraph::reset()
{
    State::Lifecycle lifecycle(*state_);
    reset_impl();
}

void NPUGraph::reset_impl()
{
    close_native_admission();
    TORCH_CHECK(!state_->capture_pending_,
                "Capture disposition is unresolved; reset cannot release its backing resources.");
    cancel_native_pending();
    if (state_->model_owned_) {
        TORCH_CHECK(c10_npu::NpuSysCtrl::GetInstance().GetInitFlag(),
                    "Cannot destroy an NPU graph after runtime shutdown.");
        NPU_CHECK_ERROR(c10_npu::acl::AclmdlRIDestroy(state_->model_ri_));
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->model_owned_ = false;
            state_->model_ri_ = nullptr;
            state_->has_graph_exec_ = false;
        }
    }
    release_native_resources();
    if (state_->pool_capture_active_) {
        c10_npu::NPUCachingAllocator::endAllocateToPool(state_->capture_dev_, state_->mempool_id_);
        state_->pool_capture_active_ = false;
    }
    if (state_->pool_held_) {
        c10_npu::NPUCachingAllocator::releasePool(state_->capture_dev_, state_->mempool_id_);
        state_->pool_held_ = false;
    }
    for (auto* generator_state : state_->prepared_generator_states_) {
        generator_state->capture_epilogue();
    }
    state_->prepared_generator_states_.clear();
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t NPUGraph::pool()
{
    TORCH_CHECK(state_->has_graph_exec_,
                "Called NPUGraph::pool() without a preceding successful capture.");
    return state_->mempool_id_;
}

NPUGraph::~NPUGraph() noexcept
{
    try {
        // Model/pool obligations must finish before RNG backing can be released.
        reset();
        while (!state_->captured_generator_states_.empty()) {
            auto it = state_->captured_generator_states_.begin();
            it->first->unregister_graph(this);
            state_->captured_generator_states_.erase(it);
        }
    } catch (const std::exception& error) {
        // Keep the same allocation and all outstanding obligations alive. The
        // wrapper may be stack-owned; callbacks must never refer to its address.
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->abandoned_ = true;
        }
        state_.reset();
        std::fprintf(stderr, "NPUGraph cleanup failed; retaining graph resources: %.512s\n", error.what());
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state_->native_mutex_);
            state_->abandoned_ = true;
        }
        state_.reset();
        std::fputs("NPUGraph cleanup failed; retaining graph resources.\n", stderr);
    }
}

} // namespace c10_npu
