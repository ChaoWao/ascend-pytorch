#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <atomic>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <new>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

#include "torch_npu/csrc/libs/torch_npu.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/native_submission.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCommand.h"

// Identical private binding access to Graph.cpp; no test-only production observer.
namespace c10_npu {
struct NPUGraphNativeAccess {
    static SimplerNativeSubmission retain(NPUGraph& graph, const SimplerNativeSubmission& api,
                                          std::vector<at::Tensor> tensors)
    {
        return graph.register_native_submission(api, std::move(tensors));
    }
};
}

namespace {
using namespace std::chrono_literals;
std::atomic<int> destroy_calls{0};
std::atomic<int> destroyed_models{0};
std::atomic<int> destroy_error{0};
std::atomic<int> end_calls{0};
std::atomic<int> end_error{0};

struct StreamWatch {
    std::promise<void> entered;
    std::atomic<bool> signalled{false};
};
thread_local StreamWatch* registration_watch = nullptr;

struct Probe {
    int retains = 0;
    int submits = 0;
    int cancels = 0;
    int releases = 0;
    int retain_error = 0;
    int release_failures = 0;
    int token_destructors = 0;
    bool registered = false;
    bool retired = false;
    bool resources = false;
    aclrtStream stream = nullptr;
    std::atomic<bool> callback_completed{false};
    Probe* predecessor = nullptr;
    bool predecessor_completed_at_retain = false;
};

// This is the native producer's protocol leaf, not an implementation of graph ownership.
struct Token {
    explicit Token(std::shared_ptr<Probe> probe) : probe(std::move(probe)) {}
    ~Token() { ++probe->token_destructors; }
    std::atomic<unsigned> references{1};
    std::shared_ptr<Probe> probe;

    static Token& self(void* context) { return *static_cast<Token*>(context); }
    static void retain(void* context) noexcept { ++self(context).references; }
    static void release(void* context) noexcept {
        if (--self(context).references == 0) delete &self(context);
    }
    static int fail(char* message, size_t capacity, const char* text) noexcept {
        if (message && capacity) std::snprintf(message, capacity, "%s", text);
        return -1;
    }
    static int acquire(void* context, int* started, char* message, size_t capacity) noexcept {
        auto& p = *self(context).probe;
        *started = 0;
        ++p.retains;
        if (p.registered) return fail(message, capacity, "already registered");
        p.registered = true;
        *started = 1;
        p.resources = true;
        if (p.predecessor) p.predecessor_completed_at_retain = p.predecessor->callback_completed.load();
        if (p.retain_error) return fail(message, capacity, "retain failed after registration started");
        return 0;
    }
    static int submit(void* context, SimplerNativePreflight preflight, void* argument,
                      char* message, size_t capacity) noexcept {
        auto& p = *self(context).probe;
        ++p.submits;
        if (!preflight) return fail(message, capacity, "graph preflight missing");
        if (preflight(argument, p.stream, message, capacity) != 0) return -1;
        p.retired = true;
        return 0;
    }
    static int cancel(void* context, NativeSubmissionCleanup* out, char*, size_t) noexcept {
        auto& p = *self(context).probe;
        ++p.cancels;
        p.retired = true;
        *out = {p.submits ? NativeSubmissionDisposition::Submitted : NativeSubmissionDisposition::Cancelled, true};
        return 0;
    }
    static int release_resources(void* context, char* message, size_t capacity) noexcept {
        auto& p = *self(context).probe;
        ++p.releases;
        if (!p.retired || destroyed_models.load() == 0)
            return fail(message, capacity, "release before checked carrier/model retirement");
        if (p.release_failures > 0) {
            --p.release_failures;
            return fail(message, capacity, "injected checked release failure");
        }
        p.resources = false;
        return 0;
    }
    SimplerNativeSubmission api() {
        return {this, retain, release, submit, acquire, cancel, release_resources};
    }
};
struct TokenDeleter { void operator()(Token* token) const { Token::release(token); } };
using TokenOwner = std::unique_ptr<Token, TokenDeleter>;

NativeSubmissionRef register_call(c10_npu::NPUGraph& graph, Token& token, std::vector<at::Tensor> tensors = {}) {
    auto api = c10_npu::NPUGraphNativeAccess::retain(graph, token.api(), std::move(tensors));
    NativeSubmissionRef result(api);
    api.release(api.context); // The original returned owner, as for the production capsule.
    return result;
}
void enqueue(const NativeSubmissionRef& submission) {
    at_npu::native::OpCommand::RunOpApi("NativeGraphStateTest", [submission]() -> int {
        try { submission.submit(); return 0; } catch (...) { return -1; }
    }, false);
}

class NativeGraphStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        destroy_calls = 0;
        destroyed_models = 0;
        destroy_error = 0;
        end_calls = 0;
        end_error = 0;
        torch_npu::init_npu(at::Device("npu:0"));
        stream.emplace(c10_npu::getStreamFromPool(false, 0));
        guard = std::make_unique<c10_npu::NPUStreamGuard>(*stream);
        input = at::ones({4}, at::TensorOptions().device(at::Device("npu:0")));
        output = at::empty_like(input);
        stream->synchronize();
    }
    void TearDown() override {
        destroy_error = 0;
        end_error = 0;
        registration_watch = nullptr;
        if (stream) stream->synchronize();
        input = at::Tensor();
        output = at::Tensor();
        guard.reset();
        stream.reset();
        torch_npu::finalize_npu();
    }
    TokenOwner token(const std::shared_ptr<Probe>& probe) {
        probe->stream = stream->stream(false);
        return TokenOwner(new Token(probe));
    }
    void node() { at::add_out(output, input, input); }
    std::optional<c10_npu::NPUStream> stream;
    std::unique_ptr<c10_npu::NPUStreamGuard> guard;
    at::Tensor input, output;
};

TEST_F(NativeGraphStateTest, RealStateKeepsTensorAndNativeReferenceUntilCheckedReset) {
    auto p = std::make_shared<Probe>();
    auto origin = token(p);
    auto rng = c10::make_intrusive<at_npu::NPUGeneratorState>();
    std::optional<c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>> weak;
    {
        c10_npu::NPUGraph graph;
        graph.register_generator_state(rng);
        graph.capture_begin();
        auto tensor = at::ones({3});
        weak.emplace(tensor.getIntrusivePtr());
        auto queued = register_call(graph, *origin, {tensor});
        tensor = at::Tensor();
        enqueue(queued);
        node();
        graph.capture_end();
        EXPECT_FALSE(weak->expired());
        EXPECT_TRUE(p->resources);
        graph.reset();
        EXPECT_TRUE(weak->expired());
        EXPECT_FALSE(p->resources);
        EXPECT_EQ(p->releases, 1);
        EXPECT_EQ(destroyed_models, 1);
    }
    EXPECT_TRUE(rng->registered_graphs_.empty());
    EXPECT_FALSE(rng->seed_extragraph_.defined());
    EXPECT_FALSE(rng->offset_extragraph_.defined());
    EXPECT_EQ(origin->references, 1u);
    origin.reset();
    EXPECT_EQ(p->token_destructors, 1);
}

TEST_F(NativeGraphStateTest, CheckedDestroyFailureRetainsBackingForTheSameResetRetry) {
    auto p = std::make_shared<Probe>();
    auto origin = token(p);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto queued = register_call(graph, *origin);
    enqueue(queued);
    node();
    graph.capture_end();
    destroy_error = 771001;
    EXPECT_THROW(graph.reset(), c10::Error);
    EXPECT_TRUE(p->resources);
    EXPECT_EQ(p->releases, 0);
    EXPECT_EQ(destroyed_models, 0);
    destroy_error = 0;
    graph.reset();
    EXPECT_EQ(destroy_calls, 2);
    EXPECT_EQ(destroyed_models, 1);
    EXPECT_EQ(p->releases, 1);
}

TEST_F(NativeGraphStateTest, ReleaseRetryDoesNotDestroyTheModelTwice) {
    auto p = std::make_shared<Probe>();
    p->release_failures = 1;
    auto origin = token(p);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto queued = register_call(graph, *origin);
    enqueue(queued);
    node();
    graph.capture_end();
    EXPECT_THROW(graph.reset(), std::runtime_error);
    EXPECT_EQ(destroyed_models, 1);
    EXPECT_TRUE(p->resources);
    graph.reset();
    EXPECT_EQ(destroy_calls, 1);
    EXPECT_EQ(p->releases, 2);
    EXPECT_FALSE(p->resources);
}

TEST_F(NativeGraphStateTest, MissingDispatchCannotDeliverASuccessfulPartialGraph) {
    auto p = std::make_shared<Probe>();
    auto origin = token(p);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto never_published = register_call(graph, *origin);
    node();
    EXPECT_THROW(graph.capture_end(), std::runtime_error);
    EXPECT_THROW(graph.replay(), c10::Error);
    EXPECT_EQ(p->submits, 0);
    EXPECT_TRUE(p->retired);
    EXPECT_EQ(end_calls, 1);
    graph.reset();
    EXPECT_EQ(destroyed_models, 1);
    EXPECT_FALSE(p->resources);
}

TEST_F(NativeGraphStateTest, StartedAcquireFailureKeepsItsOriginalEntryForCheckedCleanup) {
    auto p = std::make_shared<Probe>();
    p->retain_error = -73;
    auto origin = token(p);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    EXPECT_THROW(register_call(graph, *origin), std::runtime_error);
    node();
    EXPECT_THROW(graph.capture_end(), std::runtime_error);
    EXPECT_EQ(p->submits, 0);
    EXPECT_EQ(p->retains, 1);
    EXPECT_TRUE(p->resources);
    graph.reset();
    EXPECT_FALSE(p->resources);
}

TEST_F(NativeGraphStateTest, DuplicateRegistrationDoesNotCancelTheOriginalEntry) {
    auto p = std::make_shared<Probe>();
    auto origin = token(p);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto queued = register_call(graph, *origin);
    EXPECT_THROW(register_call(graph, *origin), std::runtime_error);
    EXPECT_EQ(p->cancels, 0);
    enqueue(queued);
    node();
    graph.capture_end();
    graph.reset();
    EXPECT_EQ(p->submits, 1);
    EXPECT_EQ(p->releases, 1);
}

TEST_F(NativeGraphStateTest, OldQueueContextCannotSubmitIntoTheNextCapture) {
    auto a = std::make_shared<Probe>();
    auto original = token(a);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto old = register_call(graph, *original);
    enqueue(old);
    node();
    graph.capture_end();
    graph.reset();
    graph.capture_begin();
    EXPECT_THROW(old.submit(), std::runtime_error);
    EXPECT_EQ(a->submits, 1);
    node();
    graph.capture_end();
    graph.reset();
}

TEST_F(NativeGraphStateTest, RegistrationDrainsTheOriginalHostQueueWithoutHoldingStateLock) {
    ASSERT_EQ(c10_npu::option::OptionsManager::GetTaskQueueEnable(), 1);
    auto a = std::make_shared<Probe>();
    auto b = std::make_shared<Probe>();
    b->predecessor = a.get();
    auto origin_a = token(a);
    auto origin_b = token(b);
    c10_npu::NPUGraph graph;
    graph.capture_begin();
    auto queued = register_call(graph, *origin_a);
    std::promise<void> callback_entered, release_callback;
    auto entered = callback_entered.get_future();
    auto release = release_callback.get_future().share();
    at_npu::native::OpCommand::RunOpApi("NativeGraphGate", [&, queued]() -> int {
        callback_entered.set_value();
        release.wait();
        try {
            queued.submit();
            a->callback_completed = true;
            return 0;
        } catch (...) { return -1; }
    }, false);
    StreamWatch watch;
    auto conversion = watch.entered.get_future();
    auto controller = std::async(std::launch::async, [&] {
        const bool saw_conversion = conversion.wait_for(5s) == std::future_status::ready;
        release_callback.set_value();
        return saw_conversion;
    });
    // The timeout is a missing-event guard, not evidence that a mutex was held.
    const bool saw_callback = entered.wait_for(5s) == std::future_status::ready;
    registration_watch = &watch;
    std::optional<NativeSubmissionRef> second;
    std::exception_ptr failure;
    try { second.emplace(register_call(graph, *origin_b)); } catch (...) { failure = std::current_exception(); }
    registration_watch = nullptr;
    const bool saw_conversion = controller.get();
    EXPECT_TRUE(saw_callback);
    EXPECT_TRUE(saw_conversion);
    EXPECT_TRUE(b->predecessor_completed_at_retain);
    if (failure) {
        try { graph.capture_end(); } catch (...) {}
        graph.reset();
        std::rethrow_exception(failure);
    }
    enqueue(*second);
    node();
    graph.capture_end();
    graph.reset();
}

// Failed End cannot be assumed abortable. Each uncertain disposition stays in
// its own re-executed child; process exit is not reported as checked graph cleanup.
TEST(NativeGraphStateDeathTest, FailedEndKeepsTheOriginalStateUnresolved) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(([] {
        torch_npu::init_npu(at::Device("npu:0"));
        auto stream = c10_npu::getStreamFromPool(false, 0);
        c10_npu::NPUStreamGuard guard(stream);
        auto p = std::make_shared<Probe>();
        p->stream = stream.stream(false);
        TokenOwner origin(new Token(p));
        c10_npu::NPUGraph graph;
        graph.capture_begin();
        auto queued = register_call(graph, *origin, {at::ones({2})});
        enqueue(queued);
        end_error = 771002;
        bool failed = false, refused = false;
        try { graph.capture_end(); } catch (...) { failed = true; }
        try { graph.reset(); } catch (...) { refused = true; }
        if (!failed || !refused || !p->resources || p->releases != 0 || destroy_calls != 0) std::_Exit(2);
        std::_Exit(0);
    }()), ::testing::ExitedWithCode(0), ".*");
}

TEST(NativeGraphStateDeathTest, LastWrapperDestroyFailureKeepsTensorAndNativeRootsAlive) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(([] {
        torch_npu::init_npu(at::Device("npu:0"));
        auto stream = c10_npu::getStreamFromPool(false, 0);
        c10_npu::NPUStreamGuard guard(stream);
        auto p = std::make_shared<Probe>();
        p->stream = stream.stream(false);
        TokenOwner origin(new Token(p));
        std::optional<c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>> weak;
        {
            c10_npu::NPUGraph graph;
            graph.capture_begin();
            auto tensor = at::ones({2});
            weak.emplace(tensor.getIntrusivePtr());
            auto queued = register_call(graph, *origin, {tensor});
            tensor = at::Tensor();
            queued.submit();
            graph.capture_end();
            destroy_error = 771003;
        }
        if (weak->expired() || !p->resources || p->releases != 0 || origin->references != 2) std::_Exit(2);
        std::_Exit(0);
    }()), ::testing::ExitedWithCode(0), ".*");
}

TEST(NativeGraphStateDeathTest, ReusedStackWrapperCannotReuseAnAbandonedRngMembership) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(([] {
        torch_npu::init_npu(at::Device("npu:0"));
        auto stream = c10_npu::getStreamFromPool(false, 0);
        c10_npu::NPUStreamGuard guard(stream);
        auto rng = c10::make_intrusive<at_npu::NPUGeneratorState>();
        alignas(c10_npu::NPUGraph) unsigned char storage[sizeof(c10_npu::NPUGraph)];
        auto* first = new (storage) c10_npu::NPUGraph();
        first->register_generator_state(rng);
        first->capture_begin();
        first->capture_end();
        destroy_error = 771004;
        first->~NPUGraph();
        if (rng->registered_graphs_.size() != 1 || !rng->seed_extragraph_.defined()) std::_Exit(2);
        destroy_error = 0;
        auto* second = new (storage) c10_npu::NPUGraph();
        second->register_generator_state(rng);
        if (rng->registered_graphs_.size() != 2) std::_Exit(3);
        second->capture_begin();
        second->capture_end();
        second->reset();
        second->~NPUGraph();
        if (rng->registered_graphs_.size() != 1 || !rng->offset_extragraph_.defined()) std::_Exit(4);
        std::_Exit(0);
    }()), ::testing::ExitedWithCode(0), ".*");
}
} // namespace

extern "C" aclError real_graph_destroy(aclmdlRI) asm("__real__ZN7c10_npu3acl15AclmdlRIDestroyEPv");
extern "C" aclError wrapped_graph_destroy(aclmdlRI) asm("__wrap__ZN7c10_npu3acl15AclmdlRIDestroyEPv");
extern "C" aclError wrapped_graph_destroy(aclmdlRI model) {
    ++destroy_calls;
    if (destroy_error.load()) return destroy_error.load();
    const aclError rc = real_graph_destroy(model);
    if (rc == 0) ++destroyed_models;
    return rc;
}
extern "C" aclError real_graph_end(aclrtStream, aclmdlRI*) asm("__real__ZN7c10_npu3acl18AclmdlRICaptureEndEPvPS1_");
extern "C" aclError wrapped_graph_end(aclrtStream, aclmdlRI*) asm("__wrap__ZN7c10_npu3acl18AclmdlRICaptureEndEPvPS1_");
extern "C" aclError wrapped_graph_end(aclrtStream stream, aclmdlRI* model) {
    ++end_calls;
    if (end_error.load()) return end_error.load();
    return real_graph_end(stream, model);
}
extern "C" aclrtStream real_graph_stream(const c10_npu::NPUStream*) asm("__real__ZNK7c10_npu9NPUStream6streamEv");
extern "C" aclrtStream wrapped_graph_stream(const c10_npu::NPUStream*) asm("__wrap__ZNK7c10_npu9NPUStream6streamEv");
extern "C" aclrtStream wrapped_graph_stream(const c10_npu::NPUStream* stream) {
    if (registration_watch && !registration_watch->signalled.exchange(true)) registration_watch->entered.set_value();
    return real_graph_stream(stream);
}
