"""Context restoration on the actual graphs.graph class; SDK leaves are controlled.

Native State, End/Destroy and queue tests live in test/cpp/api/native_graph_state.cpp.
These Python tests do not substitute a fake Graph for native State qualification.
"""

import contextlib
import types
import unittest
from unittest import mock

from torch_npu.npu import graphs


class Context:
    def __init__(self, events, name, *, enter_error=None, exit_error=None):
        self.events, self.name = events, name
        self.enter_error, self.exit_error = enter_error, exit_error

    def __enter__(self):
        self.events.append("enter-" + self.name)
        if self.enter_error is not None:
            raise self.enter_error

    def __exit__(self, *exception):
        self.events.append("exit-" + self.name)
        if self.exit_error is not None:
            raise self.exit_error


class TestNativeGraphContext(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.previous = object()
        self.old_route = graphs._current_native_graph()
        graphs._native_graph_context.graph = self.previous
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.restore_route)
        self.stream = Context(self.events, "stream")
        self.dispatch = Context(self.events, "dispatch")
        self.graph = types.SimpleNamespace(
            auto_dispatch_capture=True,
            graph_dispatch_mode=self.dispatch,
            capture_begin=mock.Mock(side_effect=lambda **kw: self.events.append("begin")),
            capture_end=mock.Mock(side_effect=lambda: self.events.append("end")),
        )
        for patch in (
            mock.patch.object(graphs.graph, "default_capture_stream", object()),
            mock.patch.object(graphs.torch.npu, "stream", return_value=self.stream),
            mock.patch.object(graphs.torch.npu, "synchronize"),
            mock.patch.object(graphs.torch.npu, "empty_cache"),
            mock.patch.object(graphs, "force_npugraph_gc", False),
        ):
            self.stack.enter_context(patch)

    def restore_route(self):
        graphs._native_graph_context.graph = self.old_route

    def manager(self):
        return graphs.graph(self.graph, stream=object(), auto_dispatch_capture=True)

    def test_success_restores_original_route_and_contexts_in_reverse_order(self):
        with self.manager():
            self.assertIs(graphs._current_native_graph(), self.graph)
        self.assertIs(graphs._current_native_graph(), self.previous)
        self.assertEqual(self.events, ["enter-stream", "enter-dispatch", "begin", "end", "exit-dispatch", "exit-stream"])

    def test_failed_begin_restores_every_successfully_entered_context(self):
        primary = RuntimeError("begin")
        self.graph.capture_begin.side_effect = primary
        with self.assertRaises(RuntimeError) as caught:
            with self.manager():
                self.fail("body entered after failed Begin")
        self.assertIs(caught.exception, primary)
        self.graph.capture_end.assert_not_called()
        self.assertEqual(self.events[-2:], ["exit-dispatch", "exit-stream"])
        self.assertIs(graphs._current_native_graph(), self.previous)

    def test_failed_dispatch_enter_exits_only_the_stream(self):
        primary = MemoryError("dispatch enter")
        self.dispatch.enter_error = primary
        with self.assertRaises(MemoryError) as caught:
            with self.manager():
                self.fail("body entered after failed dispatch enter")
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.events, ["enter-stream", "enter-dispatch", "exit-stream"])
        self.graph.capture_begin.assert_not_called()
        self.assertIs(graphs._current_native_graph(), self.previous)

    def test_failed_stream_enter_has_no_false_exit_obligation(self):
        primary = MemoryError("stream enter")
        self.stream.enter_error = primary
        with self.assertRaises(MemoryError) as caught:
            with self.manager():
                self.fail("body entered after failed stream enter")
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.events, ["enter-stream"])
        self.assertIs(graphs._current_native_graph(), self.previous)

    def test_body_error_remains_primary_when_end_and_restoration_fail(self):
        primary = ValueError("body")
        self.graph.capture_end.side_effect = RuntimeError("End unknown")
        self.dispatch.exit_error = MemoryError("dispatch exit")
        with self.assertRaises(ValueError) as caught:
            with self.manager():
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertEqual(self.events[-2:], ["exit-dispatch", "exit-stream"])
        self.assertIs(graphs._current_native_graph(), self.previous)

    def test_end_error_remains_primary_when_stream_exit_fails(self):
        primary = RuntimeError("End unknown")
        self.graph.capture_end.side_effect = primary
        self.stream.exit_error = MemoryError("stream exit")
        with self.assertRaises(RuntimeError) as caught:
            with self.manager():
                pass
        self.assertIs(caught.exception, primary)
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIs(graphs._current_native_graph(), self.previous)


if __name__ == "__main__":
    unittest.main()
