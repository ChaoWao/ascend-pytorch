"""Exercise setup's actual configuration methods without running its build actions."""
import ast
import json
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from sysconfig import get_paths
from types import SimpleNamespace
from unittest.mock import patch

from setuptools import Distribution, Extension
from setuptools.command.build_clib import build_clib


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("PYBIND11_COMPILER_TYPE", "PYBIND11_STDLIB", "PYBIND11_BUILD_ABI")


def setup_methods(directory, use_cxx11_abi=True):
    tree = ast.parse((ROOT / "setup.py").read_text())
    selected = {"get_pybind11_abi_constants", "CppExtension", "CPPLibBuild"}
    methods = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected]
    commands = []
    namespace = {
        "os": os, "sys": sys, "get_paths": get_paths, "multiprocessing": multiprocessing,
        "Extension": Extension, "build_clib": build_clib,
        "BASE_DIR": directory, "BUILD_PERMISSION": 0o750,
        "VERSION": "2.7.1", "DISABLE_TORCHAIR": "TRUE", "DISABLE_RPC": "TRUE",
        "ENABLE_LTO": "FALSE", "PGO_MODE": 0, "USE_CXX11_ABI": use_cxx11_abi,
        "get_cmake_command": lambda: "cmake", "get_build_type": lambda: "Release",
        "get_pytorch_dir": lambda: "/selected/torch",
        "subprocess": SimpleNamespace(check_call=lambda argv, **kwargs: commands.append(argv)),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(ROOT / "setup.py"), "exec"), namespace)
    return namespace, commands


class TestBuildPybindConfig(unittest.TestCase):
    def setUp(self):
        self.tags = dict(zip(NAMES, ("_test_compiler", "_test_stdlib", "_test_abi")))
        self.torch = SimpleNamespace(_C=SimpleNamespace(**{"_" + k: v for k, v in self.tags.items()}))
        self.modules = patch.dict(sys.modules, {"torch": self.torch})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_extension_uses_selected_torch_tags_and_keeps_abi_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace, _ = setup_methods(directory)
            abi = [("_GLIBCXX_USE_CXX11_ABI", "0"), ("GLIBCXX_USE_CXX11_ABI", "0")]
            extension = namespace["CppExtension"]("probe", ["probe.cpp"], define_macros=abi)
            macros = dict(extension.define_macros)
            for name, value in self.tags.items():
                self.assertEqual(macros.get(name), '"' + value + '"')
            self.assertEqual(macros["_GLIBCXX_USE_CXX11_ABI"], "0")
            self.assertEqual(macros["GLIBCXX_USE_CXX11_ABI"], "0")

    def cmake_arguments(self, use_cxx11_abi):
        with tempfile.TemporaryDirectory() as directory:
            namespace, commands = setup_methods(directory, use_cxx11_abi)
            command = namespace["CPPLibBuild"](Distribution())
            command.run()
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][0], "cmake")
            self.assertEqual(commands[1][0], "make")
            return commands[0]

    def test_core_and_atb_cmake_receive_all_selected_torch_tags(self):
        arguments = self.cmake_arguments(True)
        for name, value in self.tags.items():
            self.assertIn("-D" + name + "=" + value, arguments)
        self.assertIn("-DGLIBCXX_USE_CXX11_ABI=1", arguments)

    def test_cmake_keeps_existing_abi_zero_default(self):
        arguments = self.cmake_arguments(False)
        for name, value in self.tags.items():
            self.assertIn("-D" + name + "=" + value, arguments)
        self.assertNotIn("-DGLIBCXX_USE_CXX11_ABI=1", arguments)

    def test_cmake_tag_definitions_reach_subdirectory_targets(self):
        source = (ROOT / "CMakeLists.txt").read_text()
        match = re.search(r"foreach\(_pybind11_constant .*?endforeach\(\)", source, re.S)
        self.assertIsNotNone(match)
        block = match.group()
        self.assertLess(match.start(), source.index("add_library(${PLUGIN_NAME}"))
        self.assertLess(match.start(), source.index("add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/op-plugin"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "nested").mkdir()
            (path / "probe.cpp").write_text("int value;\n")
            (path / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\nproject(configuration_scope CXX)\n"
                + block + "\nadd_library(local_scope OBJECT probe.cpp)\nadd_subdirectory(nested)\n")
            (path / "nested/CMakeLists.txt").write_text("add_library(nested_scope OBJECT ../probe.cpp)\n")
            command = [shutil.which("cmake"), "-S", directory, "-B", str(path / "build"),
                       "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_CXX_FLAGS=-DUSER_PRESET=1"]
            command.extend("-D" + name + "=" + value for name, value in self.tags.items())
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            commands = json.loads((path / "build/compile_commands.json").read_text())
            self.assertEqual(len(commands), 2)
            for entry in commands:
                words = shlex.split(entry["command"])
                self.assertIn("-DUSER_PRESET=1", words)
                for name, value in self.tags.items():
                    self.assertIn("-D" + name + '=\"' + value + '\"', words)

    def test_tag_values_are_read_from_each_selected_torch(self):
        self.torch._C._PYBIND11_BUILD_ABI = "_different_peer"
        with tempfile.TemporaryDirectory() as directory:
            namespace, _ = setup_methods(directory)
            extension = namespace["CppExtension"]("probe", ["probe.cpp"])
            self.assertEqual(dict(extension.define_macros).get("PYBIND11_BUILD_ABI"), '"_different_peer"')


if __name__ == "__main__":
    unittest.main()
