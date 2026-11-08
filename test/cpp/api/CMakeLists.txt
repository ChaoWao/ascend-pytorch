set(TORCH_API_TEST_DIR "${PROJECT_SOURCE_DIR}/test/cpp/api")
set(TORCH_API_TEST_SOURCES ${TORCH_API_TEST_DIR}/modules.cpp PARENT_SCOPE)

# Compile the complete production graph implementation into the test executable.
# Link wrapping is confined to this executable's SDK End/Destroy and queue-drain leaves.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(test_native_graph_state
    ${TORCH_API_TEST_DIR}/native_graph_state.cpp
    ${PROJECT_SOURCE_DIR}/torch_npu/csrc/core/npu/NPUGraph.cpp)
  target_link_libraries(test_native_graph_state PUBLIC torch_npu gtest_main gtest)
  target_link_options(test_native_graph_state PRIVATE
    "-Wl,--wrap=_ZN7c10_npu3acl15AclmdlRIDestroyEPv"
    "-Wl,--wrap=_ZN7c10_npu3acl18AclmdlRICaptureEndEPvPS1_"
    "-Wl,--wrap=_ZNK7c10_npu9NPUStream6streamEv")
  add_test(NAME native_graph_state_queue1 COMMAND test_native_graph_state)
  set_tests_properties(native_graph_state_queue1 PROPERTIES
    ENVIRONMENT "TASK_QUEUE_ENABLE=1;TORCH_DEVICE_BACKEND_AUTOLOAD=0"
    TIMEOUT 300)
endif()

# Header-only native-reference contract: no Torch/SDK library is loaded by this target.
add_executable(test_native_submission_ref ${TORCH_API_TEST_DIR}/native_submission_ref.cpp)
target_include_directories(test_native_submission_ref PRIVATE ${PROJECT_SOURCE_DIR})
add_test(NAME native_submission_ref COMMAND test_native_submission_ref)
