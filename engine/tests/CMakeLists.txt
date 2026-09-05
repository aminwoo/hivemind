include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  hivemind_tests
  test_backend_compat.cc
  test_move_gen.cc
  test_tactical_movegen.cc
  test_supply_search.cc
  test_draw_detection.cc
  test_mate_detection.cc
  test_mate_probe.cc
  test_onnx_utils.cc
  test_uci.cc
  test_tournament.cc
  test_node_concurrency.cc
)
target_link_libraries(hivemind_tests PRIVATE hivemind_lib gtest_main)

include(GoogleTest)
gtest_discover_tests(hivemind_tests)
