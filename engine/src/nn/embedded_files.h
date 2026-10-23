#pragma once

#include <span>
#include <string_view>

/**
 * @brief Files compiled into the executable, so a single-file release needs
 * nothing beside it: the network (HIVEMIND_EMBED_MODEL) and the license
 * texts it must carry (HIVEMIND_EMBED_LICENSES).
 *
 * resolveModelPath() returns MODEL_NAME in place of a path when no model is
 * given and one is embedded; the ONNX Runtime backend then loads it from
 * memory.
 */
namespace embedded {

inline constexpr std::string_view MODEL_NAME = "<embedded>";

/// The embedded ONNX model, or an empty span when none was compiled in.
std::span<const unsigned char> model();

/// The embedded license texts, or an empty view when none were compiled in.
std::string_view licenses();

inline bool has_model() { return !model().empty(); }

inline bool is_model(std::string_view path) { return path == MODEL_NAME; }

}  // namespace embedded
