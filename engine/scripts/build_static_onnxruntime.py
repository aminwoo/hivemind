#!/usr/bin/env python3
"""Build ONNX Runtime from source as one static library, for single-file engines.

Microsoft publishes ONNX Runtime only as a shared library. This builds a
release tag from source (CPU only, plus Core ML on macOS), merges the static
libraries it produces into one archive and installs it with the public
headers, so CMake can link it with -DHIVEMIND_ORT_STATIC=ON
-DONNXRuntime_ROOT=<output>:

    python3 engine/scripts/build_static_onnxruntime.py --output third_party/onnxruntime-static

On Windows, run it from a Developer PowerShell so lib.exe is on PATH. The
source and build trees go to --work (default: third_party/onnxruntime-build).
"""
import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

DEFAULT_VERSION = "1.29.0"
ROOT = Path(__file__).resolve().parents[2]
# Static libraries the build produces that ONNX Runtime itself does not link:
# test frameworks, the full protobuf (ORT uses protobuf-lite) and the protoc
# compiler, whose objects would clash with protobuf-lite's.
EXCLUDED = re.compile(
    r"(gtest|gmock|benchmark|protoc|libprotobuf\.a$|libprotobufd?\.lib$|^protobufd?\.lib$"
    r"|onnx_test|custom_op|test_|_test)", re.IGNORECASE)


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def static_libraries(build_dir):
    suffix = ".lib" if os.name == "nt" else ".a"
    libraries = sorted(
        path for path in build_dir.rglob(f"*{suffix}")
        if not EXCLUDED.search(path.name) and "CMakeFiles" not in path.parts)
    if not libraries:
        raise SystemExit(f"No static libraries found under {build_dir}")
    return libraries


def merge(libraries, output):
    """Combines static libraries into one archive with the platform's tool."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    system = platform.system()
    if system == "Darwin":
        run(["libtool", "-static", "-no_warning_for_no_symbols", "-o", output, *libraries])
    elif system == "Windows":
        run(["lib.exe", "/NOLOGO", f"/OUT:{output}", *libraries])
    else:
        script = "\n".join([f"CREATE {output}", *(f"ADDLIB {library}" for library in libraries),
                            "SAVE", "END", ""])
        run(["ar", "-M"], input=script.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--output", type=Path, default=ROOT / "third_party" / "onnxruntime-static")
    parser.add_argument("--work", type=Path, default=ROOT / "third_party" / "onnxruntime-build")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    source = args.work / f"onnxruntime-{args.version}"
    build = args.work / f"build-{args.version}"
    if not (source / "cmake" / "CMakeLists.txt").is_file():
        args.work.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", f"v{args.version}",
             "https://github.com/microsoft/onnxruntime.git", source])

    system = platform.system()
    configure = [
        "cmake", "-S", source / "cmake", "-B", build,
        "-DCMAKE_BUILD_TYPE=Release",
        "-Donnxruntime_BUILD_SHARED_LIB=OFF",
        "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
        "-Donnxruntime_BUILD_BENCHMARKS=OFF",
        "-Donnxruntime_ENABLE_PYTHON=OFF",
        "-Donnxruntime_USE_FULL_PROTOBUF=OFF",
        # ONNX Runtime builds its targets with warnings as errors, which
        # newer compilers than it was tested with trip on.
        "--compile-no-warning-as-error",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        # Some dependencies (psimd on macOS) still declare CMake < 3.5, which
        # CMake 4 refuses unless told the minimum policy version to assume.
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DPython_EXECUTABLE={sys.executable}",
    ]
    # Ninja on every platform: Visual Studio solutions cannot build the
    # fetched dependencies' targets (re2) by name. On Windows this needs a
    # developer environment, which also provides lib.exe.
    configure += ["-G", "Ninja"]
    if system == "Windows":
        # Match the engine's static C++ runtime, in ONNX Runtime and in every
        # fetched dependency (what ONNX Runtime's own build.py does for
        # --enable_msvc_static_runtime); one object built against the DLL
        # runtime leaves __imp_ symbols the static link cannot resolve.
        configure += ["-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                      "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
                      "-Donnxruntime_MSVC_STATIC_RUNTIME=ON",
                      "-Dprotobuf_MSVC_STATIC_RUNTIME=ON",
                      "-DABSL_MSVC_STATIC_RUNTIME=ON",
                      "-Dgtest_force_shared_crt=OFF"]
    if system == "Darwin":
        configure += ["-Donnxruntime_USE_COREML=ON", "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0"]
    run(configure)
    run(["cmake", "--build", build, "--config", "Release", "--parallel", args.jobs])
    # The default target builds only what a static ONNX Runtime archive
    # needs to compile; re2 is needed only to link, so build it explicitly
    # where ONNX Runtime fetched it as a subproject.
    if (build / "_deps" / "re2-build").is_dir():
        run(["cmake", "--build", build, "--config", "Release", "--parallel", args.jobs,
             "--target", "re2"])

    library_name = "onnxruntime.lib" if os.name == "nt" else "libonnxruntime.a"
    libraries = static_libraries(build)
    print(f"Merging {len(libraries)} static libraries", flush=True)
    merge(libraries, args.output / "lib" / library_name)

    include = args.output / "include"
    shutil.rmtree(include, ignore_errors=True)
    shutil.copytree(source / "include" / "onnxruntime" / "core" / "session", include)
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        shutil.copy2(source / name, args.output / name)
    (args.output / "VERSION").write_text(args.version + "\n")
    print(f"Installed {args.output}")


if __name__ == "__main__":
    main()
