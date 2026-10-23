# Downloads and compatibility

Download from [GitHub Releases](https://github.com/aminwoo/hivemind/releases/latest).
The Windows, Linux and macOS downloads are single executables: the engine,
the pinned twin-s-clock network, ONNX Runtime and the license texts
(`--licenses`) are all inside. The NVIDIA downloads are ZIP bundles that also
carry TensorRT and the CUDA runtime. Python and a separate network download
are unnecessary.

| Download | Requirements | Size / first launch |
|---|---|---|
| Windows (`windows-x86_64.exe`) | Windows 10/11 x86-64 | About 35 MB; easiest starting point |
| Linux (`linux-x86_64`) | Ubuntu 24.04 x86-64 or newer (glibc 2.39+) | About 35 MB; easiest starting point |
| macOS (`macos-arm64`) | Apple silicon M1 or newer; macOS 14+ | About 35 MB; Core ML defaults to Neural Engine |
| Windows TensorRT (ZIP) | Windows 10/11 x86-64; Microsoft Visual C++ 2015–2022 x64 Redistributable; NVIDIA requirements below | About 2 GB; first launch builds a GPU-specific cache |
| Linux TensorRT (ZIP) | Ubuntu 24.04 x86-64 or newer; NVIDIA requirements below | About 2 GB; first launch builds a GPU-specific cache |

The release build and CPU smoke-test baselines are pinned to Ubuntu 24.04,
Windows Server 2022, and macOS 14 arm64. Windows 10/11 are runtime targets;
the hosted Windows check runs on Server 2022. Other Linux distributions may
work with compatible system runtimes, but are not covered by release CI.
macOS 14 is the deployment target; older macOS and Intel Macs are not covered.

NVIDIA bundles ship TensorRT 11.1.0.106 and CUDA Runtime 13.3.29. They target
Turing or newer GPUs (compute capability 7.5+, for example RTX 20 series and
newer), subject to NVIDIA's platform support matrix. Pascal/GTX 10 series and
older GPUs should use the CPU bundle. The CUDA 13.x minimum driver branch is
R580; use R610 or newer for the CUDA 13.3 baseline, especially if runtime
compilation requires newer driver features. Check your GPU and driver with
`nvidia-smi`. No separate CUDA Toolkit or TensorRT installation is needed.

See NVIDIA's [TensorRT 11.1 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.1.0.html),
[support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html),
and [CUDA 13.3 driver requirements](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html#cuda-driver).
The [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
is available from Microsoft.

## Verify your download

Download the file and its matching `.sha256` file into the same folder.
On Linux:

```bash
sha256sum -c hivemind-vX.Y.Z-linux-x86_64.sha256
```

On macOS:

```bash
shasum -a 256 -c hivemind-vX.Y.Z-macos-arm64.sha256
```

On Windows PowerShell, compare the hash to the first value in the checksum file:

```powershell
(Get-FileHash .\hivemind-vX.Y.Z-windows-x86_64.exe -Algorithm SHA256).Hash
Get-Content .\hivemind-vX.Y.Z-windows-x86_64.exe.sha256
```

The NVIDIA ZIPs are checked the same way with their `.zip.sha256` files.

## Start the engine

A single executable can live anywhere. On Linux and macOS, browsers drop the
executable bit, so run `chmod +x hivemind-*` once. Extract an NVIDIA ZIP
whole into a writable directory, preserving its folders, and use
`bin\hivemind.exe` on Windows or the top-level `hivemind` launcher on Linux.
Select the executable in a bughouse-compatible GUI; a standard chess GUI may
not understand Hivemind's two-board positions and joint moves.

Running the executable in a terminal opens the UCI protocol, not a graphical
board. It waits for commands such as `uci` and `isready`; use `quit` to exit.
The network is built in (NVIDIA bundles find theirs automatically), so it
works when a GUI starts the engine from any working directory; `--model`
loads a different network instead. `--licenses` prints the licenses of
Hivemind and the software it includes.

NVIDIA bundles take several minutes on first launch to build a TensorRT plan
beside the model. Let initialization finish; subsequent launches reuse the
cache. Allow at least 6 GB of free space for the ZIP, extracted libraries,
and cache (larger models can need more). Plans depend on the GPU and software
versions, so share the ZIP rather than a cache built on another machine.

## macOS launch

Current official downloads are not Developer ID signed or notarized. After
verifying the download, open Terminal in its folder, and if Gatekeeper blocks
execution run:

```bash
chmod +x hivemind-vX.Y.Z-macos-arm64
xattr -d com.apple.quarantine hivemind-vX.Y.Z-macos-arm64
./hivemind-vX.Y.Z-macos-arm64
```

The engine runs the network on the Neural Engine through Core ML. `HIVEMIND_COREML=gpu`
selects GPU, `all` allows both, and `off` forces CPU. The engine reports its
selected provider and warns if Core ML initialization falls back to CPU.
Hosted macOS CI checks CPU execution; it does not establish physical Neural
Engine coverage. Check the release's validation section for hardware results.

## Release process

1. Update the CMake version, packaging defaults, workflow defaults, and
   `docs/releases/vVERSION.md`. Use conventional commit titles.
2. Push the matching `vVERSION` tag. CI verifies the pinned network's SHA-256,
   builds the three single executables (against a static ONNX Runtime built
   from source by `engine/scripts/build_static_onnxruntime.py` and cached)
   and the two NVIDIA bundles on pinned OS baselines, checks checksums, and
   runs each single executable alone in an empty directory: `--licenses` and
   two 200-node searches on its built-in network.
3. CI stages assets in a **draft** release. Hosted TensorRT jobs check startup
   and archive integrity; they have no NVIDIA GPU and cannot validate inference.
4. Download the draft's ZIP and checksum on physical hardware and run:

   ```bash
   python engine/scripts/smoke_release.py hivemind-v3.0.2-linux-x86_64-tensorrt.zip \
     --backend tensorrt --timeout 1800 --report validation-linux-nvidia.json
   python engine/scripts/smoke_release.py hivemind-v3.0.2-macos-arm64 \
     --backend coreml --timeout 300 --report validation-macos-coreml.json
   ```

   On Windows use the Windows ZIP with `--backend tensorrt`. These checks
   require the expected provider, validate initialization and two searches,
   and reject CPU fallback for a Core ML check. A selected Core ML provider
   confirms that path ran; it does not prove every graph operation ran on ANE.
5. Attach hardware validation reports, record the GPU/OS/driver and any
   untested acceleration paths in the release notes, then publish the draft.

Signing is supported when Apple credentials are available on the packaging Mac:

```bash
python engine/scripts/package_single_file_release.py --model twin-s-clock.onnx \
  --macos-sign-identity 'Developer ID Application: YOUR IDENTITY' \
  --macos-notary-profile YOUR_KEYCHAIN_PROFILE
```

Import your Developer ID certificate into the keychain and store notarization
credentials using `xcrun notarytool store-credentials` first. The packager signs
the executable with hardened runtime and waits for Apple's notarization result
before producing the final checksum. Without an identity it signs ad hoc, as
Apple silicon requires. The hosted workflow currently builds unsigned
downloads because signing credentials are not configured.
