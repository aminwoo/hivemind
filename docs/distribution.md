# Downloads and compatibility

Download complete bundles from [GitHub Releases](https://github.com/aminwoo/hivemind/releases/latest).
Each ZIP includes the engine, the pinned twin-s-clock network, runtime libraries,
and licenses. Python and a separate network download are unnecessary.

| Bundle | Requirements | Download / first launch |
|---|---|---|
| Windows ONNX Runtime | Windows 10/11 x86-64; Microsoft Visual C++ 2015–2022 x64 Redistributable | About 9 MB; easiest starting point |
| Linux ONNX Runtime | Ubuntu 24.04 x86-64 or newer; system C/C++ runtime | About 15 MB; easiest starting point |
| macOS ONNX Runtime | Apple silicon M1 or newer; macOS 14+ | About 15 MB; Core ML defaults to Neural Engine |
| Windows TensorRT | Windows requirements above, plus NVIDIA requirements below | About 2 GB; first launch builds a GPU-specific cache |
| Linux TensorRT | Linux requirements above, plus NVIDIA requirements below | About 2 GB; first launch builds a GPU-specific cache |

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

Download the ZIP and its matching `.zip.sha256` file into the same folder.
On Linux:

```bash
sha256sum -c hivemind-v3.0.1-linux-x86_64-onnxruntime.zip.sha256
```

On macOS:

```bash
shasum -a 256 -c hivemind-v3.0.1-macos-arm64-onnxruntime.zip.sha256
```

On Windows PowerShell, compare the hash to the first value in the checksum file:

```powershell
(Get-FileHash .\hivemind-v3.0.1-windows-x86_64-onnxruntime.zip -Algorithm SHA256).Hash
Get-Content .\hivemind-v3.0.1-windows-x86_64-onnxruntime.zip.sha256
```

## Start the engine

Extract the entire ZIP into a writable directory, preserving its folders.
Select `bin\hivemind.exe` on Windows or the top-level `hivemind` launcher on
Linux/macOS in a bughouse-compatible GUI. A standard chess GUI may not
understand Hivemind's two-board positions and joint moves.

Running `./hivemind` or `.\bin\hivemind.exe` in a terminal opens the UCI
protocol, not a graphical board. It waits for commands such as `uci` and
`isready`; use `quit` to exit. The bundled network is found automatically,
including when a GUI starts the engine from a different working directory.

NVIDIA bundles take several minutes on first launch to build a TensorRT plan
beside the model. Let initialization finish; subsequent launches reuse the
cache. Allow at least 6 GB of free space for the ZIP, extracted libraries,
and cache (larger models can need more). Plans depend on the GPU and software
versions, so share the ZIP rather than a cache built on another machine.

## macOS launch

Current official bundles are not Developer ID signed or notarized. After
verifying the downloaded ZIP, extract it, open Terminal in the extracted
bundle folder, and if Gatekeeper blocks execution run:

```bash
xattr -dr com.apple.quarantine .
./hivemind
```

The launcher requests the Neural Engine through Core ML. `HIVEMIND_COREML=gpu`
selects GPU, `all` allows both, and `off` forces CPU. The engine reports its
selected provider and warns if Core ML initialization falls back to CPU.
Hosted macOS CI checks CPU execution; it does not establish physical Neural
Engine coverage. Check the release's validation section for hardware results.

## Release process

1. Update the CMake version, packaging defaults, workflow defaults, and
   `docs/releases/vVERSION.md`. Use conventional commit titles.
2. Push the matching `vVERSION` tag. CI verifies the pinned network's SHA-256,
   builds all five bundles on pinned OS baselines, checks archives, and completes
   two 200-node searches with each portable bundle from a different directory.
3. CI stages assets in a **draft** release. Hosted TensorRT jobs check startup
   and archive integrity; they have no NVIDIA GPU and cannot validate inference.
4. Download the draft's ZIP and checksum on physical hardware and run:

   ```bash
   python engine/scripts/smoke_release.py hivemind-v3.0.1-linux-x86_64-tensorrt.zip \
     --backend tensorrt --timeout 1800 --report validation-linux-nvidia.json
   python engine/scripts/smoke_release.py hivemind-v3.0.1-macos-arm64-onnxruntime.zip \
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
python engine/scripts/package_portable_release.py --model twin-s-clock.onnx \
  --macos-sign-identity 'Developer ID Application: YOUR IDENTITY' \
  --macos-notary-profile YOUR_KEYCHAIN_PROFILE
```

Import your Developer ID certificate into the keychain and store notarization
credentials using `xcrun notarytool store-credentials` first. The packager signs
the executable and dylibs with hardened runtime, verifies the signatures, and
waits for Apple's notarization result before producing the final checksum.
The hosted workflow currently builds unsigned bundles because signing
credentials are not configured.
