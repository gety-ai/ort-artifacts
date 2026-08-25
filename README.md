# ort-artifacts

Build pipeline for the pre-compiled [ONNX Runtime](https://onnxruntime.ai/) binaries consumed by the Rust [`ort`](https://github.com/pykeio/ort) crate.

`src/build.ts` is a Deno CLI that checks out an upstream ONNX Runtime release, applies the patches in `src/patches/all`, configures and builds it with CMake, and packs the result into `artifact.tar.lzma2`. `.github/workflows/build.yml` runs that CLI across a target matrix and uploads each result as `<target>[+<feature-set>].tar.lzma2`.

## Prerequisites

- [Deno](https://deno.land/) 2.x
- CMake 3.26 or newer
- Git
- A host C++ toolchain
  - Windows: Visual Studio 2022 or newer, with the MSVC x64/ARM64 build tools
  - Linux: GCC 11+ or Clang 12+
  - macOS: Xcode 15+ or the Command Line Tools
- LLVM (`llvm-objdump`) or binutils (`objdump`), used by the CPU baseline guard described below
- Optional: [Ninja](https://ninja-build.org/) for `-N`, the NVIDIA CUDA Toolkit for `--cuda`, the Android NDK or iOS SDK for mobile targets

## Building locally

```bash
# Static build with DirectML, Windows x86_64
deno run -A src/build.ts -v 1.28.1 -s --directml

# Static build with CoreML on Apple Silicon, using Ninja
deno run -A src/build.ts -v 1.28.1 -s -A aarch64 --coreml -N

# Static build with CUDA 12 and TensorRT, Linux x86_64
deno run -A src/build.ts -v 1.28.1 -s --cuda=12 --trt -N
```

The leading `-A` grants Deno permissions; the `-A aarch64` in the second example is the CLI's own `--arch` flag.

> [!WARNING]
> The CLI resets and cleans the `onnxruntime/` checkout (`git reset --hard`, `git clean -fdx`) before every build. Do not keep local work there.

## Flags

| Flag | Value | Description |
| :--- | :--- | :--- |
| `-v, --upstream-version` | string | **Required.** Exact upstream release to build, e.g. `1.28.1`. |
| `-A, --arch` | `x86_64` \| `aarch64` | Target architecture. Default `x86_64`. |
| `--cpu-baseline` | level | Minimum CPU instruction set the artifact requires. See below. |
| `-s, --static` | flag | Build a static library. |
| `-t, --training` | flag | Enable the Training API. |
| `-N, --ninja` | flag | Use the Ninja generator. |
| `--debug` | flag | Build the `Debug` configuration instead of `Release`. |
| `--vs2026` | flag | Use the Visual Studio 2026 generator. |
| `--android` | flag | Target Android. Requires `ANDROID_NDK_HOME` and `ANDROID_API`. |
| `--iphoneos` | flag | Target iOS / iPadOS. |
| `--iphonesimulator` | flag | Target the iOS / iPadOS simulator. |
| `--cuda` | `12` \| `13` | Enable the CUDA EP and download the matching cuDNN. |
| `--trt` | flag | Enable the TensorRT EP. Requires `--cuda`. |
| `--nvrtx` | flag | Enable the NVIDIA TensorRT RTX EP. |
| `--directml` | flag | Enable the DirectML EP. Windows only. |
| `--coreml` | flag | Enable the CoreML EP. macOS only. |
| `--webgpu` | flag | Enable the WebGPU EP. |
| `--openvino` | flag | Enable the OpenVINO EP and download the OpenVINO SDK. |
| `--dnnl` | flag | Enable the oneDNN EP. |
| `--xnnpack` | flag | Enable the XNNPACK EP. |
| `--nnapi` | flag | Enable the NNAPI EP. Android only. |

## CPU instruction set baseline

`--cpu-baseline` sets the minimum instruction set the produced artifact requires. It defaults to `v1` on x86_64 and `v8.0` on aarch64 — the most compatible value on each architecture.

> [!IMPORTANT]
> Do not raise `--cpu-baseline` expecting faster inference.
>
> ONNX Runtime's compute kernels live in MLAS, which compiles a separate SSE2, AVX, AVX2 and AVX512 kernel for every hot routine (`cmake/onnxruntime_mlas.cmake` applies the ISA flags per source file) and selects the fastest one the host supports at runtime via CPUID (`core/mlas/lib/platform.cpp`). Those kernels are already as wide as the running CPU allows, whatever the baseline is.
>
> Raising the baseline only vectorizes the code MLAS does *not* dispatch — Eigen operators, protobuf, the STL, session assembly — and it compiles FMA into MLAS's own generic fallback kernels. The artifact then faults with `EXCEPTION_ILLEGAL_INSTRUCTION` / `SIGILL` on the first inference on every CPU below the chosen level.

### Levels

| Level | Adds | Example CPUs |
| :--- | :--- | :--- |
| `v1` **(default)** | x86-64 baseline: SSE, SSE2, CMOV, FXSR | Every x86_64 CPU |
| `v2` | SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, LAHF/SAHF | Nehalem and newer, Bulldozer and newer |
| `avx` | `v2` plus AVX, without FMA or AVX2 | Sandy Bridge, Ivy Bridge, Zhaoxin KX-5000/6000 |
| `v3` | `v2` plus AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE | Haswell and newer, Zen and newer |
| `v4` | `v3` plus AVX512F, AVX512CD, AVX512BW, AVX512DQ, AVX512VL | Skylake-X and newer, Zen 4 and newer |

`v1`–`v4` are the [x86-64 psABI microarchitecture levels](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels), the same vocabulary as Go's `GOAMD64` and `-march=x86-64-vN`. `avx` is not a psABI level; it exists because AVX-without-FMA CPUs sit between `v2` and `v3` and have no standard name. There are no aliases — `sse2`, `avx2` and `native` are rejected.

On aarch64 the only accepted level is `v8.0`, which emits no architecture flags and leaves the toolchain and target ABI at their defaults.

> [!NOTE]
> MSVC has no switch for the `x86-64-v2` level. On Windows, `--cpu-baseline v2` builds as `v1` and prints a warning. MSVC's `/arch:AVX2` and `/arch:AVX512` are the closest available mappings for `v3` and `v4`, but they are not psABI-equivalent.

### Build-time guards

Two assertions run on every x86_64 build and fail it rather than shipping a broken artifact:

1. After CMake configures, `onnxruntime_USE_AVX`, `USE_AVX2` and `USE_AVX512` must be `OFF` in `CMakeCache.txt`, and the ISA flags in `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` must match the requested baseline exactly. Those three ORT options only append compiler flags, and the flags they append cannot express a psABI level, so `src/build.ts` emits `-march=` / `/arch:` itself and keeps them off.
2. When the baseline is below `v3`, the generic MLAS translation units (`logistic`, `tanh`, `erf`, `activate`, `compute` — the ones with no per-file ISA flags) are disassembled and must contain no FMA instructions. This is the DEV-666 crash signature, caught at build time.

The second guard needs `llvm-objdump`, `objdump` or `dumpbin` on `PATH`. Its absence is reported before the build starts, not after.

### Background: DEV-666 / DEV-668

Earlier builds forced `-Donnxruntime_USE_AVX2=ON` plus `/arch:AVX2` (MSVC) or `-march=x86-64-v3` (GCC/Clang) unconditionally. Because MSVC then defines `__AVX2__` in every translation unit, `mlasi.h` selected its FMA3 intrinsic path for the generic kernels, and `logistic.cpp.obj` shipped with ten `vfmadd` instructions. Users on Zhaoxin KX-6000 hardware crashed on the first CPU inference. `--cpu-baseline` replaces that hardcoded floor and defaults it back to the upstream-safe value.

## Release artifacts

Each matrix row uploads `artifact.tar.lzma2` under the name `<target>[+<feature-set>].tar.lzma2`, where the `+<feature-set>` suffix is omitted when the feature set is `none`.

| Target | Args | Feature set | Runner |
| :--- | :--- | :--- | :--- |
| `aarch64-unknown-linux-gnu` | `-A aarch64 -N` | `none` | `ubuntu-24.04` |
| `x86_64-unknown-linux-gnu` | `-N` | `none` | `ubuntu-24.04` |
| `aarch64-apple-darwin` | `--coreml -A aarch64 -N` | `coreml` | `macos-15` |
| `aarch64-pc-windows-msvc` | `-A aarch64 --directml` | `directml` | `windows-2022` |
| `x86_64-pc-windows-msvc` | `--directml` | `directml` | `windows-2022` |
| `x86_64-unknown-linux-gnu` | `--cuda=12 --trt -N` | `cuda12,tensorrt` | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | `--directml --cuda=12 --trt` | `cuda12,tensorrt,directml` | `windows-2022` |
| `x86_64-unknown-linux-gnu` | `--cuda=13 --trt --nvrtx -N` | `cuda13,tensorrt,nvrtx` | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | `--directml --cuda=13 --trt --nvrtx` | `cuda13,tensorrt,nvrtx,directml` | `windows-2022` |
| `x86_64-unknown-linux-gnu` | `--webgpu -N` | `webgpu` | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | `--webgpu --vs2026` | `webgpu` | `windows-2025` |
| `aarch64-apple-darwin` | `--coreml -A aarch64 -N --webgpu` | `coreml,webgpu` | `macos-15` |
| `aarch64-apple-ios` | `-A aarch64 --iphoneos --coreml -N` | `coreml` | `macos-15` |
| `aarch64-apple-ios-sim` | `-A aarch64 --iphonesimulator --coreml -N` | `coreml` | `macos-15` |
| `aarch64-linux-android` | `-A aarch64 --android --nnapi -N` | `nnapi` | `ubuntu-24.04` |
| `x86_64-unknown-linux-gnu` | `--nvrtx -N` | `nvrtx` | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | `--nvrtx --directml` | `nvrtx,directml` | `windows-2022` |

Every row builds at the default baseline and therefore carries no baseline segment in its filename, so downstream pinned URLs keep resolving. The archive contents change with every rebuild, so consumers still bump the sha256 they pin.

If a non-default baseline is ever shipped, the level **must** appear in `feature-set` so an artifact that cannot run everywhere is impossible to mistake for one that can:

```yaml
- target: x86_64-pc-windows-msvc
  args: "--directml --cpu-baseline v3"
  feature-set: v3,directml
  # -> x86_64-pc-windows-msvc+v3,directml.tar.lzma2
```
