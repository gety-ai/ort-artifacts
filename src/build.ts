import { exists } from '@std/fs';
import { join } from '@std/path';
import { TarStream, TarStreamInput } from '@std/tar';

import { cpus, arch as getArch, platform as getPlatform } from 'node:os';

import { Command, EnumType, ValidationError } from '@cliffy/command';
import $ from '@david/dax';

import { Compressor } from './compressor/index.ts';

const arch = getArch() as 'x64' | 'arm64';
const platform = getPlatform() as 'win32' | 'darwin' | 'linux';

const TARGET_ARCHITECTURE_TYPE = new EnumType([ 'x86_64', 'aarch64' ]);

/**
 * CPU baseline selection — read this before raising the default.
 *
 * MLAS compiles separate SSE2/AVX/AVX2/AVX512 kernels with per-file compiler flags
 * (cmake/onnxruntime_mlas.cmake) and selects the fastest compatible one at runtime via CPUID
 * (core/mlas/lib/platform.cpp). Raising the global baseline does not make those kernels faster.
 * It only vectorizes the code that is not dispatched — Eigen operators, protobuf, STL, session
 * assembly — and it compiles FMA into MLAS's own generic fallback kernels. On a CPU with AVX but
 * no FMA3/AVX2 (Zhaoxin KX-5000/6000, Sandy/Ivy Bridge) that is an EXCEPTION_ILLEGAL_INSTRUCTION
 * on the first inference.
 *
 * See Linear DEV-666 / DEV-668. The x86_64 default must stay v1.
 */
type CpuBaseline = 'v1' | 'v2' | 'avx' | 'v3' | 'v4' | 'v8.0';

const CPU_BASELINES = {
	x86_64: [ 'v1', 'v2', 'avx', 'v3', 'v4' ],
	aarch64: [ 'v8.0' ]
} as const;

type TargetArchitecture = keyof typeof CPU_BASELINES;

const DEFAULT_CPU_BASELINE: Record<TargetArchitecture, CpuBaseline> = {
	x86_64: 'v1',
	aarch64: 'v8.0'
};

const ALL_CPU_BASELINES = new Set<CpuBaseline>(Object.values(CPU_BASELINES).flat() as CpuBaseline[]);

// Baselines that grant FMA to every translation unit, so the leak guard below cannot apply.
const FMA_ENABLED_BASELINES = new Set<CpuBaseline>([ 'v3', 'v4' ]);

// MLAS translation units carrying no per-file COMPILE_FLAGS: they get whatever the global
// baseline grants, which makes them the ones that fault on older CPUs.
const GENERIC_MLAS_TUS = [ 'logistic', 'tanh', 'erf', 'activate', 'compute' ];

// llvm-objdump, GNU objdump and dumpbin all emit `<address>: [bytes] <mnemonic> <operands>`.
// Anchoring on the mnemonic column keeps symbol names and disassembly comments from matching.
const FMA_MNEMONIC = /^\s*[0-9a-f`]+:\s+(?:[0-9a-f]{2}\s+)*v(?:fmadd|fmsub|fnmadd|fnmsub)/im;

// A disassembler can exit 0 having printed only a file header, which would let the FMA check
// pass without ever inspecting an instruction.
const ANY_MNEMONIC = /^\s*[0-9a-f`]+:\s+(?:[0-9a-f]{2}\s+)*[a-z][a-z0-9.]{2,}\b/im;

interface Disassembler {
	command: string;
	args: string[];
}

const DISASSEMBLERS: Disassembler[] = [
	{ command: 'llvm-objdump', args: [ '-d' ] },
	{ command: 'objdump', args: [ '-d' ] },
	...(platform === 'win32' ? [ { command: 'dumpbin', args: [ '/disasm:nobytes' ] } ] : [])
];

function getCpuBaselineCompilerFlags(targetArch: TargetArchitecture, cpuBaseline: CpuBaseline): string[] {
	if (targetArch !== 'x86_64') {
		return [];
	}

	if (platform === 'win32') {
		// x64 implies SSE2 and MSVC has no switch for the v2 level, so v1 and v2 emit nothing.
		switch (cpuBaseline) {
			case 'avx':
				return [ '/arch:AVX' ];
			case 'v3':
				return [ '/arch:AVX2' ];
			case 'v4':
				return [ '/arch:AVX512' ];
			default:
				return [];
		}
	}

	switch (cpuBaseline) {
		case 'v1':
			return [ '-march=x86-64' ];
		case 'v2':
			return [ '-march=x86-64-v2' ];
		// No psABI level covers AVX without FMA, and ORT's own USE_AVX emits a bare -mavx that
		// drops the v2 prerequisites.
		case 'avx':
			return [ '-march=x86-64-v2', '-mavx' ];
		case 'v3':
			return [ '-march=x86-64-v3' ];
		case 'v4':
			return [ '-march=x86-64-v4' ];
		default:
			return [];
	}
}

// Everything in CMAKE_{C,CXX}_FLAGS that can move the instruction-set floor; -D defines and
// warning switches are irrelevant to the baseline contract.
function getIsaFlags(flags: string): string[] {
	return flags.split(/\s+/).filter(flag =>
		/^\/arch:/i.test(flag) ||
		/^-m(?:arch|cpu)=/.test(flag) ||
		/^-m(?:no-)?(?:avx|fma|f16c|bmi|lzcnt|movbe|popcnt|sse|ssse|xsave)/.test(flag)
	);
}

// Spawning is the probe: a usage message with a non-zero exit still proves the tool exists,
// while a missing binary throws NotFound. Anything else — a denied --allow-run, a broken
// executable — is reported rather than mistaken for absence.
async function resolveDisassembler(): Promise<Disassembler | null> {
	for (const disassembler of DISASSEMBLERS) {
		try {
			await new Deno.Command(disassembler.command, { stdout: 'null', stderr: 'null' }).output();
			return disassembler;
		} catch (error) {
			if (error instanceof Deno.errors.NotFound) {
				continue;
			}

			throw new ValidationError(
				`could not execute ${disassembler.command}: ${error instanceof Error ? error.message : String(error)}; ` +
				'check its permissions and that Deno was granted --allow-run'
			);
		}
	}

	return null;
}

async function findGenericMlasObjects(buildRoot: string): Promise<Map<string, string[]>> {
	const extension = platform === 'win32' ? 'obj' : 'o';
	const found = new Map<string, string[]>(GENERIC_MLAS_TUS.map(tu => [ tu, [] ]));

	// Ninja and the Makefile generators name objects `<source>.cpp.o`; the Visual Studio and Xcode
	// generators drop the source extension. No other MLAS source shares these basenames.
	const wanted = new Map<string, string>();
	for (const tu of GENERIC_MLAS_TUS) {
		wanted.set(`${tu}.cpp.${extension}`, tu);
		wanted.set(`${tu}.${extension}`, tu);
	}

	// CMake mirrors the source's absolute path under the target directory, and `--static` nests
	// the whole tree one level deeper, so neither layout can be addressed by a fixed path.
	async function walk(folder: string, inMlasTarget: boolean): Promise<void> {
		for await (const entry of Deno.readDir(folder)) {
			const path = join(folder, entry.name);
			if (entry.isDirectory) {
				await walk(path, inMlasTarget || entry.name === 'onnxruntime_mlas.dir');
				continue;
			}

			const tu = inMlasTarget ? wanted.get(entry.name) : undefined;
			if (tu) {
				found.get(tu)!.push(path);
			}
		}
	}

	await walk(buildRoot, false);
	return found;
}

async function assertNoFmaInGenericMlas(
	buildRoot: string,
	disassembler: Disassembler,
	cpuBaseline: CpuBaseline
): Promise<void> {
	const objects = await findGenericMlasObjects(buildRoot);
	const decoder = new TextDecoder();

	for (const [ tu, paths ] of objects) {
		if (paths.length === 0) {
			// Fail closed: a translation unit quietly dropping out of the guard is exactly how
			// DEV-666 would come back unnoticed.
			throw new Error(
				`no object found for generic MLAS TU ${tu}.cpp under ${buildRoot}, so the CPU baseline ` +
				'guard cannot vouch for this build; if upstream renamed or moved it, update GENERIC_MLAS_TUS'
			);
		}

		for (const path of paths) {
			const result = await new Deno.Command(disassembler.command, {
				args: [ ...disassembler.args, path ],
				stdout: 'piped',
				stderr: 'piped'
			}).output();
			if (!result.success) {
				throw new Error(`${disassembler.command} failed on ${path}: ${decoder.decode(result.stderr).trim()}`);
			}

			const disassembly = decoder.decode(result.stdout);
			if (!ANY_MNEMONIC.test(disassembly)) {
				throw new Error(`${disassembler.command} decoded no instructions from ${path}`);
			}

			if (FMA_MNEMONIC.test(disassembly)) {
				throw new Error(
					`FMA instructions leaked into the generic MLAS TU ${tu}.cpp at --cpu-baseline=${cpuBaseline} ` +
					`(${path}). This is the DEV-666 crash signature: the artifact would fault with ` +
					'EXCEPTION_ILLEGAL_INSTRUCTION on the first inference on any CPU without FMA3. ' +
					`Something re-introduced /arch:AVX2, -mavx2 or -mfma into the global compile flags; ` +
					`check CMAKE_CXX_FLAGS in ${join(buildRoot, 'CMakeCache.txt')}.`
				);
			}
		}
	}
}

class CompressorStream extends TransformStream<Uint8Array<ArrayBuffer>, Uint8Array<ArrayBuffer>> {
	#compressor = new Compressor();

	constructor() {
		super({
			transform: (chunk, controller) => {
				const res = this.#compressor.push(chunk);
				if (res.byteLength) {
					controller.enqueue(res);
				}
			},
			flush: controller => {
				const res = this.#compressor.flush();
				if (res.byteLength) {
					controller.enqueue(res);
				}
			}
		});
	}
}

const CUDA_ARCHIVES: Record<number, Record<'win32' | 'linux', Record<'cudnn' | 'trt', string>>> = {
	12: {
		linux: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn_jit/linux-x86_64/cudnn_jit-linux-x86_64-9.23.2.1_cuda12-archive.tar.xz',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz'
		},
		win32: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.23.2.1_cuda12-archive.zip',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/zip/TensorRT-10.15.1.29.Windows.amd64.cuda-12.9.zip'
		}
	},
	13: {
		linux: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn_jit/linux-x86_64/cudnn_jit-linux-x86_64-9.23.2.1_cuda13-archive.tar.xz',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-13.1.tar.gz'
		},
		win32: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.23.2.1_cuda13-archive.zip',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/zip/TensorRT-10.15.1.29.Windows.amd64.cuda-13.1.zip'
		}
	}
};
const OPENVINO_ARCHIVES: Record<'win32' | 'linux', string> = {
	linux: 'https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.0/linux/openvino_toolkit_ubuntu24_2026.0.0.20965.c6d6a13a886_x86_64.tgz',
	win32: 'https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.0/windows/openvino_toolkit_windows_2026.0.0.20965.c6d6a13a886_x86_64.zip'
};
const NVRTX_ARCHIVES: Record<'win32' | 'linux', string> = {
	linux: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.4/TensorRT-RTX-1.4.0.76-Linux-x86_64-cuda-13.2-Release-external.tar.gz',
	win32: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.4/TensorRT-RTX-1.4.0.76-Windows-amd64-cuda-13.2-Release-external.zip'
};

// CUDA 12.x ships CCCL 2.8, whose clusterlaunchcontrol PTX wrappers hand `long2` members to an
// "l" (64-bit) asm constraint. `long` is 32 bits under MSVC, so every TU that reaches libcu++ —
// ~120 of them in the CUDA EP — fails with "asm operand type size(4) does not match ... 'l'" as
// soon as an sm_100+ architecture is in the mix, because the offending branch is gated on
// `__CUDA_ARCH__ >= 1000`. CCCL 3.0 (bundled with CUDA 13) fixed this by switching to
// `longlong2`; apply the same rename in place. Unaffected on Linux, where `long` is 64 bits.
async function patchToolkitClusterLaunchControl(): Promise<void> {
	const cudaPath = Deno.env.get('CUDA_PATH');
	if (!cudaPath) {
		console.warn('CUDA_PATH is unset, skipping CCCL clusterlaunchcontrol fixup');
		return;
	}

	const header = join(cudaPath, 'include', 'cuda', '__ptx', 'instructions', 'generated', 'clusterlaunchcontrol.h');
	let source: string;
	try {
		source = await Deno.readTextFile(header);
	} catch {
		console.warn(`${header} not found, skipping CCCL clusterlaunchcontrol fixup`);
		return;
	}

	if (!source.includes('reinterpret_cast<long2*>')) {
		return;
	}

	await Deno.writeTextFile(header, source.replaceAll('reinterpret_cast<long2*>', 'reinterpret_cast<longlong2*>'));
	console.log(`patched ${header}`);
}

async function *makeTarInput(...folders: string[]): AsyncGenerator<TarStreamInput> {
	for (const folder of folders) {
		let entries: Deno.DirEntry[];
		try {
			entries = await Array.fromAsync(Deno.readDir(folder));
		} catch {
			continue;
		}

		for (const entry of entries) {
			if (!entry.isFile) {
				continue;
			}

			const path = join(folder, entry.name);
			const { size } = await Deno.stat(path);
			yield {
				type: 'file',
				path: entry.name,
				size,
				readable: (await Deno.open(path, { read: true })).readable
			};
		}
	}
}

await new Command()
	.name('ort-artifact')
	.version('0.1.0')
	.type('target-arch', TARGET_ARCHITECTURE_TYPE)
	.option('-v, --upstream-version <version:string>', 'Exact version of upstream package', { required: true })
	.option('-t, --training', 'Enable Training API')
	.option('-s, --static', 'Build static library')
	.option('--iphoneos', 'Target iOS / iPadOS')
	.option('--iphonesimulator', 'Target iOS / iPadOS simulator')
	.option('--android', 'Target Android')
	.option('--cuda <version:integer>', 'Enable CUDA EP', {
		value(value: number) {
			if (value !== 12 && value !== 13) {
				throw new ValidationError('--cuda must be either 12 or 13');
			}
			return value;
		}
	})
	.option('--trt', 'Enable TensorRT EP', { depends: [ 'cuda' ] })
	.option('--nvrtx', 'Enable NV TensorRT RTX EP')
	.option('--directml', 'Enable DirectML EP')
	.option('--coreml', 'Enable CoreML EP')
	.option('--dnnl', 'Enable DNNL EP')
	.option('--xnnpack', 'Enable XNNPACK EP')
	.option('--webgpu', 'Enable WebGPU EP')
	.option('--openvino', 'Enable OpenVINO EP')
	.option('--nnapi', 'Enable NNAPI EP')
	.option('-N, --ninja', 'build with ninja')
	.option('--vs2026', 'Use Visual Studio 2026 generator')
	.option('--debug', 'Build with Debug config instead of Release')
	.option('-A, --arch <arch:target-arch>', 'Configure target architecture for cross-compile', { default: 'x86_64' })
	.option('--cpu-baseline <level:string>', 'Minimum CPU instruction set the artifact requires (x86_64: v1|v2|avx|v3|v4, default v1; aarch64: v8.0)', {
		value(value: string): CpuBaseline {
			if (!ALL_CPU_BASELINES.has(value as CpuBaseline)) {
				throw new ValidationError(
					`unsupported CPU baseline '${value}'; expected one of: ${[ ...ALL_CPU_BASELINES ].join(', ')}`
				);
			}

			return value as CpuBaseline;
		}
	})
	.action(async (options, ..._) => {
		const root = Deno.cwd();

		// Everything past this block clones, resets and cleans onnxruntime/, so validation has to
		// come first: a typo must not cost the caller their checkout.
		const targetArch = options.arch as TargetArchitecture;
		const cpuBaseline = options.cpuBaseline ?? DEFAULT_CPU_BASELINE[targetArch];
		const allowedBaselines: readonly CpuBaseline[] = CPU_BASELINES[targetArch];
		if (!allowedBaselines.includes(cpuBaseline)) {
			throw new ValidationError(
				`--cpu-baseline=${cpuBaseline} is invalid for --arch=${targetArch}; ` +
				`expected one of: ${allowedBaselines.join(', ')}`
			);
		}

		if (targetArch === 'x86_64' && platform === 'win32' && cpuBaseline === 'v2') {
			console.warn('MSVC has no flag for the x86-64-v2 level; building with the v1 (SSE2) baseline instead.');
		}

		// Resolved up front so a missing toolchain fails in seconds rather than after the build.
		const needsFmaGuard = targetArch === 'x86_64' && !FMA_ENABLED_BASELINES.has(cpuBaseline);
		const disassembler = needsFmaGuard ? await resolveDisassembler() : null;
		if (needsFmaGuard && !disassembler) {
			throw new ValidationError(
				`--cpu-baseline=${cpuBaseline} needs a disassembler to verify that no FMA instructions leaked ` +
				`into the generic MLAS kernels; put one of these on PATH: ${DISASSEMBLERS.map(d => d.command).join(', ')}`
			);
		}

		const onnxruntimeRoot = join(root, 'onnxruntime');
		const isExists = await exists(onnxruntimeRoot)
		let isBranchCorrect = false;
		if (isExists) {
			$.cd(onnxruntimeRoot);
			const currentBranch = (await $`git branch --show-current`.stdout("piped")).stdout.trim()
			isBranchCorrect = currentBranch === `rel-${options.upstreamVersion}`;
			$.cd(root);

			if (!isBranchCorrect) {
				console.log(`Removing onnxruntime directory because branch is incorrect: ${onnxruntimeRoot}, current branch: ${currentBranch}, expected branch: rel-${options.upstreamVersion}`);
				await Deno.remove(onnxruntimeRoot, { recursive: true });
			}
		}
		if (!isExists || !isBranchCorrect) {
			await $`git clone https://github.com/microsoft/onnxruntime --recursive --single-branch --depth 1 --branch rel-${options.upstreamVersion}`;
		}

		$.cd(onnxruntimeRoot);

		await $`git reset --hard HEAD`;
		await $`git clean -fdx`;

		const patchDir = join(root, 'src', 'patches', 'all');
		for await (const patchFile of Deno.readDir(patchDir)) {
			if (!patchFile.isFile) {
				continue;
			}

			await $`git apply ${join(patchDir, patchFile.name)} --ignore-whitespace --recount --verbose`;
			console.log(`applied ${patchFile.name}`);
		}

		const env = { ...Deno.env.toObject() };
		const args = [];
		const compilerFlags = [];
		const cudaFlags: string[] = [];

		const cudaArchives = options.cuda ? CUDA_ARCHIVES[options.cuda][platform as 'win32' | 'linux'] : null;

		if (platform === 'linux' && !options.android) {
			if (options.cuda === 12) {
				// nvcc only accepts clang up to 19 on CUDA 12.x. setup-clang symlinks cc/c++ to
				// clang-21, so gcc has to be named explicitly to keep cmake off of those.
				env.CC = 'gcc';
				env.CXX = 'g++';
				cudaFlags.push('-ccbin', 'g++');
			} else {
				env.CC = 'clang-21';
				env.CXX = 'clang++-21';
				if (options.cuda) {
					cudaFlags.push('-ccbin', 'clang++-21');
				}
			}
		} else if (platform === 'win32') {
			args.push('-G', options.vs2026 ? 'Visual Studio 18 2026' : 'Visual Studio 17 2022');
			if (options.arch === 'x86_64') {
				args.push('-A', 'x64');
			}
		}

		// Build for iOS on macOS.
		if (platform === 'darwin' && (options.iphoneos || options.iphonesimulator)) {
			args.push(`-DCMAKE_OSX_DEPLOYMENT_TARGET=${Deno.env.get("IPHONEOS_DEPLOYMENT_TARGET")}`)
			args.push('-DCMAKE_TOOLCHAIN_FILE=../cmake/onnxruntime_ios.toolchain.cmake');
			if(options.iphoneos) {
				args.push('-DCMAKE_OSX_SYSROOT=iphoneos');
			} else {
				args.push('-DCMAKE_OSX_SYSROOT=iphonesimulator');
			}
		}

		// Build for Android on Linux.
		if (platform === 'linux' && options.android) {
			// ANDROID_NDK_HOME and ANDROID_SDK_ROOT are expected to be set in the environment.
			args.push(`-DANDROID_PLATFORM=android-${Deno.env.get("ANDROID_API")}`);
			args.push('-DANDROID_ABI=arm64-v8a');
			args.push('-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false');
			args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(Deno.env.get('ANDROID_NDK_HOME')!, 'build', 'cmake', 'android.toolchain.cmake')}`);
		}

		if (options.cuda) {
			args.push('-Donnxruntime_USE_CUDA=ON');
			// https://github.com/microsoft/onnxruntime/pull/20768
			args.push('-Donnxruntime_NVCC_THREADS=1');

			const cudnnOutPath = join(root, 'cudnn');
			let should_skip = await exists(cudnnOutPath);
			if (should_skip) {
				// Check dir whether is empty
				const files = await Array.fromAsync(Deno.readDir(cudnnOutPath));
				if (files.length === 0) {
					await $`rm -rf ${cudnnOutPath}`;
					should_skip = false;
				}
			}

			if (!should_skip) {
				const cudnnArchiveStream = await fetch(cudaArchives!.cudnn).then(c => c.body!);
				await Deno.mkdir(cudnnOutPath);
				await $`tar xvJC ${cudnnOutPath} --strip-components=1 -f -`.stdin(cudnnArchiveStream);
			}

			args.push(`-Donnxruntime_CUDNN_HOME=${cudnnOutPath}`);

			if (platform === 'win32') {
				// nvcc < 12.4 throws an error with VS 17.10
				cudaFlags.push('-allow-unsupported-compiler');
				if (options.cuda === 13) {
					compilerFlags.push('-DCUDA_VECTOR_TYPE_ALIGNMENT_16_32_ENABLED=1');
				} else {
					await patchToolkitClusterLaunchControl();
				}
			}
		}

		if (options.cuda || options.trt || options.nvrtx) {
			args.push('-Donnxruntime_USE_FPA_INTB_GEMM=OFF');
			args.push('-Donnxruntime_USE_FLASH_ATTENTION=OFF');
			args.push('-Donnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF');
			args.push('-Donnxruntime_USE_FP8_KV_CACHE=OFF');
			args.push('-Donnxruntime_QUICK_BUILD=ON');

			// 75/80/90a = Turing/Ampere/Hopper; 120 adds Blackwell (RTX 50xx, sm_120)
			args.push('-DCMAKE_CUDA_ARCHITECTURES=75;80;90;120');
			cudaFlags.push('-compress-mode=size');
		}

		if (options.trt) {
			args.push('-Donnxruntime_USE_TENSORRT=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
		}
		if (options.nvrtx) {
			args.push('-Donnxruntime_USE_NV=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
		}

		if (options.trt) {
			const trtArchiveStream = await fetch(cudaArchives!.trt).then(c => c.body!);
			const trtOutPath = join(root, 'tensorrt');
			await Deno.mkdir(trtOutPath);
			await $`tar xvzC ${trtOutPath} --strip-components=1 -f -`.stdin(trtArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_HOME=${trtOutPath}`);
		}
		if (options.nvrtx) {
			const trtxArchiveStream = await fetch(NVRTX_ARCHIVES[platform as 'linux' | 'win32']).then(c => c.body!);
			const trtxOutPath = join(root, 'nvrtx');
			await Deno.mkdir(trtxOutPath);
			await $`tar xvzC ${trtxOutPath} --strip-components=1 -f -`.stdin(trtxArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_RTX_HOME=${trtxOutPath}`);
		}

		if (platform === 'win32' && options.directml) {
			args.push('-Donnxruntime_USE_DML=ON');
		}
		if (platform === 'darwin' && options.coreml) {
			args.push('-Donnxruntime_USE_COREML=ON');
		}
		if (options.webgpu) {
			args.push('-Donnxruntime_USE_WEBGPU=ON');
			args.push('-Donnxruntime_BUILD_WEBGPU_EP_STATIC_LIB=ON');
			args.push('-Donnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS=OFF');
			args.push('-Donnxruntime_USE_EXTERNAL_DAWN=OFF');
			args.push('-Donnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON');
			args.push('-Donnxruntime_WGSL_TEMPLATE=static');
		}
		if (options.dnnl) {
			args.push('-Donnxruntime_USE_DNNL=ON');
		}
		if (options.xnnpack) {
			args.push('-Donnxruntime_USE_XNNPACK=ON');
		}
		if (options.openvino) {
			const ovinoOutPath = join(root, 'openvino');
			const ovinoArchiveUrl = OPENVINO_ARCHIVES[platform as 'win32' | 'linux'];
			const ovinoArchiveStream = await fetch(ovinoArchiveUrl).then(c => c.body!);
			await Deno.mkdir(ovinoOutPath);
			if (platform === 'win32') {
				const tmpZip = join(root, 'openvino.zip');
				await Deno.writeFile(tmpZip, ovinoArchiveStream);
				await $`tar xvf ${tmpZip} -C ${ovinoOutPath} --strip-components=1`;
				await Deno.remove(tmpZip);
			} else {
				await $`tar xvzC ${ovinoOutPath} --strip-components=1 -f -`.stdin(ovinoArchiveStream);
			}

			args.push(`-DCMAKE_PREFIX_PATH=${join(ovinoOutPath, 'runtime', 'cmake')}`);
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
			args.push('-Donnxruntime_USE_OPENVINO=ON');
			args.push('-Donnxruntime_USE_OPENVINO_CPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_GPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_NPU=ON');
			// args.push('-Donnxruntime_USE_OPENVINO_INTERFACE=ON');
		}
		if(options.nnapi) {
			args.push('-Donnxruntime_USE_NNAPI_BUILTIN=ON');
		}

		if (platform === 'darwin') {
			if (options.arch === 'aarch64') {
				args.push('-DCMAKE_OSX_ARCHITECTURES=arm64');
			} else {
				args.push('-DCMAKE_OSX_ARCHITECTURES=x86_64');
			}
		} else {
			if (options.arch === 'aarch64' && arch !== 'arm64') {
				args.push('-Donnxruntime_CROSS_COMPILING=ON');
				switch (platform) {
					case 'win32':
						args.push('-A', 'ARM64');
						compilerFlags.push('-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS');
						break;
					case 'linux':
						if (!options.android) {
							args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(root, 'toolchains', 'aarch64-unknown-linux-gnu.cmake')}`);
						}
						break;
				}
			}
		}

		if (options.training) {
			args.push('-Donnxruntime_ENABLE_TRAINING=ON');
			args.push('-Donnxruntime_ENABLE_LAZY_TENSOR=OFF');
		}

		if (options.training) {
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
		}

		if (platform === 'win32' && !options.static) {
			args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dgtest_force_shared_crt=ON');
		}

		if (!options.static) {
			args.push('-Donnxruntime_BUILD_SHARED_LIB=ON');
		} else {
			if (platform === 'win32') {
				args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dgtest_force_shared_crt=ON');
				args.push('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL');
			}
		}

		if (platform === 'win32') {
			compilerFlags.push('-D_CRT_SECURE_NO_WARNINGS');
			// https://github.com/microsoft/onnxruntime/pull/21005
			compilerFlags.push('-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR');
		}

		args.push('-Donnxruntime_BUILD_UNIT_TESTS=OFF');
		args.push(`-Donnxruntime_USE_KLEIDIAI=${options.arch === 'aarch64' ? 'ON' : 'OFF'}`);
		args.push('-Donnxruntime_CLIENT_PACKAGE_BUILD=ON');

		// ORT's global ISA switches only append compiler flags, and the ones they append cannot
		// express a psABI level (USE_AVX2 gives -mavx2, which is strictly narrower than
		// -march=x86-64-v3). Keep them off so the mapping above is the single source of truth.
		args.push('-Donnxruntime_USE_AVX=OFF');
		args.push('-Donnxruntime_USE_AVX2=OFF');
		args.push('-Donnxruntime_USE_AVX512=OFF');

		compilerFlags.push(...getCpuBaselineCompilerFlags(targetArch, cpuBaseline));

		if (compilerFlags.length > 0) {
			const allFlags = compilerFlags.join(' ');
			args.push(`-DCMAKE_C_FLAGS=${allFlags}`);
			args.push(`-DCMAKE_CXX_FLAGS=${allFlags}`);
		}

		if (options.ninja && !(platform === 'win32' && options.arch === 'aarch64')) {
			args.push('-G', 'Ninja');
		}

		if (cudaFlags.length) {
			args.push(`-DCMAKE_CUDA_FLAGS_INIT=${cudaFlags.join(' ')}`);
		}

		const sourceDir = options.static ? join(root, 'src', 'static-build') : 'cmake';
		const artifactOutDir = join(root, 'artifact', 'onnxruntime');

		const buildConfig = options.debug ? 'Debug' : 'Release';
		await $`cmake -S ${sourceDir} -B build -D CMAKE_BUILD_TYPE=${buildConfig} -DCMAKE_CONFIGURATION_TYPES=${buildConfig} -DCMAKE_INSTALL_PREFIX=${artifactOutDir} -DONNXRUNTIME_SOURCE_DIR=${onnxruntimeRoot} --compile-no-warning-as-error ${args}`
			.env(env);

		const cachePath = join(onnxruntimeRoot, 'build', 'CMakeCache.txt');
		const cacheLines = (await Deno.readTextFile(cachePath)).split(/\r?\n/);
		const readCacheEntry = (name: string, type: string): string => {
			const prefix = `${name}:${type}=`;
			const line = cacheLines.find(line => line.startsWith(prefix));
			if (line === undefined) {
				throw new Error(`${prefix.slice(0, -1)} is missing from ${cachePath}`);
			}

			return line.slice(prefix.length).trim();
		};

		for (const option of [ 'onnxruntime_USE_AVX', 'onnxruntime_USE_AVX2', 'onnxruntime_USE_AVX512' ]) {
			const value = readCacheEntry(option, 'BOOL');
			if (value !== 'OFF') {
				throw new Error(`${option} is unexpectedly ${value} in ${cachePath}`);
			}
		}

		if (targetArch === 'x86_64') {
			const expected = getCpuBaselineCompilerFlags(targetArch, cpuBaseline);
			// A baseline that emits no ISA flags is normal (v1/v2 on MSVC), so name the empty case
			// rather than printing an empty string that reads as a formatting bug.
			const describe = (flags: string[]): string => flags.length ? `'${flags.join(' ')}'` : 'none';
			for (const name of [ 'CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS' ]) {
				const actual = getIsaFlags(readCacheEntry(name, 'STRING'));
				if (actual.join(' ') !== expected.join(' ')) {
					throw new Error(
						`${name} does not match --cpu-baseline=${cpuBaseline}; ` +
						`expected ISA flags ${describe(expected)}, got ${describe(actual)}`
					);
				}
			}
		}

		await $`cmake --build build --config ${buildConfig} --parallel ${cpus().length}`;

		if (disassembler) {
			await assertNoFmaInGenericMlas(join(onnxruntimeRoot, 'build'), disassembler, cpuBaseline);
		}

		await $`cmake --install build`;

		const artifactOut = await Deno.open(join(root, 'artifact.tar.lzma2'), { create: true, write: true });
		await ReadableStream.from(makeTarInput(join(artifactOutDir, 'lib'), join(artifactOutDir, 'bin')))
			.pipeThrough(new TarStream())
			.pipeThrough(new CompressorStream())
			.pipeTo(artifactOut.writable);
	})
	.parse(Deno.args);
