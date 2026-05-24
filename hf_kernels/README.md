# Huggingface Kernels

- Get and load a kernel in one line
- replace layers in model with a faster kernel.
- stardardise shipping of kernels, same as what docker did for software.

3 components:
- the builder -> building kernels
- the hub -> distributing kernels
- the library -> using kernels

- kernels have bespoke build pipelines and difficult to maintain.
- no stardard structure around how to write kernels, etc.

## Solution

- explore using **nix** to cache and repoduce binaries
- build once correctly, and use multiple times

## Requestments
- Reproducible(nix, dev env)
- consistent source structure(configuration, naming conventions)
- consistent build output(binary format, library to fetchm hub as registry).

- simple mechansim(python lib, reuse hub)
- work on arnge of hardware/torch versions(auto-select correct binary)
- integrate with existing code(drop-in replacement, non-constraining)

## Standardization; kernel code

- Key elements
    - build.toml: build config, don't specify steps, specify input, output, etc.
    - flake.nix: points to kernel builder, and run on this
    - torch-ext: binds hardware specific code to torch. allows to use torch tensors, etc.
    - example_kernel_metal:
        - has the hardware specific code.
- also has kernel skill to generate these kernels repo structure.

## Standardize: kernel binaries

- key elements:
    - torchX-backend-arch-os(variant)
    - variant/_ops.py
    - variant/metadata.json
    - variant/*.so(kernel binary)
- for every combo of python + cuda + os versino, AOT compilation can be expensive. juse use python ab3 and torch stable api support to reduce this.
- But, the current setup also supports JIT compilation.

- protect against malware using: co-signed code, load_remote_code=True explicit, etc.
- use new toolchain with nix packages.

## Standardiztion: fetching kernels

- fetch with kernel lib 
- get_kernel
- version specification

# Deepdive

## Resolution

### How does get_kernel work? How is correct kernel resolved at runtime?

- detect env by inspecting torch(backend, version, platform, ABI)
- builds filtered, priority-ordered fallback list: arch specific
- download only matching variants directories from hub(not entire repo)
- first existing match wins: tries each condidate in order and usese the first one found in build/
- validates python deps, then dynamically imports the module.
    - can assume deps for jitted kernels, so need to validate again

### How does the project handle new updates to existing kernels?

- use `version` identifier. looks for branch `v1` on hf. 
- version is incremented with each new release.
- version is only bumped to breaking changes to kernel, such as change to API that would require users to change code to use the new version
- other changes, such as perf, etc. does not version bump
- initlaly creater a full version system(vx.x.x), but would not properly tag and users wouldn't really use it. want as little versioning as possible. so use an integer.
- kernel dev should bymp up version if they break anything.
- can load multiple versions of the same kernel can be loaded into a process:
    - each kernel ahs a unique id
    - kernels register themselves in the torch namespace, and we cannot have kernels with the same functions in their op namespace
    - solution: every kernel has unique id: name of kernel, backend, git-shorthash: _flash_attn3_cuda_8213fs
        - needed backend since someone tried to load the same kernel on both CPU and GPU. This is a valid case, where a kernel might be faster on CPU for smaller layers and might be faster on GPU for larger layers.
    - directly load the kernel into the python module table using this unique id.

### Layers: does this require rewriting modelling code, or drop-in replacement?

- drop-in replacement
- kernel replacement
- exposed as nn.Module classes
- layers get a decorator with a name. a mapping maps names to kernel layers.
  - this is how transformers uses kernels

```py
mapping = {"SiluAndMul": {"cuda": LayerRepository(repo_id="huggingface_repo", layer_name="SiluAndMul", version=1}}

@use_kernel_forward_from_hub("SiluAndMul")
class SiluAndMul(nn.Module):
  def forward(self, a: torch.Tensor) -> torch.tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d] * x[..., d:])

model = SiluAndMul()

with use_kernel_maping.mapping():
  # when we run kernelize, all forward layers are replaced with the kernels
  model = kernelize(
    model, 
    mode=Mode.TRAINING | Model.TORCH_COMPILE,
    device="cuda"
  )
```

- can also refine mapping based on mode i.e one for training vs one for inference.
- kernel will only replace the forward layer.
- can modify state of function if kernel is aware of it.

### Is it possible to use kernels offline i.e fetch kernels before runtime?

- important for TGI which ships with docker.
- also important for quick deployments
- yes it's possible!
- via **locked kernels**
- locked kernels are fetched at build time and tracked in kernels.lock
- instead of using `get_kernel`, we can use `load_kernel`, which loads the pre-downloaded kernel and ensures it matches the contents of the lockfile.
- specified inside pyproject.toml 

```toml
[tool.kernels.dependencies]
"kernels-community/activations" = 1
```

### Build config: how to build kernels

- build.toml is config surface
- specify:
  - general (metadata)
  - torch (binding files)
  - target specific source
    - metal
    - cuda
- some kernels require addiotnal opts, possible in this file.

### Nix Configuration

- to deal with dep hell.
- nix is a build system
- builds are described in a pure, functinoal language
  - closest system is bazel
  - multiple runs give same 'build recipe'
- build recipes are build in a sandbox that only has exact deps specificed in the build recipe(almost reproducible)
  - sandbox is empty when starting, so all deps need to be explicit.
- this combo of nix helps tacking 'runs on my machine': fully vendor the dev env and build process in code.
- different from docker; not just image, but all internals deps(libc), etc. will be fixed. apt-get update will give different projects

### nix-related projects

- nixpkgs: package set defined in nix. pull compilers, python, cuda, etc. from this package set.

### nixos

- built on nixpkgs
- pure declaritve os based on nixpkgs.

### Benefits

- isolated build envs
- fully reprodicilble and cacheable
  - if nccl is built as part of a dep, it can be reused by the kernel builder later because everything is cached
- flexible and extensible(full control of compilation and deps)


### How to go from build to binary?

- each kernel is defined as a nix flake that fetches builder to compile to kernel
- nix build reads build.toml and creates reproducible 'build recipe'
- depending on config, build recipe specifies pinned ersions of all deps
- recipe build inside sandbox
  - cmake files, setup.py
  - kernel is built using ninja
  - after building the kernel is vetted: only python abi3/manylinues_2_28 symbols, kernels can be loaded with kernels lib, etc.
  - final outputs are placed in build dir, organised by variant(torch version x backedn x platform)
- summary: Nix -> build.toml -> kernel-builder -> CMake -> Ninja, -> output(variants)

### Kernel Repo

- New repo type: kernel repo
- push variants up to hub
