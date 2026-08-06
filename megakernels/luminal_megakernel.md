# Luminal Megakernel

- Fundamental limitations to inference:
  - compute(flops)
  - comms/bandwidth(TB/s)
- Expensive to increase since need to buy new hardware.
- Anytime GPU is 
  - not loading data = wasting bandwidth
  - not running computations = wasting compute

## Bottlenecks

![Typical transformer forward pass](https://blog.luminal.com/p/compiling-models-to-megakernels) 

## Main problems:

- every time we finish a kernel and start a next one, GPU sits idle while CPU schedules kernel
- Some SMs in GPU finish their work early and sit idle with other SMs continue i.e lots of stragglers.

- Kernel launch overhead can be mitigated by [CUDA graphs], but it's not perfect.
- **Wave Quantization**: occurs when a kernel's work cannot be equally distributed across all SMs
  - leaves straggler SMs
  - can lead to significant delays

- For tensor ops, we don't need to wait for full sync to begin next op. Eg: For tiled matmul, so long as row of A and col of B is ready, the first element of C(i.e elem 0,0) can be computed. Don't need to wait for full A and B to be ready.


- Third bottleneck:
  - kernel does no compute until it loads enough weights to start working. This means, even if kernel can do perfect load-compute overlapping during main loop exec, it cannot skip idle time waiting for initial weights to load. 
    - Techniques like [Programatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) help address this by letting second kernel start loading weights while first kernel is running compute, but this is done at device level, not SM level. LOTS OF BUBBLES LEFT!

## Key idea: One kernel per model

- Fuse every op in a forward pass into a single kernel. Advatanges:
  - kernel launch latency eliminated, since we only launch a single kernel
  - immediately start running work from the next op on SMs that have early-finished work, removing wave quant effect
  - Start loading weights for SMs next op during the epilogue of the current op, thereby eleminating gaps between compute spans.

## Megakernel = Interpreter on a GPU

- Interpreter: reads, decodes and runs instructions one-by-one.
- GPU is a large mutli-core processor, but each core is capable of executing a very specific and narrow instruction set.
- provide instructions either in:
  - core-specific instructions on a shared-memory
  - or, global memory in a global instruction stream.

- Need to decide if we statically schedule instructions to independent streams per SM or a single stream all SMs share.

- For each path
  - **Static scheduling**: 
    - benefits from being able to prefetch and load many inst at a time, directly into shared mem.
    - overhead of fetching new instruciton is low since it's already fetched at execution time and resides in fast memory.
    - downside: 
      - requires compiler to statically partition instructions across SMs ahead of time, which is difficult since instructions have variable latency.
      - jitter also causes hardware to somtimes run slower for unpredictable hardware reasons.
  - **Dynamic(global) scheduling**:
    - More overhead bcoz need roundtrip to global memory and atomic local to fetch each instruction.
      - can be hidden by doing the fetch during the execution of the previous instruction.
    - Does not require programmer/compiler to partition instructions AOT, and instead allows SMs to opportunistically pop off instructions from the queue once they're ready. 
      - naturally correct jitter, since faster SMs will pickup slack from slower SMs.

- tradeoffs are worth pursuing Dynamic scheduling. Global memory comms mean that we still do usual fusion patterns as in traditional kernels.
![Image](https://substackcdn.com/image/fetch/$s_!w6rx!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91112858-cba1-4a8f-8b39-d0ffa6218953_1776x916.jpeg)

- overlap between current running instructions and next instruction after adding megakernels! 
- One main problem not addressed: **synchronisation**
  - normal kernels have downside that future work cannot be ran until _all_ SMs finish on the current kernel.
  - however, the result of this is that all data is readily available at the start of next kernel.
  - With megakernels, once we start running future ops before past ops are finished, this guarantee goes us, and we need to now synchronzie and assert if the input data is ready for the next op, before running it.
  - generally, use standard barrier counter to do this.
  - unlike Hazyresearch however, luminal uses incremenet-then-decrement barrier. When op has started, they increment assigned barriw at launch, run and then decrement barrier once completed.
  - each barrier is "inflight producer" counter i.e we don't need the consuder to know how many producers to wait for a given piece of data. just need to wait till number of inflight producers = 0

## Generating Megakernels

- Luminal is a graph-based compiler, represents models as compute graphs.
- Main challenge: Transform compute graph into a instruction queue, with fine-grained data deps wired up correctly.

- Two steps:
  - rewrite existing ops into block ops, partitioned over SMs, with strided input and output data deps
  - deriving barrier strides given all present input-output op pairrings.

- First step is _easy_: Given an op(Matmul), we can easily translate it to TileMatmul to handle a tile of data at a time.
  - use shape-layout algebra(CuTE) instead a e-graph(egglog) to derive correct strides for each input and output tiles
- once we have partitioned ops, derive barriers each op should consume from(check = 0 before running) and produce(increment and decrement)


## Example: tiled matmul    

![Tiled Matmul](https://substackcdn.com/image/fetch/$s_!yrRt!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa4d6943-3e6e-47c0-915a-45e9731fcd69_1178x484.png)

- Assume M=N=K=128 and tile_size=32x32
- Launch grid of (128/32)x(128/32)=4x4 = 16 tile matmul instances to cover C
- Need to figure out: expression that would map launch index(0-15) to a barrier index for source C.
  - look at producer of A's launch dimensions.
  - If same size along M, we can prove independence along that dim, since we only consume one tile's worth of data along M
  - therefore, along M we initialize 128/32=4 barriers, and use a stride of 1 to specify that as we launch down this dim, we want to step our barriers by 1.
  - Along K dim, we are always consuming the whole dimension(whether row of A or col of B), so our stride should be 0.
  - THEREFORE, our final A barrier stride would be $m*1 + n*0$, or, flattened along a single launch axis, it would be $ \frac{x}{4} * 1 + (x \bmod 4) * 0 = \frac{x}{4} $, which maps our launch index (0-15) to our barrier(0-3) we want to consume from.

- Preserve as much independence as possible between launch divisions.
- If every producer SM and every consumer SM need to share a single barrier, this is full-sync of traditional kernels(i.e worst case)
- Best case, we have full independence where each next op depends on only one previous op, and can immediately launch when an SM completes.

- Everything compiles down to a single struct that looks as follows:

```zig
struct BlockOp {
  src_a_data: Expression,
  src_b_data: Expression,
  src_a_barrier: Expression,
  src_a_barrier: Expression,
  dest_data: Expression,
  dest_barrier: Expression,
}
```

- each expression defines a stride mapping the logical launch index to a physical index.
- each op now knows where to 
  - get source data
  - barriers to look at before running
  - where to write destination data
  - which barrier to increment/decrement

- next step is to generate op implementations for all these ops, from each block-op's definitions. standard implementation takes the form:

```c

__device__ void mk_op(
  OpPayload payload, //op-specific payload struct containing metadata
  const float* const source_ptrs[3], // source data pointers resolved by the interpreter
  float* out_ptr, // dest data pointer resolved by the interpreter
  const int current, // the current logical launch index of this op
  int t // the current thread index in this threadblock
) {
  // body
}
```

- interpreter resolves the data pointers and barriers, waits on barriers, and passes in data pointers to our implementation function.
- can also create payload structs and place them in the instruction queue to be passed to the implementation.
- also have metadata, such as runtime dims or pointers to special data stores like external KV caches.

## Symbolic Work Queues

- how to handle rebuilding work queues(instruction queues) in between executions?
- reallocating and rescheduling every op on a queue before each and every execution can be large and become a bottleneck
- some queues can be cached for multiple runs, but we don't want to worry about the costly process of re-allocating and rebuilding queues every time something as simple as a sequence length changes.

- Luminal solution: represent instructions in the work queue, rahter than instruction instances. Instead of N instances of a instruction, keep it once it in thequeue, init a running counter of how many remaining instruction instances we need to launch for the given instruction before moving the program counter. Auto decrement when an SM pops an increment off the queue.

## Conclusion

- trad kernels cause bubbles through kernel launch overhead, wave quant, and inter-instruction memory bubbles.
- fusing into megakernel can help overcome all 3 challenges.
- generate megakernels through a multi-stage process of rewriting an op to be partitioned over SMs, deriving data and barrier strides, and generating an interpreter by inlining each op's implementation functions. Then, we visit each op in the graph again to build the work queue, and bring the queue and interpreter together to execute.

