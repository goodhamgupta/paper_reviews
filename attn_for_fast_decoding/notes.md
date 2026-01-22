# Hardware efficient attention for fast deocding

## Bottlenecks in LLM inference

- Sequenctial decoding
- KV cache
    - large batch/long-ctx exceed GPU memory
    - loading KV cache per later per doecoding step dominates latency
- Large Batch: Multi-step agent loop
- Long context: video
- Test-time compute

## Background: Inference Efficiency(pre-training)

- 2019: MQA -> Single head for K and V
- Reduces cost of MHA but at the price of model quality
- GQA
    - One KV head per device
    - Free perf boost over MQA
- MLA ~ MHA
    - Compresses hidden state into a single large latent with a big head dimension
    - This latent is used to reconstruct unique set of K and V heads
    - Mimics MHA during training
    - Mimics MQA in inference 
    - Suffers same issue as MQA: Duplicating latent is more costly

## Ideal inference attenttion design

- High quality
- Parallel friendly
    - moar GPUs mitigates KV cache bottleneck
- hardware efficient (maximize FLOPs/byte)

## Attention is memory-bound during decoding

- measure using arithmetic intensity
- H100: Compute to memory bandwidth is 295 FLOPs per byte. Naive attention is 2 FLOPS/2 bytes(when calcuating arithmetic intensity)
    - Large part of cores are left unused
- Arithmetic intensity for all attn vairants boils down to: how many query head share a single KV head?
- MLA has a really high arithmetic intensity
    - 256 FLOPs/bte
    - First algorithm that almost reaches compute bound regime on H100
    - 3 main reasons:
        - single head latent loaded from off chip to on chip, and uses this across all query heads when computing attention. Just like MQA
        - however, it DOUBLES because it uses same latent for keys and values. Hence, double the AI for MQA
        - Deepseek models tend to use large number of Q heads: 128. 400B llama model uses only 40.
            - If you reduce KV cache by X amount, can increase Q heads to Y, to compensate for loss in quality by reducing KV cache

## Group-Tied Attention

- Typing KV states to reduce KV cache
    - Tying K and V onto a single state: used full head dim for the value dim, and half of it for the key
    - 2nd half of key comes from a separate single head half dim projection that is used specifically for RoPE. Broadcasted to all heads.
        - Keys tend to be low rank
            - Singular value plots (eg llama3 8B) reveal a steep decay in keys, so it resides in low-rank subspace
            - pre-RoPE keys collapse into an evan smaller subspace. 25x smaller rank, no loss
        - Partial RoPE (~50%)
            - Sufficient to rotate the head dim only by 50% for positional encoding
        - GQA style grouping for ease of parallelization
- Halves KV cache, doubles Arithmetic intensity compared to GQA. Performs rougly the same, upto 1.5B

## Grouped Latent Attention (GLA)

- More parallelisation friendly
- MLA -> (1,4d)
- GLA -> (G, 2d)
- shards latent across multiple devices
- Instead of using 4d head size, as proposed by deepseek in MLA, GLA uses 2d
- BUT, instead of using a single head, GLA uses G heads, where each head is on a different device
- This has good outcomes for throughput:
    - TP=8 i.e 8 GPUs GLA-8(8, 2d) vs MLA(4d)
    - 1/2 KV cache per device and 4X larger KV cache budget
    - Higher throughput and lower latency due to loading 2d vs 4d per device
    - becuase we load 4 more smaller KV cache per device
- Some moar low level optimisation. Across many seqlen, outperforms deepseek MLA
- For large seqlen, hits theoretical max for MLA.

## Speed - test with live SGLang server

- GLA > MLA
- MLA needs some form of DP, to prevent the KV cache duplication issue?
    - Need to explore what this issue is. 
- GLA-8 has 2.5x higher throughput
- GLA > MLA and GLA >=GQA, when we have the same number of groups
