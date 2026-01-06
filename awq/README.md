# Activation-aware Weight Quantization

Notes on the AWQ algorithm, which was used to quantize the tomoro embedding models. Based on the [talk here](https://www.youtube.com/watch?v=dcINVsqxQgQ).

## Notes

- LLMs are hard to serve on teh edge.
- Quantization lwoers the bit-width and improves efficiency. For a 70B model:
    - FP16: 140 GB -> 2x80GB A100
    - INT8: 70 GB
    - INT4: 35GB
- SmoothQuant
    - Activations are very hard to quantize due to number of outliers.
    -  Outliers presist in fixed channels.
    -  Migrate the quant difficulty from activations to weights.
- Is W8A8 enough?
      - No
      - Edge LLMs are memory bound and requires low-bit weight quantization.
- Weight loading is the efficiency bottleneck for edge LLM inference.
    - Generate stage is slower. AR generation.
    - Generation stage is bounded by memory bandwidth. Memory bandwidth is more important, for decoding/generation. For prefill, compute is more important.
        - AWQ helpful when we are VRAM constraint, and want to have larger/batch/longer max seq length.
    - Weight loading is more expensive.
        - Weight traffic vs activation traffic. Weight traffic dominates most of the workload.

### AWQ

- Targeting group-wise low-bit weight-only quantization.
- Weight-only quant reduces memory requirements, accelerates token generation by alleviating the memory bottleneck
- First tried RTN(Round-To-Nearest) quantization. Works well.
- Group-wise/block-wise quant offers better accuracy-model size tradeoff. BUT not as good as RTN.
- Not all weights are important. Keeping only 1% of the salient weightr channels unquantized can greatly improve perplexity.
- How do we select the channels/weights? 
- LOOK AT ACTIVATION DISTRUBTION, NOT WEIGHT. USE THIS TO DETERMINE WHICH WEIGHTS TO QUANTIZE.

### Protecting salient weights by scaling

- After identifying the salient weights, we aim to scale UP these weights. Scaling upto 2x, gives significantly better performance.
- However, scaling up too much leads to performance degregation.
- Need to consider **activation-awareness** for salient channels.
- As long as AWQ scales **s**, which is the scaling factors, is not too large, quant error is inversely proportional to **s**.


### Advantages

- Accurate, simple, easyt
- High hardware efficiency
- Less dependecy on calibration set compared to regression-based method
    -Better data efficency and dsitrbution robestness
    - Generalize to instruction-tuned model and mutli-modal LMs
- AWQ is also applicable for VLMs. Quantize just the text-tower.
- Hardware-aware packing: A way of storing and loading quantized weights faster on certain hardware.

- AWQ assumes not all weights are important. Small fraction of weights that will be skipped during quantization which helps with quantization loss

## Thoughts

- For the model I created, I should've created a roofline model comparing performance against different quants.
