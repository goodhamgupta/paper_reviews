# Mixture-of-Experts

- Old idea that is being recycled given data and compute.

## Normal Network

input -> expert -> output


## MoE

- Multiple experts

input -> gating layer -> expert models(more than 2) -> output

- Gating Layer: Which expert to allocate to the existing input.
- Each is an expert at a particular task i.e specialized experts.
- Mistral MoE has 8 experts.
- Training
    - Train each at particular task
    - Train gating to determine best expert for the given task.
    - Can possibly allocate to multiple experts.
    - Have a final layer(final softmax) to get a combined output.

Parameters to learn
- Actual experts
- Gating network

- Apparently, GPT-4 is a MoE model.
    - 8 experts
    - Each expert is 220B parameter model
    - Gating layer.

- Maybe Gemini has a similar architecture.


- Original MoE paper is 10 years old. Ilya was one of the authors.
- Paper: Sparsely-Gated Mixture-of-experts layer: Quoc Le, Noam Shazeer, Jeff Dean.
- Switch Transformers: Almost trillion parameters size: Noam Shazeer and his group. Based on T5 architecture.
- OpenMoE:
    - Started sometime back
    - OPen source
- Mistral is NOT the first MoE.

Mixtral:
- Outperform GPT-3.5 on most benchmarks
- Handles 32k tokens
- English, french, italian, german and spanish
- Decoder only model
- SHARING key parts of the decoder network
- Experts are in the FF blocks
- Routers => Choose 2 groups of experts to process given token, and then give output.
- So it's not 8 x 7B. It only using 12B parameters per token(because experts on in FF blocks)
- Also released intruct version of Mixtral.
    - Used DPO
    - Model has not been censored
    - Allows fasters inference speed at low batch sizes
    - Higher throughput at large batch-sizes
