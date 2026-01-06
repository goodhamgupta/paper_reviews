# Attention Sinks

- Emerge because of a property of attention.
- TLDR: Keeping the KV cache of the initial tokens(called "sinks") helps recover performance of the window attention, even as text size increases past the KV cache size
- Impact: Can generate text indefinitely/process text infinitely, even if the original model was trained to only process sequences of length N.

## Motivation

- When processing LLMs for infinite input streams, 2 problems:
  - KV cache grows with seq len N, leading to excession memory usage.
  - Existing models have limited length extrapolation abilities. perf degrades when seqlen goes beyond attention window size set during pre-training.
- Window Attention:
  - Maintains only fixed-size sliding window on KV states of recent tokens.
  - Ensures constant memory usage and deocding speed after cache is filled.
  - But model collapses once seqlen exceeds cache size.

## Attention Sinks

- Large attention scores allocated to initial tokens.
- Primarily because of "Softmax":
  - Attention needs to sum up to 1
  - Even whent eh current query doesn not have a strong match in many previous tokjens, the model still needs to allocate these unneded attention values somewhere to sum up to 1.
  - Allocate to "initial tokens" because initial tokens are visible to all all subsequent tokens because of autoregressive modelling nature. Hence, they're more readily used as attention sinks.
- StreamingLLM keeps only the KV cache of the first 4 tokens together with the sliding window's KV. Can reliably generate 4 million tokens or even more.
- Can extend this inside: LMs can be pre-trained to only require a **single attention sink token** for all streaming deploymenent.
  - Add extra lernable token at the beginning of all training samples, which will serve as a designated attention sink.

## Props of attention sinks

### [Massive activations](https://arxiv.org/pdf/2402.17762)
  - Widespread existence and can detect their locations
  - Their values remain more-or-less the same irrespective of input. Hence they act as bias terms to LLMs.
  - their massive activations leads to concentration of attention probabilities to their corresponding tokens and _implicit_ bias in the attention probs.
  - Sometimes, the magnitude of _massive attention_ values is about 10000x more than the median magnitude.
  - Paper shows that most consistently, massive attention values are within the first 4-5 tokens.
  - Massive activations emerge in the initial layers, stay mostly the same in intermediate layers, and start to dimish in the last few layers.
  - Paper runs an experiment, where interventions are applied to certain layer showing massive activations, and the activations are manually set to zero or mean. Setting to zero leads to perf degration, but setting to mean maintains perf.
  - Explicit attention bias: Explicitly adding 2 learnable params(k' and v') to the attention formula and training a model shows that the massive activations disappear.

- In Vision Transfomers(ViTs):
  - Massive activations exist in many BUT NOT ALL ViTs. 
    - Exist in DINO and CLIP
    - Don't exist in MAE ViT-L
  - they only appear in the later stages on the model.
  - Applying interventions, setting activations to 0 leads to **significant** degration of perf, but setting to mean leads to no perf degration. Confirms they hypothesis that these activations are "important bias" to the models, in both AR LLMs and VLMs.

### [Value Drains](https://arxiv.org/pdf/2410.10781)

- Attention sinks emerges after effective optimisation on sufficient training data
- Highly related to loss function and data distribution
- Attention sinks acts more like _key biases_, storing extra attention scores, which could be non-informative and not contribute to the value computation. Also stems because of softmax norm. replacing softmax norm with other activation functions(such as sigmoid without norm) fixes the issue.
- Attention sinks under different inputs:
  - Randomly sample `T` tokens from vocab `V`.
  - Create random samples of input for the models.
  - Sinks still exist even when input is not natural language, but collection of random tokens.
  - Sinks disappear in MIstral and LLaMa models. In models with NoPE/relative PE/ALiBi/RoPE, if first T toekns are the same, hidden states are the same. They all have massive activations, thus dispersing the attention sinks.
- Weight decay ecouranges the emergence of attention sink
- With prefix LM, attention sink appears among the prefix tokens rather than the firs totken.
- With shifted window attention, attention sink appears on "absolute", not "relative" first token.
    - Smaller window size prevents the emergence of attention sink.
- Positional embedding, FFN design, LN location and multi-head design do not affect emergence of attention sink
- Having explicit attention biases, helps remove emergence of attentions sinks. However, even after introducing this, the sink token acts as key biases, soring extra attention scores, which could be non-informative and not contribute to the value computation.
    - Removing any of the introduced biases/sink tokens will lead to no attention sink in the first position(if the model is trained with them), but leads to degraded performance.
- When relaxing tokens' inner dependence on attention scores(via the softmax norm), attention sink does not emerge in LMs.

## Applications of attention sinks

- Long ctx understanding/generation
- KV cache optimsation
    - Only store KV cache for the attention sink token(explciit or first token) + sliding window tokens.
- Model quantisation
    - Papers: IntactKV: 
    - Preserving the full precision of KV cache of sink token
- Multimodal LM:
    - Papers: SEED-Story: Multimodal long story generation with LLM
    - Have 4 special attention sink tokens: 0, Punctuation, Beginning of Image(BOI) and End of Image(EOI)

## Mechanism Understanding of Attention Sink

- Attention sink is due to the key **key** bias of the sink token
- **key** of the sink token is located in the different manifold. It has small angles with any queries.
- Existence of massive activations is to support attention sink
- Why all these phenomenon tend to happen in the first token?
    - UNiqueness of first token: For the first token only, self attention involves no other tokens. All hidden states in the forward path are equaivalent to MLP transformations of input embeddings.
    - LLMs learn to map the input embeddings to massive activations after certain layers, leading to key bias and then to attention sink. Here, key refers to the _keys_ matrix used as part of the attention computation.
- Attention sink approximates as a "no-op":
    - `Value` i.e content in the value matrix for the first token is quite small.
    - Softmax(Q*K^T) is multiplied with value in attention computation. When computing self attention, we need to do the operation for all tokens preceding the current token. BUT, for the first token, there are no preceding tokens. Therefore, this value IS ALSO SMALL.
- Effects of optimisation:
    - Large learning rate encourages attention sink, even under the same LR* steps.
- Effects of data distribution: Attention sink emerges when we have enough unique training data amount.
- Effects of loss function: Sink token shifts from the firs ttoken to other positions within the prefix.

## Why LLMs need attention sinks?

- Attention blocks try to mix rrepresentations
- Attention sink serves as a mechanism to prevent over-mixing
- With attention sink, pertubation on one token("greatest" to "best") won't change token reps a lot
- Implements "no-op": either sharply to atetend to one important token or attend to the first token.
    - From representation mxiing perspective, LLMs need "no-op" to prevent over-mixing.
- Attention sink tends to attach to <BOS> in the first position.
- Widely exists in Transformer family.


## Why GPT-OSS and Qwen3-Next consider attention sink in model design?

- GPT-OSS adopts attention biases
  - Learnable key biases, zero value biases
  - Softmax off by one
  - Learnable attention scores biases(single number for each head, layer)
- First token does not develop strong attention sink, thus mitigating massive activations/outliers
    - Facilitate quantization, pre-training stability
- Attention sink only hapens in **absolute** first token, not **relative** first token.
- Tokens beyond window size have no sinks to attend, possible over-mixing
    - facilitate long context, especially in LLMs with alternative shifted window/full attention

## Qwen3-Next adopts Gated Attention

- Sigmoid gate allow "no-op", no need to only rely on attention sink for "no op".

Generate Full Report
