# LFM2

- Hardware-in-the-loop architecture search
- Hybrid backbone
    - Gated short convolutions
    - Small number of GQA blocks
- Fast prefill and decode in CPU
- Training pipeline:
    - Top-K knowledge distillation objection that avoids support mismatch
    - curriculum learning with difficulty-ordered data
    - 3 stage post-training of SFT, length-norm preference optimization and model merging(ofc).
- LFM2-VL, LFM2-Audio and LFM2-ColBERT also available
- Primarily aiming to deploy on the edge.

## Architecture

- New archs combine 3 usual ingredients:
    - alternative seq operators such as linear attention or SSMs
    - short range convolutions
    - non-trivial fraction of standard softmax attention layers to recover quality degradation observed in models without softmax attention layers(i.e mamba like models).
- Doing a search on the above 3 shows that:
    - minimum hybrid of gated shortt convs for most layers + some GQA layers, without ssm or linear-attention ops, give good perf
- Gated Short convolution
    - Structure
        - each blocks applies input-dependent multipliate gating around a depthwise short conv
        - conv(x) is a depthwise 1D conv with kernel size k=3
        - gate(x) is a learned gating function
    - Why
        - fast local mixing: Short conv (k=3) captures local ctx
        - good cache behavior: small kernel = minimal memory access patterns
        - input-dependent modulation: gating mechanism allows model to selectively filter information. same as LSTMs/GRUs
- Search space for the hardware-in-the-loop apprach:
    - Decoder-only stacks
    - Local ctx and subquadratic blocks: 
        - gated short conv blocks with diff kernel slides, SWA
        - linear attention variats
        - SSM variats(S4, Mamba)
    - Global context blocks: GQA
    - Position-wise blocks: SwiGLU feed-forward blocks
    - Layout: interleaving patters of local ctx blocks, global ctx blocks, position-wise blocks
    - MoE options: per-layer sparse FFNs with varying width and expert granularity
- On-device profiling
    - every candidate is compiled to deployment stacks with same settings(batch=1, ctx=4K/32K)
    - CPU path: Executorch(8da4w) and llama.cpp(Q4_0)
    - Accelerator path: vLLM for single-request and online batching
- TLDR: search always selects gated short conv blocks interleaved with GQA
    - match or exceed quality of models like conv+linear/ssm/conv
    - reduce decode latency and increase prefill throughput
    - lower peak RSS(peak memory measured as maximum resident set size) at long ctx(4k/32K)
- Use BPE with 65K vocab size. FIM objectives, tool calling and ChatML chat template

### Decoupled Top-K knowledge distillation

- Distill teacher distribution $P_T(x | x_c)$ for each token $x$ and it's context $x_c$, to student $P_S(x|x_c)$ using only Top-k=32 teacher logits per token.
- naively apply KL diveragence, especially under temperature scaling can cause support mismatch and ustable losses
- Decompose KL via chain rule into:
    - binary term that matches total prob mass assigned to top-k set
    - conditinoal KL within top-k, to which we apply temperature.


```
Standard (broken):     KL(softmax(z_T/τ) || softmax(z_S/τ))
                     ↑ temperature applied to truncated dist = wrong mass

Decoupled (correct):   Binary(mass_in_topK) + τ·KL(conditional_within_topK)
                     ↑ temperature only affects relative ranking inside top-K

The support mismatch problem

# WRONG: Naive temperature on truncated logits
teacher_probs = softmax(topk_logits / temperature)  # sums to 1, but shouldn't!
# This artificially inflates probability of top-K tokens

# CORRECT: Separate the "how much mass in top-K" from "distribution within top-K"

What to store from teacher

For each token, store:
1. topk_logits: (K,) - the K highest logit values
2. topk_indices: (K,) - which vocab tokens they correspond to
3. Optionally: mass_in_topk: scalar - actual probability mass in top-K (I approximated as 0.95 above)

This reduces storage from V floats (~128K) to ~65 values per token.
```

### Training Pipline

- 3 stage: SFT, Alignment and Merging
- SFT data mixture
    - 5.4million samples for 350M, 700M, 1.2B and 2.6B
    - Dedupliucate data using CMinHash locality-sensitive hashing to catch similar but non-identical content
    - Exact n-gram matching against evaluation benchmarks
- Curriculum learning
    - use model ensemble(12 models) and run them over dataset and record binary outcomes, indicating whether model answered item correctly.
        - LFM2-350M, LFM2-700M
        - Hunyuan-500M
        - Phi-4-mini
        - Qwen3-b
        - ERNIE-4.5
        - Qwen3-30B-A3b-Intruct
        - Qwen3-30B-A3b-Intruct-Coder
    - Compute overall success rate for each item. Items with high success rate are easier, and low success rate are hard problems.
    - Train classifier to predict and order probabilities based on prompt features, and providing ranking from east to most challenging.

#### Training Protocol

- 32K token ctx length during mid training
- dropout only applied to embedding at training, and nothing at intermediate layers, to preserve latency.
- mixed precision traininig 8 H100-80GB GPU

### Stage 2: Preference Alignment

- Combine family of length-normalised objectives with offline dataset consisting of both on-policy samples from SFT checkpoint and off-policy reference responses.

#### Preference dataset creation

- base: intruction _ preference data from 1 million conversations
- For each conversation, sample n=5 responses from selected SFT ckpt to recover N+1 total responses for instructions datasets and N + 2 total responses for preference datasets.
- Score individual responses with LLM jury, selecting highest scored sample as the chosen response 
- For instruction data, responses are further refined using Constrastive learning from AI revisions.
    - apply quantitative score-based filtering to remove low quality preference pairs
    - mitigate potential undesirable behaviour models
- Finally, filter out partial samples that exceed ctx len
- Resulting dataset: 700K conversations

#### Length-Normalized Direct Alignment

- DPO objective: $L_DPO = -log σ(β · (log P(y_w|x) - log P(y_l|x) - log P_ref(y_w|x) + log P_ref(y_l|x)))$
    - Log probs are sums over tokens, so longer responses naturally have lower(more negative) log probs. 
    - Creates bias towards shorter responses(if they have "higher" log probs)
    - or if rejected samples are shorter, models learns to be verbose i.e prefers long answers.
- Solution: Normalize log probs by sequence length:
    - Instead of: log P(y|x)
    - We use:     log P(y|x) / |y|  (average log prob per token)

- 700K preference conversations during training. Without length normalization, the model might:
    - Learn to give terse responses (if rejected samples tend to be longer)
    - Or become overly verbose (if preferred samples are longer)
- Length normalization ensures the model learns quality per token, not just sequence-level likelihood.

### Stage 3: Model Merging

- Multiple-parameter space merging techniques to combine fine tuned models without requiring additional training.
    - techniques: model soup, task arithmetic, TIES-Merging, DARE and DELLA
    - Model soup: Weighted average of corresponding params across models.
    - Task arithmetic: Extend model soups by using param deltas relative to base model. Compute task vectors (base_model_theta - task_specific_theta), and weight-merge them to base model
    - DARE: Drop and REscale: random sparsification i.e dropout. Each param in task vector is dropped with prob p, and retrained otherwise. Maintain magnitude of updates by rescaling retrained params by 1/(1-p).
    - DELLA: Drop and rEscale via samlLing and Magnitude: Extend DARE by replacing unfirom random dropout with magnitude-aware sampling

### Evaluation

- Small models fail evals primarily because output format doesn't amtch expectations, even if the answer is actually correct.
- Built custom harness with good parsing logic.


## Vision Language LFM2

- Augment language model with lightweight visual encoder and connector
- Encoder: SigLIP2
- Adopt NaFlex variant of SigLIP2. Supports variable input resolutions and native aspect ratios.
- Connector applies a PixelUnshuffle op that lower number of visual tokens. Reduces tokens by 4x.
- MLP maps image embeddings into LFM2 hidden dimension.


### Training Protocol

- Stage 1: Connector Pretraining
    - focus on visual and textual embedding spaces while keeping image encoder and language backbone frozen
    - establish mapping between image patch embeddings into the LM embedding space before full multimodal embedding.
    - ctx len of 2048 tokens.
- Stage 2: Multimodal Mid-training
    - jointly optimise LM, pre-trained connector and vision encoder
    - All are training using learning rate ratio of 5:5:1 for backbone:connector:encoder
    - smaller LR here benefits vision training perf
    - ctx len at 32K tokens. Samples are packed
    - language pretrianing mixture is reused + augmented with multimodal data
        - recaption existing image datasets and generate VQA examples targeetted at visual understanding like reasoning about objects, layouts, etc.
- Stage 3: Joint Multimodal SFT
    - fine-tuning on multimodal instruction-following data
    - High-resoltuion tiling is enabled to explose model to realistic deployment resolutions.
    - ctx 32k
    - train variants on 50B tokens to measure tradeoff
    - this stage needed for downstream usecases like vqa, doc understanding, etc.
- Checkpoint selection and Merging:
    - For LFM2-VL-3B, trainin several candidates runs that vary in training procedure or data composition.
    - Measured on internal multimodal benchmark
    - Highest candidates are merged with simple linear merge in weight space.

### Evaluations

- InternVL scores about the same as LFM2, with approximately same model sizes.
- Good perf overall

## LFM2 Audio

- Extend LFM LM with audio input and output components.
- Explicitly seperate continous audion input features from discrete autio codes
- Audio input handle via log-mel features and an ecoder stack
- audio generation is implemented using RVQ(Residual Vector Quantization) and lightweight audio detokenizer.
- Seperation presevers rich representation for speech recognitino and understanding

### Architecture

- LFM2-1.2B language backbone
- audio encoder on input side
- discrete audio code path + detokenizer on output
    - Discrete Audio Tokens
        - Audio is quantized into a finite vocabulary of token IDs (like text tokens)
        - Created using neural audio codecs like EnCodec, SoundStream, or DAC
        - Uses vector quantization (VQ) to map continuous audio embeddings to nearest codebook entries
        - Enables using standard language model architectures (autoregressive transformers) directly
        - Lossy by nature—quantization introduces information loss
        - Example: "Audio token 423, 891, 156, ..."

    - Continuous Audio Tokens
        - Audio represented as continuous-valued embedding vectors (floats)
        - No quantization bottleneck—preserves full information
        - Often extracted from encoder models (like Whisper encoder, HuBERT, or audio VAE latents)
        - Requires modified architectures that can handle continuous inputs/outputs (e.g., diffusion, flow matching, regression heads)
        - Higher fidelity potential but can't directly use cross-entropy loss or standard LM sampling

### Encoder

- Audio encoder produces sequence of continous embeddings
- First, convert 16kHz waveforms into 128 dimensional log-mel features using 0.025swindow and 0.01 stride.
- Small stack of strided convolutions performs 8x temporal downsamping , tehn 17FatConformerlayers, resulting in 512 dimensional continous features
- MLP connector projects to LFM2-1.2B hidden dim of 2048
- Each resulting embedding corresponds to 0.08 seconds of audio, opeartion at a temporal resolution of 12.5 embeddings per seconds of input audio


### Decoder

- Predicts 8-codebook Mini RVQ codfes, which are converted to waveform by a detokenizer.
- 8 codebooks
    - 1 semantic codebooks
    - 7 acoustic codebooks
- Produces 2049 tokens per codebook.
- Once 8-code frame has been generated, eight code tokens are embedding and summed to form a single 2049 size embedding. this is fed into LFM2 backbone for AR generation. In parallel, 8-code frame is streamed to detokenizer for real-time playback.
- To reduce latency, LFM2 backbone DOES NOT predict all 8 codes in a sequence
- Use RQ-tranformesr(similar to Moshi) as a decoder for code generation
- LFM2 backbone produces a single 2048 dim embedding. Used to condition RQ transformer, which runs for 8 steps to AR genenrate 8 codes used in frame(one per codebook)
    - When model is in audio generation state, each backbone step is followed by 8 RQ-transformers steps(which is quite fast). 

### Audio Detokeinizer

- Mimi decoder too slow for target hardware.
- Train LFM2-stlle detokenizer for edge deployment
- Follows Vocos architecture
- Map sequence of discrete 8-code Mimi frames to a sequence of complex short-time Fourier transform coefficients, which are then converted to 24kHz audio by an inverse STFT.
- 35M variant of LFM2 with 8 layers(5 gated short conv blocks + 3 sliding-window attention blocks)
- fast enouigh for real-time synthesis on edge hardware

### Inference

- LFM-Audio supports 2 distinct inference models: interleved and sequential
- Interleaves
    - suited to settings that require non-trivial reasoning and low-latency speech output(conversational assistants)
    - model can emit text and audio in real time
- Sequential
    - produces one modality at time
    - switch between modalities as needed
    - useful for ASR, TTS or mixed text/audio response
- both models are *stateful*: alow seperation of text and audio vocabularies.
- In _text_ generation state, model behaves like LFM2 base model, using output embedding to compute next-token probs
- In _audio_ generation state, models sends output embedding to RQ_transformer and computes next-code probs over audio vocab.

#### Interleaved generation

- In this mode, LFM2-Audio generates the assistant response in both text and audio, with **predetermined** interleaved pattern.
    - LM first only generates text
    - Then translates text into audio
    - Generate 6 text tokens, followed by 12 audio tokens, and repeat till text tokens are exhausted.
    - Once done, output <end_of_text> token, followed by outputting all audio tokens, and finally <end_of_audio_> token
    - 6:12 ratio of text to audio is to keep text gen ahead of audio gen and minimize time-to-first audio token.
- Fixed ratio and separate audio and text vocabs enables independence from extra control tokens, as require dby GLM-4 Voice and similar models.

#### Sequential generation

- LFM2-Audio can change it's own generation state by generating special control tokens.
- Model always starts in text generation mode. Only text vocab used.
- When <start_of_audio> token is seen, the model switches to audio generation state and starts using RQ-Transformers and audio vocab to generate outputs.
    - Continues till it sees <end_of_audio> token, and switches back to text state.
- Suitable for ASR tasks, since they only involve text output.
- Can also be used for TTS, by generating the <start_of_audio> first, and then stopping at <end_of_audio>

### Training Details

- LFM2-Audio is deocder only
- Trained with autoregressive cross-entropy over discrete targets, with sepearate loss for text and audio tokens.
- Compute loss separately for text in $L_T$ and for each of the 8 audio mimi codecs in $L_1,...,L_8$.
- Combine the loss via weighted average. Weight 100 assigned to $L_1$, and exponential decay weights such as $L_8$ gets weight 1.
- Final loss is mean of $L_T$ and $L_A$, weighted by number of modality tokens present in each batch.
- Do not compute loss on continous audio input emebddings.
- **Training phases**:
    - 3 phases: alignment, mid-training and post-trianing
    - Alignment: ASR data to trian MLP connector. Keep LFM2 backbone and encoder frozen. Move to mid-training when loss flattens.
    - Mid training: Unfreeze entire model and trainin on large-scale mixture of text and audio datasets. 100 billion tokens, 62% text and 25% audio output frames, 13% audio input embeddings(log-mel features)
- **Detokenizer training**:
    - Detokenizer trained separately as GAN, using both a multi-period discriminator and multi-resolution discriminator.
    - Training bojective: combo of log-mdel loss(L1), discriminator loss(hinge loss on each frame), and feature mathcing loss.

### Evaluation

- Evaluated on general-purpose speech-to-speech conversational abilities and ASR.
- VoiceBench
- Performs much better compared for Qwen2.5-Omni-3b, moshji, mini-omni2

## LFM2-ColBERT

- Extend LFM2 backbone through LFM2-ColBERT-350M, a late interaction retriever optimized for multilingual and cross-lingual semantic search.
- Maintain embeddings of query and doc, and compare them at retrieval time.
- Comparison is done using the **MaxSim** operator.

### Training

- Continue LFM2-350M ckpt to 25T tokens
- trained using PyLate library and knowledge distallation
- KD better than pure contrastive learning approaches
- ctx len 32k
- apply min-max norm to teacher scores to ensure numerical stability and consistent gradient magnitudes across teacher model scales.
- During training, each isntance contains a query and 8 docs with their corresponding teacher scores. Doc inputs are truncated to 512 tokens and query inputs to 32 tokens, matching target deployment constraints.

### Evaluation

- Performs well on NanoBIER
- Compariable performance to GTE-ModernColBERT-v1 model
