# Jetformer

- Jetformer seems to be the first model that can synthesize text and images autoregressively.
- It's primarily based on using a normalizing flow model for the images, which because it's invertible, automatically acts as a lossless decoder.
  - However, I'm not sure I agree with this statement. The normalizing flow model still predicts 'soft-tokens' i.e distributions of probabilities for embeddings. By the nature of embeddings, when we try to reconstruct the image, we will loss information.
- The scale of the experiments is quite small. When compared to the smaller CLIP models, Jetformer still lags behind by a LOT.
- Due to the nature of Jetformer, which is end-to-end trainable, and attention can be applied in the entire model. The size of the model has increased, which also increases the computational cost.


Overall, interesting paper. I don't have the background yet to understand diffusion/normalizing flow models, so I might revisit this paper later.

