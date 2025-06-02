# Harnessing the Universal Geometry of Embeddings

This paper shows that:
- Given just the embeddings from a model, a **different** dataset and an encoder to convert this dataset to embeddings, the embedding representations can be aligned such that **original information** can be recovered from just the embeddings.
- They use the same methodology as Generative Adversarial Networks(GANs) i.e have a generator and a discriminator network.
  - Embeddings are encoded and decoded using adapter modules and passed through a shared backbone network.
  - Embeddings **dont** have spatial bias, hence use MLP instead of CNN.
- The model is training using 3 loss functions to align the embeddings:
  - **Reconstruction**: Ensure that when embedding is mapped to a latent space, and convert back from latent space to embedding, it closely matches intial representation.
  - **Cycle-consistency**: Unsupervised proxy for supervised pair alignment.
  - **Vector Space Preservation(VSP)**: Relationships between generated embeddings remain consisntent under translation.
- The implications of this paper are pretty significant imo: This means that if I have a large enough dataset, I can use a provider-based embedding model(voyage, openai, etc.), to generate an intial dataset of embeddings. I can then train my own network using this methodology, and end up with an embedding model that's as good as these providers, essentially STEALING quality.
