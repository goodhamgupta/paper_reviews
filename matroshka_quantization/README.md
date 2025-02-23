# Matroshka Quantization

Review of the paper "Matroshka Quantization" paper by Google Deepmind.

Overall, this paper shows that:

- It's possible to joining train a model to be good at multiple levels of integer precision. Their experiments show that you can get int2, int4 and int8 quant models from the same model weights by just slicing, and it's better than training each model separately.
- Because the model aims to improve the performance of the lower precision, the weight values are shifted right i.e they move to higher value buckets. For eg: for int2, the models are allowed to change the value of other 6 MSB's to improve the performance.
- However, because the above technique needs to also cater for high precision models, the model performance is quite good at int8 and int4, but drops a little when using int2.
- Single Precision MatQuant helps train a better model specifically for int2.
- Becuase the quants are obtained by just slicing the model, we also get other quantizations for free. For eg: int3 and int6 quants will also be available.
- During training, experiments with 4 different quantization strategies:
  - Pyramid: Low precision at start and end of model, and high precision(int8) in the middle.
  - Reverse Pyramid: High precision at start and end of model, and low precision(int2) in the middle.
  - Increasing: Increasing precision from start to end.
  - Decresing: Decreasing precision from start to end
- The best strategy was Pyramid, which is also the most intuitive one.
- They also demonstrated a Mix n Match strategy, which allows mixing different quantization strategies in the same model. It is implemented using a Straight Through Estimator (STE) to backpropagate gradients through the quantization function.
- This paper has a lot of experiments and ablation studies, and the results are quite convincing. 
- It's well suited for inference time workloads. Can pick the right combination of quant depending on hardware + task to get pareto optimal performance.
- Authors say it could also lead to new hardware design, which can support multiple levels of quantization and can switch between them on the fly. This is super interesting, and I  wouldn't be surprised if google already has a implemented in their internal TPU chip.
