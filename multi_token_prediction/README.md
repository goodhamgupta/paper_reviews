# Multi Token Prediction

Review of the paper titled "Better & Faster Large Language Models via Multi-token Prediction" by FAIR.

Interesting paper, and their ideas seem to be catching stream as DeepSeek V3 was also trained using MTP.

Overall:

- MTP is acheived by using multiple heads to predict n tokens in the future, but with a shareable trunk.
- The result of the first head, is normalised, and passed through n-1 transformer layers to predict n-1 tokens.
- They show that MTP is generally better when used for large models, with n=4 giving the best result. I think this matches what DeepSeek V3 uses.
- They also provide explanation for why MTP improves performance. Not all tokens have the same weight. Some tokens are inconsequential, and sometokens are "choice" tokens, which can change the final answer being generated.
- Vanilla next-token-prediction uses teacher forcing, which increases the models ability to improve short-term prediction, at the cost of long-term prediction. 
- MTP however forces the model to consider n subsequent tokens, thereby average out the weight over these tokens. 
- Authors also say the current benchmarks are not a good fit for MTP based prediction. Instead, they evaluate and summarization and arithimetic tasks, where MTP outperforms vanilla models.
- For finetuning, the authors say that while n=4 MTP improves training performance, when finetuning, reverting back to vanilla NTP gives the best performance.
- The reasoning for the above is "induction heads": Essentially, during training, the model uses MTP to develop induction heads for reasoning. However, after a certain amount of training, the model learns to model the questions as a single token prediction, and the MTP heads are no longer useful. In fact, they reduce performance.
- There was a section about self-speculative decoding. Based on this [repo](https://github.com/dilab-zju/self-speculative-decoding), there are 2 stage:
  - **Drafting Stage**: The model drafts a seuqnece of tokens by skipping some layers
  - **Verification Stage**: The model verifies the drafted sequence by running the full model on the drafted sequence.
  - Note that the draft model is a **separate** model.
