# Optimising LLM Test-Time Compute Involves Solving a Meta-RL Problem

This blog aims to draw a connection between Test-Time Compute and Meta-RL.

The key idea is that:

- In Meta-RL, models are allowed to run a few 'training episodes' on the test sample, before their response is evaluated. Test-time compute is similar, where the 'training episodes' here are just extra tokens.
- The way to optimise these models is to use dense reward models. To use these models, we need a way to distinguish each training episode from just a series of tokens. One way to do this is to use the QuietSTaR apporach. Note that this is a difficult problems, and the authors have just provided some ideas.
- Assuming we have a way to distingush training episodes, we can then use a "dense reward model". This model adds a reward for the value of each training episode(or step) and a weighted reward for the final output.
  - This is slightly different from Process Reward Models(PRMs), which only include a reward for each individual step. This is why the current setup is called "dense reward model"
- The 'training episode' length can only consume a specific amount of compute, C. We can specify this compute to the model in 2 ways:
  - Explicit: Model the compute C explicitly as part of the loss function. Eg: Assume compute C is the 'number of tokens' produced in the output. If we allow only 1K 'training episode' tokens, then the loss term will penalize all answers about 1K tokens.
  - Implicit: Cut off the model response after compute C is consumed. This is more tricky to optimise.


Really nice blog overall.
