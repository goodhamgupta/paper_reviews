# SFT Memorizes, RL Generalizes

Paper review of ["SFT Memorizes, RL Generalizes"](https://arxiv.org/abs/2501.17161).

Overall, they do the following:

- Create two 2 datasets:
  - `GeneralPoints`: 
      - This is a card-game based dataset.
      - Given 4 cards, whether in images or pure text, the model is asked to produce a final number(usually 24) using a combination of these cards.
      - When using images, models are trained on all cards of only one color, and tested on another color(to test OOD generalization).
  - `V-IRL`
    - Real world navigation.
    - Recognize different landmarks from the visual observation.
    - In domain uses usual directions(north south, east west, north-east, etc), while OOD 4 directions(left, right, slightly left and slightly right).
- Run 2 stage post-training: SFT and RL(with PPO).
- Include a self-verifier, which is just appending the current state, previous state, action, and output to the prompt and generating next token.
- Through benchmarks, they show that RL generalizes better than SFT, and SFT memorizes better than RL.
- However, they also note that SFT is necessary for RL to get it's full potential. SFT helps control the model output behaviour. This matches the results obtained by Deepseek R1, where they noted that direct RL on the base model led to instable vhains of thought(language mixing).
- While the number are promising, the authors provide no concrete reasoning as to WHY RL generalizes better than SFT.
- They do note that verifiers are important for RL to generalise better. This follows Rich Sutton's lession: "An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself."

Good paper overall!
