# Alphamaze

Review of the paper "Alphamaze" by Homebrew Research.

## Notes

- The demonstrate the ability of SFT + RL on a maze solving task.
- The maze is programatically generated. 
- Special tokens are created to tokenise the maze. Tokens also created for directions of movement.
- Results show that baseline model is unable to solve the task(0% success rate). SFT solves with 87% success, and SFT + RL model is able to solve the task with 93&% success rate.
- They used GRPO as the RL algorithm to train the model.
- It's a bit surprising that they didn't try structured decoding for the baseline model. I would definitely not expect the base model to score 0%.
  - Their conclusion is a bit confusing. They say that baseline model scored around 70%, but this metric was not mentioned before in the paper.
  - However, the mentioned that the "distilled" base models from R1 scored 0% as well. I'm guessing they've mixed up some results here.
- The reward function has the highest effect on the model output. They have 3 rewards:
  - Correctness Reward: +0.2 for a step in the right direction
  - Integrity Reward: +0.5 for every valid direction token. Encourage generation of meaningful and valid paths.
  - <think> tag reward: +0.25 for printing the <thinking> token, which is to ensure that the model performs "CoT" before making a decision.
- Based on the performance of the distilled models, they conclude that spatial reasoning ability is not propogated well to smaller models through distillation. 
- Larger base model still scored decently well on long range tasks.
- As part of the model, they also created "MazeBench", a benchmark for solving 100 maze related tasks. There are 3 categories:
  - Easy: Solvable in 1-4 steps
  - Medium: Solvable in 5-8 steps
  - Hard: Solvable in 9-13 steps

## Questions

- How would structured decoding affect the baseline model?
- How does the reward function affect the model output? Are there other factors we can include in the reward function?
