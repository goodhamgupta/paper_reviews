# Thinking Machines: On Policy Distillation

Review of the [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

## Intro

- LLMs capable of expert perf in many domains because of stacked capabilities
- Needs stack of training approaches:
  - Pre-training: teaches general capacities such as language use, broad reasoning and world knowledge
  - Mid-training imparts domain knowledge, such as code, medical databses or internal company documents.
  - Post-training: elicits targetted behaviour, such as instruction following, reasoning through math problems, or chat.

- Smaller models with stronger training > larger generalist models 
- Smaller models good: deployed locally, continously train, save inference costs.

- Posttraining student approaches can be divided into two kinds:
  - On-policy training: samples rollouts from the student model, and assigns reward.
  - Off-policy training: relies on target outputs from some external source, that student learns to imitate.

- Model to solve math questions: can use on-policy via rl. grad each student rollout on whether it solves question. grade by human or 'teacher' llm.

**On-policy training**
- Strengths: See traces that the model itself predicts i.e closer to the training distribution. Learns to avoid mistakes in a more direct way.
- BUT, RL provides very sparse feedback, teaching a fixed number of buts per training episode, irrespective of the number of tkens used. 
  - If student produces wrong answer, the teacher's trace updates the student away from producing the original answer, even if some of the original reasoning trace was correct. 
    - Couldn't this be solved with process-reward-models, instead of objective-reward-models?

**Off-policy training**
- done with sft: trianing on a curated set of task-specific labelled examples.
- Source of these examples can be teacher model good at the task, or human annotations.

- **Distillation**: Training student to match output distribution of a teacher model.
  - Train on teacher _trajectories_: Complete sequence of generated tokens including intermediate thinking steps.
  - Use teacher's full next-token distribution at each step(**logit distillation**) or sample given sequences.
  - Sampling sequences provides an unbiased estimation of the teacher's distribution and arrives at the same objective.
  - Student updates towards each token in the sequence in proportion to how unliekly it was to generate that token itself i.e how likely the token the student predicted was incorrect, according to the teacher distribution.
- Distillation from large models is effective in training small models to follow instructions, reason on math, etc. 

- **Drawback of off-policy training**:
  - Student learns in contexts frequented by the teachers, not ones the student itself will often find itself in.
    - Can cause compounding error: if student makes and early mistake, that teacher never makes, it finds tiself diverging even farther from the states it observed in training i.e diverge from training data. 
    - This problem increases when we are more concerned about the student's performance on long sequences. 
    - To avoid divergence, student must recover from it's own mistakes.
  - Student tends to learn to mimic the style and confidence of the teacher, but it still **not factually accurate**.

- If playing chess:
  - On-policy RL is similar to playing games with no coaching.
    - Feedback of winning/losing is tied directly to your own play, BUT RECEIVED only once per match and doesn't tell you which move contributed to the outcome.
  - Off-policy RL is similar to watch best and most complex moves by a grandmaster.
    - observe extremely strong moves, but they are in board states that the novice player will rarely find themselves in.


- **GOAL: combine on-policy relevance of RL with dense reward signal of distillation**
  - For learning chess, this would be a teacher that grades each of your own moves on a scale of "blunder" to "brilliant". 
  - For LLM post-training, this is called **on-policy distillation**.


## On-policy Distillation

- **Core idea**: Sample trajectories from the student model, and use a high-performing teacher to grade each token of each trajectory.
- This would punish mistakes that caused the student to arrive at the wrong answer, while reinforcing the ones that were executed correctly.
- Aim to train models for math reasoning and training an assistantn models that combines domain knowledge with instruction following.
- OPD is based on [DAGGER](https://arxiv.org/abs/1011.0686), an iterative SFT algorithm that includes teacher evals of student-visited states.
- THIS IS SIMILAR TO PROCESS REWARD MODELLING!

### Impplementation

- **Loss function**: Reverse KL

