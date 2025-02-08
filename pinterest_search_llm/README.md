# Pinterest: Improve Search using LLMs

Review of the paper titled: Improving Pinterest Search Relevance Using Large Language Models

In summary:

- They trained an LLM to predict relevance labels based on pin data + features carefully engineered and obtained from previous methods like PinSage.
- First, they trained a teacher LLM using SFT. Then, they used this teacher model to label a large dataset of pins. The labels were of 5 levels(from most relevant to least relevant).
- After labelling data, they trained a student LLM using the labelled data. Overall, even though the labelling was noisy, they get impresive results, leading to around 2% improvement in search relevance.

Since neither the dataset nor the code is available, I will not be able to reproduce the results.
