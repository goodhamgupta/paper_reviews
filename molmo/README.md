# Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models

Review of the Molmo paper. Really interesting paper, since this is one of the few papers that open-source both the dataset, model AND training methodology.

Some key points:

- Created a novel point dataset, which is quite useful in counting objects in an image. Leads to substantial improvements in the downstream tasks. Authors say this is useful for other tasks such as pointing in the direction to go, pointing to UI elements, etc. This dataset is also easier to collect than asking annotators to draw bounding boxes. However, I think a SAM-2 model would've done this well, and the authors do point it out later in the paper.
- NO VLMs were used to create the dataset. The authors stress this point multiple times throughout the paper.
- Dataset collection is quite interesting: For the images, instead of asking users to type out the caption, they are given a series of 7 questions and asked to describe the image in speech. The transcript is then used to annotate the image. This is done to avoid usage of VLMs in the training data.
- Special tokens were used to distinguish patches of high and low resolution images. Text-only dropout was used in pretraining.
- Authors say that they could'nt use flash attention because their dataset setup required funky attention masks, which were not implemented. Maybe FlexAttention based modules could help here.
- Dataset specific prompts are used while generating answers. Markers such as "vqa2:". Some markers also include expected text length output, which is shown to improve downstream performance. The length if computed as: (len(answer) + noise_from_normal_distribution_with_std_dev(25)/ 15 to keep length in range [0, 100]
- They used a novel cropping strategy when creating the pathces for images, focusing on overlap between patches. This has improved performance significantly.
- At the time of release, Molmo was the best OS model. On ChatArea, it beats all other open-source models, but lags behind other API based provider models.
- Their best model is based on Qwen 2 72B. 
- Molmo is not very good at reasoning tasks. Authors say this is because of lack of training data diversity, which doesn't have any examples requiring comprehensive reasoning.
- Molmo training required full precision(i.e 32 bit), as bfloat16 led to poor performance. They use Pytorch FSDP when training the model, which gets same speed as flash attention.
- Training 7B molmo model required 64 H100 GPUs for ~9 hours. This comes up to around ~\$2700. Should be easy to replicate thier model.
- Precision/recall metrics are based entirely on GPT-4o, where GPT-4o is the judge.
- They also created a novel clock reading dataset. Since their model is trained on this dataset, it beats all other models on this task.
- For the vision encoder, they find that fully open MetaCLIP performs at the same level as CLIP and SigLIP. This is important since MetaCLIP is also fully open-source(including dataset).
- Authors note that captioning their Pixmo dataset with GPT-4o and then training the models improves the performance, mostly because 4o is a better VLM.
- Claude and GPT-4o used to generate code and instruction finetuning datasets.

Really enjoyed reading this paper! It's a great example of how to open-source a model and dataset.
