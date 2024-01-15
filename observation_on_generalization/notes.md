# Observation on Generalization
## Speaker: Ilya Sutskever

Claims to have very interesting results on AI Alignment(at OpenAI), but can't be talked about publicly for now.

## A theory of Unsupervised Learning

What is learning and why should it work at all?
    - Supervised Learning
        - Gives precise math condition under which learning must succeed.
        - Low training error + more training data than "degrees of freedom" = low test error
            - **ONLY ON THE SAME DISTRIBUTION OF DATA**
        - VC Dimension
            - Why was it invented?
                - Handle parameters of infinite precision
                - BUT in reality, on a computer, all floats have a precision
    - Unsupervised learning
        - An emperical phenomenon
            - Did not really show up at small scale
            - Old ideas were there: Autoencoders, etc.

    - Unsupervised learning is confusing
        - Very much like supervised learning
        - You optimize one objective
        - But you care about different objective
        - Doesn't make sense!

Digression: Unsupervised Learning via Distribution Matching

- Got data X and y, no correspondence
- Find F so that `(F(x)) ~ distribution(Y)`
- Works for high-D X and Y's
    - Substitution ciphers
    - Unsupervised machine translation
- Critically: it's clear why Distribution Matching **has** to work.

**Compression to the rescue**

- Prediction is compression, so can switch the language at "no cost".
- Say you have datasets X and Y
- Have good compression C(data)
- Say you have to compress X and Y jointly
- **What will a "sufficiently good compressor" do?**
    - Use patters that exist in X to help compress Y! and vice versa!!! i.e called Algorithic mutual information between the datasets.

#### Formalization of supervised learning?

- consider an ML algorithm, A, tha ttries to compress Y
- say it has access to X
- What's our regret of using this algorithm?
    - And regret relative to what?
- Low regret = "we got all the value" out of the unlabelled data X
    - And nobody could get much more vlaue than we did!

#### Kolmogorov complexity as the ultimiate compressor

- K(X) = length of the shortest program that outputs X
    - Given a dataset X, KC will give you the shortest program, such that when you run the program, you will get back X!
- If C is a computable compressor, then
    - For all X
    - `K(X) < |C(X)| + K(C) + O(1)`
    - K(C) is the number of characters in your prgram

Computing K(X)
- Undecidable
- A deep NN / transformers is a parallel computer that has a decent(but very finite) amount of resource(memory, sequential steps)
    - This is key!
    - The ability of NNs to simulate programs was known for decades
- SGD = searching ober the program/circuit
    - SGD miracle is that we can find circuits
