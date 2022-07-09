NeuralSharp

A native C# neural network library in progress.


Latest update: July 9, 2022 - Restructured forward and backward passes and exposed training methods for custom training loops, created Generative Adversarial Network (GAN) example.

My goal is to create a working high level C# neural network library with basic functionality (for fun)! I'm not following any tutorials that provide any code so that I can hone my fundamental neural network knowledge and practice creating a well-structured design.
Performance is not a top priority (otherwise C# would not be the language of choice), however, making computations within C# efficient is.
Everything is made from scratch, including the Matrix class, DataLoader, etc.

Current notable features:
** Generative Adversarial Network example**
1. (stochastic, mini-batch) Gradient descent.
2. Data loading from csv files.
3. Data encoder.
4. Dense and dropouts layers.

Plans:

1. Implement softmax activation and categorical cross entropy loss.
2. Implement saving model (as a csv file most likely).
3. Implement automatic differentiation.
4. Implement graph neural networks.
5. Implement an autoencoder example
6. Implement basic Recurrent Neural Network and LTSM model

![Image of NeuralSharp code](https://github.com/john-zhang-uoft/NeuralSharp/blob/master/NeuralSharp%20Picture.png)
