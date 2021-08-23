NeuralSharp

A native C# neural network library in progress.


Latest update: August 14, 2021 - SGD (Nesterov) Momentum optimizer.


My goal is to create a working high level C# neural network library with basic functionality (for fun)! I'm not following any tutorials that provide any code so that I can hone my fundamental neural network knowledge and practice creating a well-structured design.
Performance is not a top priority (otherwise C# would not be the language of choice), however, making computations within C# efficient is.
Everything is made from scratch, including the Matrix class, DataLoader, etc.

Current notable features:
1. (stochastic, mini-batch) Gradient descent.
2. Data loading from csv files.
3. Data encoder.
4. Dense and dropouts layers.

Plans:

1. Implement softmax activation and categorical cross entropy loss.
2. Implement saving model (as a csv file most likely).
3. Add optimizers.
4. Implement automatic differentiation.

![Image of NeuralSharp code](https://github.com/john-zhang-uoft/NeuralSharp/blob/master/NeuralSharp%20Picture.png)
