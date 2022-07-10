using System;
using System.Reflection.Metadata;
using NeuralSharp;

namespace NeuralSharp
{
    public static class DeepNeuralNetwork
    {
        public static void Example(string[] args)
        {
            // Load Mnist dataset
            const string trainPath = @"C:\Users\johnz\RiderProjects\JohnsNeuralSharp\NeuralSharp\NeuralSharp\mnist_train.csv";
            var data = DataLoader.ReadCsv(trainPath, ",", numHeaderRows: 1);
            
            // Get features and labels
            var (y, x) = Matrix.ExtractCol(data, 0);

            // One-hot encode labels
            var encoder = new Encoder<Matrix>();
            y = encoder.ConfigureAndTransform(y);

            // Turn features into proper format
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = x[i].Transpose();
            }
            
            // Create dense model
            Model model = new Model(
                new Dense(784, shape: 64, ActivationFunctions.ReLU),
                new Dense(shape: 32, ActivationFunctions.ReLU),
                new Dense(shape: 10, ActivationFunctions.Sigmoid)
            );
            
            model.Compile(optimizer: new StochasticGD(alpha: 0.001f, gamma: 0.001f), LossFunctions.MeanSquaredError, new[] {Metric.None});
            model.Fit(x, y, epochs: 10, batchSize: 16, validationFrac: 0.2f, shuffle: false);
            
            // Load test data
            const string testPath = @"C:\Users\johnz\RiderProjects\JohnsNeuralSharp\NeuralSharp\NeuralSharp\mnist_test.csv";
            var testData = DataLoader.ReadCsv(testPath, ",", numHeaderRows: 1);

            var (testY, testX) = Matrix.ExtractCol(testData, 0);
            
            // One-hot encode labels with same encoder
            testY = encoder.Transform(testY);
            
            // Turn features into proper format
            for (int i = 0; i < testX.Length; i++)
            {
                testX[i] = testX[i].Transpose();
            }
            model.Evaluate(testX, testY, Array.Empty<Metric>());
            
        }
        
    }
}
