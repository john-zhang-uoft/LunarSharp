using System;
using System.Reflection.Metadata;
using NeuralSharp;

namespace NeuralSharp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Load Mnist dataset
            const string trainPath = @"C:\Users\johnz\RiderProjects\NeuralSharp2\NeuralSharp2\mnist_train.csv";
            Matrix[] data = DataLoader.ReadCsv(trainPath, ",", numHeaderRows: 1);
            
            // Get features and labels
            (Matrix[] y, Matrix[] x) = Matrix.ExtractCol(data, 0);

            // One-hot encode labels
            Encoder<Matrix> encoder = new Encoder<Matrix>();
            y = encoder.ConfigureAndTransform(y);

            // Turn features into proper format
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = x[i].Transpose();
            }
            
            // Create dense model
            Model model = new Model(
                new Dense(784, shape: 128, ActivationFunctions.ReLU),
                new Dense(shape: 128, ActivationFunctions.ReLU),
                new Dropout(0.2f),
                new Dense(shape: 64, ActivationFunctions.ReLU),
                new Dense(shape: 10, ActivationFunctions.Sigmoid)
            );
            
            model.Compile(optimizer: new StochasticGD(alpha: 0.001f, beta: 0.001f, momentum: 0.9f), LossFunctions.MeanSquaredError, new[] {Metric.None});
            model.Fit(x, y, epochs: 10, batchSize: 16, validationFrac: 0.2f);

            
            // Load test data
            const string testPath = @"C:\Users\johnz\RiderProjects\NeuralSharp2\NeuralSharp2\mnist_test.csv";
            Matrix[] testData = DataLoader.ReadCsv(testPath, ",", numHeaderRows: 1);

            (Matrix[] testY, Matrix[] testX) = Matrix.ExtractCol(testData, 0);
            
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