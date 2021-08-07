using System;
using NeuralSharp;

namespace NeuralSharp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Load Mnist dataset
            string path = @"C:\Users\johnz\RiderProjects\NeuralSharp2\NeuralSharp2\mnist_test.csv";
            Matrix[] data = DataLoader.ReadCsv(path, ",", numHeaderRows: 1);

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
                new Dense(784, shape: 64, ActivationFunctions.ReLU),
                new Dropout(0.2f),
                new Dense(shape: 128, ActivationFunctions.ReLU),
                new Dropout(0.2f),
                new Dense(shape: 10, ActivationFunctions.Sigmoid)
            );
            
            model.Compile(Optimizer.None, LossFunctions.MeanSquaredError, new[] {Metric.None});
            model.Fit(x, y, epochs: 100, alpha: 0.01f, gamma: 0.01f, batchSize: x.Length / 8, validationFrac: 0.2f);

            model.Evaluate(x, y, Array.Empty<Metric>());
        }
    }
}