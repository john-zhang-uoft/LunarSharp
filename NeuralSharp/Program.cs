﻿using System;
using System.Text;
using NeuralSharp;

namespace NeuralSharp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Load Mnist dataset
            const string path = @"C:\Users\johnz\RiderProjects\NeuralSharp2\NeuralSharp2\mnist_test.csv";
            Matrix[] data = DataLoader.ReadCsv(path, ",", numHeaderRows: 1);

            // Get features and labels
            (Matrix[] y, Matrix[] x) = Matrix.ExtractCol(data, 0); 

            // One-hot encode labels
            y = Encoder<Matrix>.Encode(y);

            // Turn features into proper format
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = x[i].Transpose();
            }
            
            // Create dense model
            Model model = new Model(
                new Dense(inputShape: 784, shape: 64, ActivationFunctions.ReLU),
                new Dense(shape: 128, ActivationFunctions.ReLU),
                new Dense(shape: 128, ActivationFunctions.ReLU),
                new Dense(shape: 10, ActivationFunctions.Sigmoid)
            );
            
            model.Compile(Optimizer.None, LossFunctions.MeanSquaredError, new[] {Metric.None});
            model.Fit(x, y, epochs: 100, alpha: 0.001f, gamma: 0.001f, batchSize: x.Length / 8, validationFrac: 0.2f);

            Console.WriteLine(y[0]);
            Console.WriteLine(model.Predict(x[0]));
            
            Console.WriteLine(y[1]);
            Console.WriteLine(model.Predict(x[1]));
            
            Console.WriteLine(y[2]);
            Console.WriteLine(model.Predict(x[2]));
        }
        
    }
}