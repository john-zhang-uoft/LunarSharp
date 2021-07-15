using System;
using NeuralSharp.Tests;
using NeuralSharp;

namespace NeuralSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            Model model = new Model(
                new Dense(inputShape: 2, shape: 2, ActivationFunctions.Sigmoid),
                new Dense(shape: 256, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(shape: 3, ActivationFunctions.Sigmoid)
            );
            
            Matrix input = new Matrix((2, 1), 100, 100);
            Matrix output = new Matrix((3, 1), 1, 1, 1);
            Matrix[] x = new[] {input};
            Matrix[] y = new[] {output};
            
            model.Compile(Optimizer.None, Loss.MeanSquareDError, new Metric[] {Metric.None});
            model.Fit(x, y, epochs: 100);
        }
    }
}