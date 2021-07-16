using System;
using NeuralSharp;

namespace NeuralSharp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Model model = new Model(
                new Dense(inputShape: 2, shape: 128, ActivationFunctions.Sigmoid),
                new Dense(shape: 256, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(shape: 2, ActivationFunctions.Sigmoid)
            );

            Matrix input1 = new Matrix((2, 1), 5, 25);
            Matrix output1 = new Matrix((2, 1), 0.7f, 0.2f);

            Matrix input2 = new Matrix((2, 1), 1, 144);
            Matrix output2 = new Matrix((2, 1), 0.4f, 0.1f);

            Matrix[] x = {input1, input2};
            Matrix[] y = {output1, output2};

            model.Compile(Optimizer.None, Loss.MeanSquareDError, new[] {Metric.None});
            model.Fit(x, y, epochs: 1000, alpha: 0.0001f, gamma: 0.0001f);

            Console.WriteLine(model.Predict(x[0]));
            Console.WriteLine(model.Predict(x[1]));
        }
    }
}