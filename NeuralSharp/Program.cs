using System;
using NeuralSharp.Tests;
using NeuralSharp;

namespace NeuralSharp2
{
    class Program
    {
        static void Main(string[] args)
        {
            Model test = new Model(
                new Dense(inputShape: 2, shape: 2, ActivationFunctions.Sigmoid),
                new Dense(shape: 256, ActivationFunctions.Sigmoid),
                new Dense(512, ActivationFunctions.Sigmoid),
                new Dense(shape: 16, ActivationFunctions.Sigmoid)
            );
            
            for (int i = 0; i < 1000; i++)
            {
                test.Layers[0].FeedForward(new Matrix(shape: (2, 1), 100, 100));
                test.Layers[1].FeedForward(test.Layers[0].Neurons);
                test.Layers[2].FeedForward(test.Layers[1].Neurons);
                test.Layers[3].FeedForward(test.Layers[2].Neurons);
                
                test.Layers[3].BackPropagate(null, test.Layers[2].Neurons, new Matrix((16, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.001f, 0.001f);
                test.Layers[2].BackPropagate(test.Layers[3], test.Layers[1].Neurons, new Matrix((16, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.001f, 0.001f);
                test.Layers[1].BackPropagate(test.Layers[2], test.Layers[0].Neurons, new Matrix((16, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.001f, 0.001f);
                test.Layers[0].BackPropagate(test.Layers[1], new Matrix((2, 1), 100, 100), new Matrix((16, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.001f, 0.001f);

                Console.WriteLine(Output.MeanSquaredError(test.Layers[3].Neurons, new Matrix((16, 1), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)));
            }
            
            Console.WriteLine(test.Layers[0].Weights.Data[0] + "\t" + test.Layers[0].Weights.Data[1] + "\t" + test.Layers[0].Biases.Data[0] + "\t" + test.Layers[0].Biases.Data[1]);
            Console.WriteLine(test.Layers[1].Weights.Data[0] + "\t" + test.Layers[1].Weights.Data[1] + "\t" + test.Layers[1].Biases.Data[0] + "\t" + test.Layers[1].Biases.Data[1]);
            Console.WriteLine(test.Layers[2].Weights.Data[0] + "\t" + test.Layers[2].Weights.Data[1] + "\t" + test.Layers[2].Biases.Data[0] + "\t" + test.Layers[2].Biases.Data[1]);

        }
    }
}