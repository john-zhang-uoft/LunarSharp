using System;
using NeuralSharp.Tests;
using NeuralSharp;

namespace NeuralSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            Model test = new Model(
                new Dense(2, 2, ActivationFunctions.Sigmoid),
                new Dense(2, 2, ActivationFunctions.Sigmoid)
            );

            for (int i = 0; i < 10000; i++)
            {

                test.Layers[0].FeedForward(new Matrix(shape: (2, 1), 1, 1));
                test.Layers[1].FeedForward(test.Layers[0].Neurons);
                
                test.Layers[1].BackPropagate(null, test.Layers[1], new Matrix((2, 1), 2, 2), 0.01f, 0.01f);
                test.Layers[0].BackPropagate(test.Layers[1], null, new Matrix((2, 1), 2, 2), 0.01f, 0.01f);

                Console.WriteLine(Output.MeanSquaredError(test.Layers[1].Neurons, new Matrix((2, 1), 2, 2)));
            }
            
            Console.WriteLine(test.Layers[0].Weights.Data[0] + "\t" + test.Layers[0].Weights.Data[1] + "\t" + test.Layers[0].Biases.Data[0] + "\t" + test.Layers[0].Biases.Data[1]);
            Console.WriteLine(test.Layers[1].Weights.Data[0] + "\t" + test.Layers[1].Weights.Data[1] + "\t" + test.Layers[1].Biases.Data[0] + "\t" + test.Layers[1].Biases.Data[1]);
        }
    }
}