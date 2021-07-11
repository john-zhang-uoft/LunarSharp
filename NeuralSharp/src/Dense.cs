using System;
using System.Diagnostics;
using System.IO;

namespace NeuralSharp
{
    public class Dense : Layer
    {
        private Matrix Weights;
        private Matrix Biases;
        private ActivationFunctions ActivationFunction;

        public Dense(int inputShape, int outputShape, ActivationFunctions activation) : base((inputShape, 1, 1), (outputShape, 1, 1))
        {
            ActivationFunction = activation;
            Weights = RandomMatrix(1, inputShape, inputShape);
            Biases = RandomMatrix(1, inputShape, 1);
        }

        public Dense((int, int) inputShape, (int, int) outputShape) : base((inputShape.Item1, inputShape.Item2, 1),
            (outputShape.Item1, outputShape.Item2, 1))
        {
        }

        public Dense((int, int, int) inputShape, (int, int, int) outputShape) : base(inputShape, outputShape)
        {
        }

        public override Matrix FeedForward(Matrix inputs)
        {
            if (inputs.Shape != (InputShape.Item1, InputShape.Item2))
            {
                throw new InvalidDataException(
                    $"Matrix shape is {inputs.Shape} while dense layer has input shape {InputShape}");
            }

            return ActivationFunction switch
            {
                ActivationFunctions.Sigmoid => (Weights * inputs + Biases).ApplyToElements(Activations.Sigmoid),
                ActivationFunctions.Tanh => (Weights * inputs + Biases).ApplyToElements(Activations.Tanh),
                ActivationFunctions.ReLU => (Weights * inputs + Biases).ApplyToElements(Activations.ReLU),
                _ => throw new InvalidOperationException("Unimplemented Activation Function")
            };
        }

        public override void Update()
        {
            throw new NotImplementedException();
        }

        public override Matrix BackPropagate(Matrix inputs)
        {
            throw new NotImplementedException();
        }


        public static Matrix RandomMatrix(float maxWeight, int rows, int cols)
        {
            // Creates a matrix with random elements between -maxWeight and maxWeight

            float[] data = new float[rows * cols];

            Random randObj = new Random();

            for (int i = 0; i < rows * cols; i++)
            {
                data[i] = (float) (maxWeight * (randObj.NextDouble() * 2 - 1));
            }

            return new Matrix((rows, cols), data);
        }
    }
}