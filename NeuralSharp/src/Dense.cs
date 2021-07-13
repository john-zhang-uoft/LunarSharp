using System;
using System.Diagnostics;
using System.IO;

namespace NeuralSharp
{
    public class Dense : Layer
    {
        public Dense(int inputShape, int outputShape, ActivationFunctions activation) : base((inputShape, 1, 1),
            (outputShape, 1, 1))
        {
            ActivationFunction = activation;
            Weights = RandomMatrix(1, inputShape, inputShape);
            Biases = RandomMatrix(1, inputShape, 1);
        }

        public override void FeedForward(Matrix inputs)
        {
            if (inputs.Shape != (InputShape.Item1, InputShape.Item2))
            {
                throw new InvalidDataException(
                    $"Matrix shape is {inputs.Shape} while dense layer has input shape {InputShape}");
            }

            Neurons = ActivationFunction switch
            {
                ActivationFunctions.Sigmoid => (Weights * inputs + Biases).ApplyToElements(Activations.Sigmoid),
                ActivationFunctions.Tanh => (Weights * inputs + Biases).ApplyToElements(Activations.Tanh),
                ActivationFunctions.ReLU => (Weights * inputs + Biases).ApplyToElements(Activations.ReLU),
                _ => throw new InvalidOperationException("Unimplemented Activation Function")
            };
        }
        
        public override void BackPropagate(Layer nextLayer, Matrix target, float alpha, float gamma)
        {
            SetGradient(nextLayer, target);
            Weights -= alpha * Gradient;
            Biases -= gamma * Gradient;
        }

        private void SetGradient(Layer nextLayer, Matrix target)
        {
            if (nextLayer == null)
            {
                Gradient = ActivationFunction switch
                {
                    ActivationFunctions.Sigmoid => Output.DMeanSquaredError(Neurons, target)
                        .HadamardMult(Neurons.ApplyToElements(Activations.DSigmoid)),

                    ActivationFunctions.Tanh => Output.DMeanSquaredError(Neurons, target)
                        .HadamardMult(Neurons.ApplyToElements(Activations.Tanh)),

                    ActivationFunctions.ReLU => Output.DMeanSquaredError(Neurons, target)
                        .HadamardMult(Neurons.ApplyToElements(Activations.ReLU)),
                    
                    _ => throw new InvalidOperationException("Unimplemented activation function")
                };
                return;
            }

            Gradient = ActivationFunction switch
            {
                ActivationFunctions.Sigmoid => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.DSigmoid)),

                ActivationFunctions.Tanh => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.Tanh)),

                ActivationFunctions.ReLU => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.ReLU)),

                _ => throw new InvalidOperationException("Unimplemented activation function")
            };
        }
        
        
        public static Matrix RandomMatrix(float maxWeight, int rows, int cols)
        {
            // Creates a matrix with random elements between -maxWeight and maxWeight

            float[] data = new float[rows * cols];

            Random randObj = new Random();

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float) (maxWeight * (randObj.NextDouble() * 2 - 1));
            }

            return new Matrix((rows, cols), data);
        }
    }
}