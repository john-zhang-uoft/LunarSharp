using System;
using System.Diagnostics;
using System.IO;

namespace NeuralSharp
{
    /// <summary>
    /// Neural network dense layer fully connected to the previous layer
    /// </summary>
    public class Dense : Layer
    {
        private Matrix previousWeightDelta;
        private Matrix previousBiasDelta;
        
        /// <summary>
        /// Constructor for dense layers.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="activation"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Dense(int shape, ActivationFunctions activation = ActivationFunctions.None) : base(
            (shape, 1, 1), activation)
        {
            if (shape < 1)
            {
                throw new InvalidDataException($"Invalid dense layer shape (shape = {shape}).");
            }
        }

        /// <summary>
        /// Constructor for the dense layer if it is the first layer in the network.
        /// </summary>
        /// <param name="inputShape"></param>
        /// <param name="shape"></param>
        /// <param name="activation"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Dense(int inputShape, int shape, ActivationFunctions activation = ActivationFunctions.None) : base(
            (inputShape, 1, 1), (shape, 1, 1), activation)
        {
            if (inputShape < 1 || shape < 1)
            {
                throw new InvalidDataException($"Invalid dense layer shape (input = {inputShape}, shape = {shape}).");
            }
        }

        /// <summary>
        /// Pass a set of inputs through this dense layer and update neuron activations.
        /// </summary>
        /// <param name="inputs"></param>
        /// <exception cref="InvalidDataException"></exception>
        public override void FeedForward(Matrix inputs)
        {
            if (inputs.Shape != (InputShape.Item1, InputShape.Item2))
            {
                throw new InvalidDataException(
                    $"Matrix shape is {inputs.Shape} while dense layer has input shape {InputShape}.");
            }

            Neurons = ActivationFunction(Weights * inputs + Biases);
        }

        /// <summary>
        /// Passes error back through this dense layer and updates weights and biases.
        /// </summary>
        /// <param name="nextLayer"></param>
        /// <param name="previousLayerNeurons"></param>
        /// <param name="target"></param>
        /// <param name="dLossFunction"></param>
        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target,
            Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            // Kronecker multiplication returns a matrix where the i-th row is the i-th neuron of the previous layer
            // multiplied by the gradient of the neurons of this layer
            // The element [i, j] in the matrix is the i-th input Neuron multiplied by the j-th Neuron's delta

            // But the element [j, i] of the weights matrix is the weight that connects the i-th input neuron
            // to the j-th neuron of this layer

            // So we transpose the Kronecker product to match the weights

            Gradient = nextLayer switch
            {
                null => dLossFunction(Neurons, target).HadamardMult(DerivativeActivationFunction(Neurons)),
                
                Dense => (nextLayer.Weights.Transpose() * nextLayer.Gradient).
                    HadamardMult(DerivativeActivationFunction(Neurons)),
                
                Dropout => nextLayer.Gradient.HadamardMult(DerivativeActivationFunction(nextLayer.Neurons))
            };

            DeltaWeight += Matrix.KroneckerVectorMult(previousLayerNeurons.Transpose(), Gradient).Transpose();
            DeltaBias += Gradient;
        }

        public override void ResetGradients()
        {
            Gradient = new Matrix(OutputShape.Item1, OutputShape.Item2);
            DeltaWeight = new Matrix(Weights.Shape.rows, Weights.Shape.cols);
            DeltaBias = new Matrix(Biases.Shape.rows, Biases.Shape.cols);
        }
        
    }
}