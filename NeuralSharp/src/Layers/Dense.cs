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
                throw new InvalidDataException("Invalid dense layer shape.");
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
                throw new InvalidDataException("Invalid dense layer shape.");
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

            Neurons = (Weights * inputs + Biases).ApplyToElements(ActivationFunction);
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
            SetNeuronsGradient(nextLayer, target, dLossFunction);

            // Kronecker multiplication returns a matrix where the i-th row is the i-th neuron of the previous layer
            // multiplied by the gradient of the neurons of this layer
            // The element [i, j] in the matrix is the i-th input Neuron multiplied by the j-th Neuron's delta

            // But the element [j, i] of the weights matrix is the weight that connects the i-th input neuron
            // to the j-th neuron of this layer

            // So we transpose the Kronecker product to match the weights

            DeltaWeight += Matrix.KroneckerVectorMult(previousLayerNeurons.Transpose(), Gradient).Transpose();
            DeltaBias += Gradient;
        }

        private void SetNeuronsGradient(Layer nextLayer, Matrix target, Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            if (nextLayer == null)
            {
                Gradient = dLossFunction(Neurons, target)
                    .HadamardMult(Neurons.ApplyToElements(ActivationFunction));
                return;
            }
            else
            {
                Gradient = (nextLayer.Weights.Transpose() * nextLayer.Gradient).HadamardMult(
                    Neurons.ApplyToElements(ActivationFunction));
            }
        }
    }
}