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
        public Dense(int shape, ActivationFunctions activation = ActivationFunctions.None) : base()
        {
            if (shape < 1)
            {
                throw new InvalidDataException("Invalid dense layer shape");
            }
            OutputShape = (shape, 1, 1);
            ActivationFunction = activation;
        }
        
        public Dense(int inputShape, int shape, ActivationFunctions activation = ActivationFunctions.None) : base()
        {
            if (shape < 1)
            {
                throw new InvalidDataException("Invalid dense layer shape");
            }

            if (inputShape < 1)
            {
                throw new InvalidDataException("Invalid dense layer shape");
            }
            
            OutputShape = (shape, 1, 1);
            InputShape = (inputShape, 1, 1);
            ActivationFunction = activation;
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
                ActivationFunctions.None => Weights * inputs + Biases,
                _ => throw new InvalidOperationException("Unimplemented Activation Function")
            };
        }

        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target, float alpha, float gamma)
        {
            SetGradient(nextLayer, target);
            UpdateWeights(alpha, previousLayerNeurons);
            UpdateBiases(gamma);
        }

        /// <summary>
        /// Calculates the gradient of the loss function with respect to the weights of the dense layer and updates weights.
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="lastLayerNeurons"></param>
        private void UpdateWeights(float alpha, Matrix lastLayerNeurons)
        {
            // Kronecker multiplication returns a matrix where the i-th row is the i-th neuron of the previous layer
            // multiplied by the gradient of the neurons of this layer
            // The element [i, j] in the matrix is the i-th input Neuron multiplied by the j-th Neuron's delta

            // But the element [j, i] of the weights matrix is the weight that connects the i-th input neuron
            // to the j-th neuron of this layer
            
            // So we transpose the Kronecker product to match the weights
            
            Weights -= alpha * Matrix.KroneckerVectorMult(lastLayerNeurons.Transpose(), Gradient).Transpose();
        }

        private void UpdateBiases(float gamma)
        {
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
                        .HadamardMult(Neurons.ApplyToElements(Activations.DTanh)),

                    ActivationFunctions.ReLU => Output.DMeanSquaredError(Neurons, target)
                        .HadamardMult(Neurons.ApplyToElements(Activations.DReLU)),
                    
                    ActivationFunctions.None => Output.DMeanSquaredError(Neurons, target)
                        .HadamardMult(Neurons),

                    _ => throw new InvalidOperationException("Unimplemented activation function")
                };
                return;
            }

            Gradient = ActivationFunction switch
            {
                ActivationFunctions.Sigmoid => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.DSigmoid)),

                ActivationFunctions.Tanh => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.DTanh)),

                ActivationFunctions.ReLU => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons.ApplyToElements(Activations.DReLU)),
                
                ActivationFunctions.None => (nextLayer.Weights.Transpose() * nextLayer.Gradient)
                    .HadamardMult(Neurons),

                _ => throw new InvalidOperationException("Unimplemented activation function")
            };
        }
        
    }
}