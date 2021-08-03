using System;
using System.IO;
 
namespace NeuralSharp
{
    public class Dropout : Layer
    {
        private readonly float _rate;
        
        /// <summary>
        /// Constructor for dense layers.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Dropout(int shape, float rate) : base((shape, 1, 1), (shape, 1, 1), ActivationFunctions.None)
        {   // Input and output shape are the same for dropout layers
            if (shape < 1)
            {
                throw new InvalidDataException($"Invalid dropout layer shape (shape = {shape}).");
            }

            _rate = rate;
        }


        public override void FeedForward(Matrix inputs)
        {
            int numToKeep = (int) Math.Round(InputShape.Item1 * _rate); // number of items to select

            // Multiply inputs by the multiplier to keep the sum of the neurons the same
            float multiplier = 1 / (1 - _rate);
            
            Neurons = 1 / (1 - _rate) * inputs.HadamardMult(MathUtil.RandBernoulliDistribution(_rate, inputs.Shape));
        }

        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target,
            Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            Gradient = 1 / (1 - _rate) *
                       nextLayer.Gradient.HadamardMult(MathUtil.RandBernoulliDistribution(_rate, nextLayer.Gradient.Shape));
        }
        
    }
}