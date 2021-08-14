﻿using System;
using System.IO;
 
namespace NeuralSharp
{
    public class Dropout : Layer
    {
        private readonly float _rate;
        private readonly float _multiplier;

        /// <summary>
        /// Constructor for dropout layers.
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
            _multiplier = 1 / (1 - _rate);
        }

        /// <summary>
        /// Constructor for dropout layers.
        /// </summary>
        /// <param name="rate"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Dropout(float rate) : base()
        {   // Input and output shape are the same for dropout layers

            _rate = rate;
            _multiplier = 1 / (1 - _rate);
        }
        
        
        public override void FeedForward(Matrix inputs)
        {
            // Multiply inputs by the multiplier to keep the sum of the neurons the same
            Neurons = _multiplier * inputs.HadamardMult(MathUtil.RandBernoulliDistribution(_rate, inputs.Shape));
        }

        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target,
            Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            if (nextLayer == null)
            {
                Gradient = dLossFunction(Neurons, target);
            }
            else
            {
                Gradient = nextLayer.Weights.Transpose() * nextLayer.Gradient;
            }
        }

        public override void ResetGradients()
        {
        }
        
    }
}