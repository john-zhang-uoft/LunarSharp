using System;
using System.IO;
using System.Security.Cryptography;

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

            // Store current number of needed items and number of available ones left to select from 
            double needed = numToKeep;
            double available = InputShape.Item1;

            Random rand = new Random();
            float[] neurons = new float[InputShape.Item1];

            // Iterate through matrix
            // The probability that an element is selected is needed / available,
            // Guaranteeing that the required number of elements are selected in one pass through
            for (int i = 0; i < InputShape.Item1; i++)
            {
                // If the element is randomly selected
                if (rand.NextDouble() < needed / available)
                {
                    neurons[i] = inputs[i, 0];
                    needed--;
                }
                else
                {
                    neurons[i] = 0;
                }

                available--;
            }

            Neurons = new Matrix((InputShape.Item1, 1), neurons);
        }

        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target, Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            throw new NotImplementedException();
        }
    }
}