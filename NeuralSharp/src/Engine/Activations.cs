using System;

namespace NeuralSharp.Engine
{
    public class Activations
    {
        public float Sigmoid(float x)
        {
            return (float) (1 / (1 + Math.Exp(-x)));
        }

        public float DSigmoid(float x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public float Tanh(float x)
        {
            return (float) ((Math.Tanh(x) + 1) / 2);
        }

        public float DTanh(float x)
        {
            return (float) ((1 - Math.Pow(Math.Tanh(x), 2)) / 2);
        }

        public float ReLU(float x)
        {
            return (x > 0) ? x : 0;
        }

        public float DReLU(float x)
        {
            return (x > 0) ? 1 : 0;
        }
        
    }
}