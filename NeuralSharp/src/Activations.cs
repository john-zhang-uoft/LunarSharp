using System;

namespace NeuralSharp
{
    public enum ActivationFunctions
    {
        Sigmoid,
        Tanh,
        ReLU,
    }
    
    public static class Activations
    {
        public static float Sigmoid(float x)
        {
            return (float) (1 / (1 + Math.Exp(-x)));
        }
        
        public static float DSigmoid(float x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static float Tanh(float x)
        {
            return (float) ((Math.Tanh(x) + 1) / 2);
        }

        public static float DTanh(float x)
        {
            return (float) ((1 - Math.Pow(Math.Tanh(x), 2)) / 2);
        }

        public static float ReLU(float x)
        {
            return (x > 0) ? x : 0;
        }

        public static float DReLU(float x)
        {
            return (x > 0) ? 1 : 0;
        }
        
    }
}