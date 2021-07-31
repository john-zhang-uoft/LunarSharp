using System;

namespace NeuralSharp
{
    public enum ActivationFunctions
    {
        Sigmoid,
        Tanh,
        ReLU,
        None
    }

    public static class Activations
    {
        private static float Sigmoid(float x)
        {
            return (float) (1 / (1 + Math.Exp(-x)));
        }

        private static float DSigmoid(float x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        public static Matrix Sigmoid(Matrix x)
        {
            return x.ApplyToElements(Sigmoid);
        }

        public static Matrix DSigmoid(Matrix x)
        {
            return x.ApplyToElements(DSigmoid);
        }

        private static float Tanh(float x)
        {
            return (float) ((Math.Tanh(x) + 1) / 2);
        }
        
        private static float DTanh(float x)
        {
            return (float) ((1 - Math.Pow(Math.Tanh(x), 2)) / 2);
        }
        
        public static Matrix Tanh(Matrix x)
        {
            return x.ApplyToElements(Tanh);
        }

        public static Matrix DTanh(Matrix x)
        {
            return x.ApplyToElements(DTanh);
        }
        
        private static float ReLU(float x)
        {
            return (x > 0) ? x : 0;
        }

        private static float DReLU(float x)
        {
            return (x > 0) ? 1 : 0;
        }

        public static Matrix ReLU(Matrix x)
        {
            return x.ApplyToElements(ReLU);
        }

        public static Matrix DReLU(Matrix x)
        {
            return x.ApplyToElements(DReLU);
        }
        
        private static float None(float x)
        {
            return x;
        }

        private static float DNone(float x)
        {
            return 1;
        }
        
        public static Matrix None(Matrix x)
        {
            return x;
        }

        public static Matrix DNone(Matrix x)
        {
            return Matrix.MakeFullMatrixOfNum(x.Shape, 1);
        }

        public static Matrix Softmax(Matrix x)
        {
            Matrix res = x.ApplyToElements(e => (float) Math.Exp(e));
            return res / res.SumElements();
        }

        public static Matrix DSoftmax(Matrix x)
        {
            throw new NotImplementedException();
        }
    }
}