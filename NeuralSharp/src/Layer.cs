using System;

namespace NeuralSharp
{
    public abstract class Layer
    {
        protected Matrix Neurons;
        protected readonly (int, int, int) InputShape;
        protected readonly (int, int, int) OutputShape;

        public Matrix Weights { get; protected set; }
        public Matrix Biases { get; protected set; }
        public ActivationFunctions ActivationFunction { get; protected set; }

        public Matrix Gradient { get; protected set; }
        
        protected Layer((int, int, int) inputShape, (int, int, int) outputShape)
        {
            InputShape = inputShape;
            OutputShape = outputShape;
        }

        public abstract void FeedForward(Matrix inputs);
        public abstract void BackPropagate(Layer nextLayer, Matrix target, float alpha, float gamma);
        
    }
}