using System;

namespace NeuralSharp
{
    public abstract class Layer
    {
        protected Matrix Neurons;
        protected readonly (int, int, int) InputShape;
        protected readonly (int, int, int) OutputShape;
        protected Layer((int, int, int) inputShape, (int, int, int) outputShape)
        {
            InputShape = inputShape;
            OutputShape = outputShape;
        }
        public abstract void Update();
        public abstract Matrix FeedForward(Matrix inputs);
        public abstract Matrix BackPropagate(Matrix inputs);
    }
}