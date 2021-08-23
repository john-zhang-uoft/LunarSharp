using System;
using System.IO;

namespace NeuralSharp
{
    public class Convolution2D : Layer
    {
        #region Constructors

        /// <summary>
        /// Constructor for dense layers.
        /// </summary>
        /// <param name="bias"></param>
        /// <param name="activation"></param>
        /// <param name="numFilters"></param>
        /// <param name="kernelSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Convolution2D(int numFilters, (int, int) kernelSize, int stride, int padding, int dilation, bool bias,
            ActivationFunctions activation = ActivationFunctions.None) : base(
            (kernelSize.Item1, kernelSize.Item2, 1), activation)
        {
            if (kernelSize.Item1 < 1 || kernelSize.Item2 < 1)
            {
                throw new InvalidDataException(
                    $"Invalid Convolution2D layer shape (shape = {kernelSize.Item1}, {kernelSize.Item2}).");
            }
        }

        /// <summary>
        /// Constructor for dense layers.
        /// </summary>
        /// <param name="bias"></param>
        /// <param name="activation"></param>
        /// <param name="inputShape"></param>
        /// <param name="numFilters"></param>
        /// <param name="kernelSize"></param>
        /// <param name="stride"></param>
        /// <param name="padding"></param>
        /// <param name="dilation"></param>
        /// <exception cref="InvalidDataException"></exception>
        public Convolution2D((int, int, int) inputShape, int numFilters, (int, int) kernelSize, int stride, int padding, int dilation, bool bias,
            ActivationFunctions activation = ActivationFunctions.None) : base(
            (kernelSize.Item1, kernelSize.Item2, 1), activation)
        {
            if (kernelSize.Item1 < 1 || kernelSize.Item2 < 1)
            {
                throw new InvalidDataException(
                    $"Invalid Convolution2D layer shape (shape = {kernelSize.Item1}, {kernelSize.Item2}).");
            }
        }

        #endregion

        public override void FeedForward(Matrix inputs)
        {
            throw new NotImplementedException();
        }

        public override void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target,
            Func<Matrix, Matrix, Matrix> dLossFunction)
        {
            throw new NotImplementedException();
        }

        public override void ResetGradients()
        {
            throw new NotImplementedException();
        }
    }
}