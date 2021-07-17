using System;
using System.Linq;

namespace NeuralSharp
{
    public enum LossFunctions
    {
        MeanSquareDError,
    }

    public static class Loss
    {
        public static float MeanSquaredError(Matrix output, Matrix target)
        {
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating mean squared error");
            }

            return output.Data.Zip(target.Data,
                (outputElem, targetElem) => (outputElem - targetElem) * (outputElem - targetElem)).Sum();
        }

        public static Matrix DMeanSquaredError(Matrix output, Matrix target)
        {
            // Returns a column vector containing the partial derivatives of the cost function with respect to each output neuron's brightness
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating mean squared error derivative");
            }

            return 2 * (output - target);
        }


        public static float Accuracy(Matrix output, Matrix target)
        {
            return 0;
        }
    }
}