using System;
using System.Linq;

namespace NeuralSharp
{
    public enum LossFunctions
    {
        MeanSquaredError,
    }

    public static class Loss
    {
        private const float Tolerance = 0.000001f;

        public static float MeanSquaredError(Matrix output, Matrix target)
        {
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating mean squared error");
            }

            return output.Data.Zip(target.Data,
                       (outputElem, targetElem) => (outputElem - targetElem) * (outputElem - targetElem)).Sum() /
                   target.Data.Length;
        }

        public static Matrix DMeanSquaredError(Matrix output, Matrix target)
        {
            // Returns a column vector containing the partial derivatives of the cost function with respect to each output neuron's brightness
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating mean squared error derivative");
            }

            return 2 * (output - target) / target.Data.Length;
        }

        /// <summary>
        /// Calculates binary cross entropy loss between matrices of zeros and ones.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static float BinaryCrossEntropy(Matrix output, Matrix target)
        {   // Matrix output and target must be 1 by 1 matrices
            // Target matrix must be either a 1 or 0
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating binary cross entropy.");
            }

            return (float) -(1.0 / target.Data.Length * output.Data.Zip(target.Data, (outputElem, targetElem) =>
                targetElem * Math.Log(Tolerance + outputElem) + (1 - targetElem) * Math.Log(Tolerance + 1 - outputElem)).Sum());
        }
        
        public static float DBinaryCrossEntropy(Matrix output, Matrix target)
        {   // Matrix output and target must be 1 by 1 matrices
            // Target matrix must be either a 1 or 0
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating binary cross entropy derivative.");
            }

            return output.Data.Zip(target.Data, (outputElem, targetElem) =>
                (outputElem - targetElem) / (outputElem * (1 - targetElem))).Sum();
        }
        
        /// <summary>
        /// Calculates categorical cross entropy loss between matrices of one hot encoded values for multi-class classification.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static float CategoricalCrossEntropy(Matrix output, Matrix target)
        {   // Target matrix must contain one hot encoded values
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException(
                    "Matrices must be the same size for calculating mean squared error derivative");
            }

            throw new NotImplementedException();
        }
    }
}