using System;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public enum LossFunctions
    {
        MeanSquaredError,
        BinaryCrossEntropy
    }

    public static class Loss
    {
        private const float Tolerance = 0.000001f;

        public static float MeanSquaredError(Matrix output, Matrix target)
        {
            if (output.Shape != target.Shape)
            {
                throw new InvalidDataException(
                    "Matrices must be the same size for calculating mean squared error.");
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
                throw new InvalidDataException(
                    "Matrices must be the same size for calculating mean squared error derivative.");
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
        {
            // Matrix output and target must be 1 by 1 matrices
            // Target matrix must be either a 1 or 0
            if (output.Shape != (1, 1) || target.Shape != (1, 1))
            {
                throw new InvalidDataException(
                    "Matrices must be 1 by 1 for calculating binary cross entropy.");
            }

            return (float) -(target[0, 0] * Math.Log(Tolerance + output[0, 0]) +
                             (1 - target[0, 0]) * Math.Log(Tolerance + 1 - output[0, 0]));
        }

        public static Matrix DBinaryCrossEntropy(Matrix output, Matrix target)
        {
            // Matrix output and target must be 1 by 1 matrices
            // Target matrix must be either a 1 or 0
            if (output.Shape != (1, 1) || target.Shape != (1, 1))
            {
                throw new InvalidDataException(
                    "Matrices must be 1 by 1 for calculating binary cross entropy.");
            }

            return (output - target) / (output[0, 0] * (1 - output[0, 0]) + Tolerance);
        }

        public static float CategoricalCrossEntropy(Matrix output, Matrix target)
        {
            // Matrix out and target must be the same shape
            if (output.Shape != target.Shape)
            {
                throw new InvalidDataException(
                    "Matrices must be the same shape for calculating categorical cross entropy");
            }

            // return -output.ApplyToElements(e => (float) Math.Log(e)).HadamardMult(target).SumElements();

            return (float) -output.Data.Zip(target.Data, (outputElem, targetElem)
                => targetElem * Math.Log(outputElem + Tolerance)).Sum() / output.Data.Length;
        }

        public static float DCategoricalCrossEntropy(Matrix output, Matrix target)
        {
            return 0;
        }
        
    }
}