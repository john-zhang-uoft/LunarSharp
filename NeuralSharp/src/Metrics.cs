using System;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public enum Metric
    {
        Accuracy,
        MeanSquaredError,
        None
    }


    public class Metrics
    {
        private const float Tolerance = 0.000001f;
        public static float Accuracy(Matrix[] outputs, Matrix[] targets)
        {
            if (outputs.Length != targets.Length)
            {
                throw new InvalidDataException("Matrices must be the same size for calculating accuracy.");
            }

            return outputs.Zip(targets, Accuracy).Sum() / outputs.Length;
        }
        // Optimized evaluating inside model.evaluate so we only iterate over outputs once
        public static float Accuracy(Matrix output, Matrix target)
        {
            return Encoder<Matrix>.ProbabilitiesToOneHot(output) == target ? 1f : 0f;
        }

        public static float MeanSquaredError(Matrix[] outputs, Matrix[] targets)
        {
            return outputs.Zip(targets, MeanSquaredError).Sum() / outputs.Length;
        }

        // Optimized evaluating inside model.evaluate so we only iterate over outputs once
        public static float MeanSquaredError(Matrix output, Matrix target)
        {
            return Loss.MeanSquaredError(output, target);
        }
    }
}