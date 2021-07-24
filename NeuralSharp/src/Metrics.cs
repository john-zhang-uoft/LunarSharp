using System;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public enum Metric
    {
        Accuracy, None
    }

    public class Metrics
    {
        private const float Tolerance = 0.000001f;
        public static float Accuracy(Matrix output, Matrix target)
        {
            if (output.Shape != target.Shape)
            {
                throw new InvalidDataException("Matrices must be the same size for calculating accuracy.");
            }
            
            return output.Data.Zip(target.Data,
                (outputElem, targetElem) => (Math.Abs(outputElem - targetElem) < Tolerance) ? 1f : 0f).Sum() / target.Data.Length;
        }
    }
}