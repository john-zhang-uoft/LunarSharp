using System;
using System.Linq;

namespace NeuralSharp
{
    public class Output
    {
        public static float MeanSquaredError(Matrix output, Matrix target)
        {
            if (output.Shape != target.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for calculating mean squared error");
            }

            return output.Data.Zip(target.Data,
                (outputElem, targetElem) => (outputElem - targetElem) * (outputElem - targetElem)).Sum();
        }
        
        
    }
}