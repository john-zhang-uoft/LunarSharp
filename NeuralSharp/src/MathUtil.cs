using System;

namespace NeuralSharp
{
    public class MathUtil
    {
        /// <summary>
        /// Returns a matrix with a fraction successFrac of elements as 1s and the rest 0s.
        /// </summary>
        /// <param name="successFrac"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Matrix RandBernoulliDistribution(float successFrac, (int rows, int cols) shape)
        {
            (int rows, int cols) = shape;
            double needed = (int) Math.Round(successFrac * rows * cols);
            double available = rows * cols;
            
            Random rand = new Random();

            // Create array for matrix data
            float[] res = new float[rows * cols];
            
            // Iterate through matrix
            // The probability that an element is selected is needed / available,
            // Guaranteeing that the required number of elements are selected in one pass through
            for (int i = 0; i < res.Length; i++)
            {
                // If the element is randomly selected
                if (rand.NextDouble() < needed / available)
                {
                    res[i] = 1;
                    needed--;
                }
                else
                {
                    res[i] = 0;
                }

                available--;
            }

            return new Matrix(shape, res);
        }
    }
}