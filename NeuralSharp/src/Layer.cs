using System;

namespace NeuralSharp
{
    public class Weights
    {
        public static Matrix RandomizeWeights(float maxWeight, int rows, int cols)
        {   // Creates a matrix with random elements between -maxWeight and maxWeight
            
            float[] data = new float[rows * cols];

            Random randObj = new Random();
            
            for (int i = 0; i < rows * cols; i++)
            {
                data[i] = (float) (maxWeight * (randObj.NextDouble() * 2 - 1));
            }
            return new Matrix((rows, cols), data);
        }
    }
}