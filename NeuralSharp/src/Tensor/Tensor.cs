using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public class Tensor
    {
        /// <summary>
        /// The Tensor class is implemented as 1D arrays for better performance
        /// </summary>
        public float[] Data { get; private set; }

        public int[] Shape { get; private set; }

        public Tensor(params int[] shape)
        {
            Data = new float[shape.Aggregate((product, next) => product * next)];
            Shape = shape;
        }

        public Tensor(float[] data, params int[] shape)
        {
            Data = data;
            Shape = shape;
        }

        private Tensor(IEnumerable<float> data, int[] shape)
        {
            Data = data.ToArray();
            Shape = shape;
        }

        // Indexing
        public float this[params int[] i]
        {
            get => GetElement(i);
            set => SetElement(value, i);
        }

        public float GetElement(params int[] i)
        {
            if (i.Length != Shape.Length)
            {
                throw new InvalidDataException("Dimension of index does not match Tensor shape.");
            }

            for (int j = 0; j < i.Length; j++)
            {
                if (Shape[j] < i[j])
                {
                    throw new InvalidDataException($"Index[{j}] = {i[j]} exceeds Tensor shape.");
                }
            }

            int index = 0;

            for (int j = 0; j < i.Length; j++)
            {
                int add = i[j];

                for (int k = j + 1; k < i.Length; k++)
                {
                    add *= i[k];
                }

                index += add;
            }

            return Data[index];
        }


        private void SetElement(float value, params int[] i)
        {
            if (i.Length != Shape.Length)
            {
                throw new InvalidDataException("Dimension of index does not match Tensor shape.");
            }

            for (int j = 0; j < i.Length; j++)
            {
                if (Shape[j] < i[j])
                {
                    throw new InvalidDataException($"Index[{j}] = {i[j]} exceeds Tensor shape.");
                }
            }

            int index = 0;

            for (int j = 0; j < i.Length; j++)
            {
                int add = i[j];

                for (int k = j + 1; k < i.Length; k++)
                {
                    add *= i[k];
                }

                index += add;
            }

            Data[index] = value;
        }

        public Tensor ApplyToElements(Func<float, float> expression)
        {
            return new Tensor(Data.Select(expression), Shape);
        }

        public static Tensor HadamardMult(Tensor a, Tensor b)
        {
            // Element-wise multiplication
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for element-wise multiplication.");
            }

            return new Tensor(a.Data.Zip(b.Data, (elemA, elemB) => elemA * elemB), a.Shape);
        }

        /// <summary>
        /// Performs element-wise multiplication.
        /// </summary>
        /// <param name="b"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Tensor HadamardMult(Tensor b)
        {
            // Element-wise multiplication
            if (Shape != b.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for element-wise multiplication.");
            }

            return new Tensor(Data.Zip(b.Data, (elemA, elemB) => elemA * elemB), Shape);
        }

        /// <summary>
        /// Returns the sum of the all the elements of the tensor. 
        /// </summary>
        /// <returns></returns>
        public float SumElements()
        {
            float sum = 0;
            for (int i = 0; i < Data.Length; i++)
            {
                sum += Data[i];
            }

            return sum;
        }

        public static Tensor RandomTensor(float maxWeight, int[] shape)
        {
            // Creates a tensor with random elements between -maxWeight and maxWeight

            if (shape == null || shape.Length == 0 || shape.Any(x => x < 1))
            {
                throw new InvalidDataException("Invalid tensor shape.");
            }
            
            float[] data = new float[shape.Aggregate((product, next) => product * next)];

            Random randObj = new Random();

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(maxWeight * (randObj.NextDouble() * 2 - 1));
            }

            return new Tensor(data, shape);
        }
        
        
    }
}