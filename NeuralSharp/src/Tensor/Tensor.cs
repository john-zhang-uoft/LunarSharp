using System;
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
            Data = new float[shape.Aggregate((product, next) => product * next )];
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
        
        /// <summary>
        /// Returns a copied version of the transposed Tensor. Only defined for 2D tensors.
        /// </summary>
        /// <returns></returns>
        public Tensor Transpose()
        {
            if (Shape.Length != 2)
            {
                throw new InvalidDataException("Tensor must be 2D to be transposed.");
            }
            
            Tensor temp = new Tensor(Shape);

            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    temp[j, i] = this[i, j];
                }
            }

            return temp;
        }
    }
}