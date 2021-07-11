using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public partial class Matrix
    {   
        /// <summary>
        /// The matrix class is implemented as 1D arrays for better performance compared to 2D arrays
        /// </summary>

        public readonly float[] Data;

        public readonly (int rows, int cols) Shape;

        public Matrix(int nRows, int nCols)
        {
            Data = new float[nRows * nCols];
            Shape = (nRows, nCols);
        }

        public Matrix((int nRows, int nCols) shape)
        {
            Data = new float[shape.nRows * shape.nCols];
            Shape = shape;
        }

        public Matrix(Matrix a)
        {
            if (a.Data.Length != a.Shape.rows * a.Shape.cols)
            {
                throw new InvalidDataException("Matrix shape does not match element data");
            }
            Data = a.Data;
            Shape = a.Shape;
        }

        public Matrix((int rows, int cols) shape, params float[] data)
        {
            if (data.Length != shape.rows * shape.cols)
            {
                throw new InvalidDataException("Matrix shape does not match element data");
            }
            
            Data = data;
            Shape = shape;
        }
        public Matrix(float[] data, (int rows, int cols) shape)
        {
            if (data.Length != shape.rows * shape.cols)
            {
                throw new InvalidDataException("Matrix shape does not match element data");
            }
            
            Data = data;
            Shape = shape;
        }

        public Matrix(IEnumerable<float> data, (int rows, int cols) shape)
        {
            Data = data.ToArray();

            if (Data.Length != shape.rows * shape.cols)
            {
                throw new InvalidDataException("Matrix shape does not match element data");
            }
            
            Shape = shape;
        }

        public float this[int i, int j]
        {
            get => Data[i * Shape.cols + j];
            private set => Data[i * Shape.cols + j] = value;
        }

        public Matrix ApplyToElements(Func<float, float> expression)
        {
            return new Matrix(Data.Select(expression), Shape);
        }
        
        public Matrix Transpose()
        {   // Returns a copied version of the transposed matrix
            Matrix temp = new Matrix(Shape);

            for (int i = 0; i < Shape.rows; i++)
            {
                for (int j = 0; j < Shape.cols; j++)
                {
                    temp[j, i] = this[i, j];
                }
            }

            return temp;
        }

        // Matrix Multiplications
        
        public static Matrix operator *(Matrix a, Matrix b)
        {   // Regular matrix multiplication
            if (a.Shape.cols != b.Shape.rows)
            {
                throw new InvalidOperationException("Invalid matrix shapes, cannot perform matrix multiplication");
            }

            Matrix res = new Matrix(a.Shape.rows, b.Shape.cols);

            for (int i = 0; i < a.Shape.rows; i++)
            {
                for (int j = 0; j < b.Shape.cols; j++)
                {
                    float dotProduct = 0;

                    for (int m = 0; m < a.Shape.cols; m++)
                    {
                        dotProduct += a[i, m] * b[m, j];
                    }

                    res[i, j] = dotProduct;
                }
            }
            return res;
        }

        
        public static Matrix HadamardMult(Matrix a, Matrix b)
        {   // Element-wise multiplication
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for element-wise multiplication");
            }
            return new Matrix(a.Data.Zip(b.Data, (elemA, elemB) => elemA * elemB), a.Shape);
        }


        public static Matrix KroneckerVectorMult(Matrix a, Matrix b)
        {   // Kronecker product of a row vector and column vector

            if (a.Shape.rows != 1 || b.Shape.cols != 1)
            {
                throw new InvalidOperationException(
                    "Kronecker product is only implemented between a row vector and column vector");
            }
            
            Matrix res = new Matrix(a.Shape.cols, b.Shape.rows);
            for (int i = 0; i < a.Shape.cols; i++)
            {
                for (int j = 0; j < b.Shape.rows; j++)
                {
                    res[i, j] = a[0, i] * b[j, 0];
                }
            }

            return res;
        }
        
        public static Matrix HorizontalConcat(Matrix a, Matrix b)
        {   // Concatenate the matrices horizontally and returns a new matrix
            if (a.Shape.rows != b.Shape.rows)
            {
                throw new InvalidOperationException("Matrices must have the same number of rows");
            }

            Matrix res = new Matrix(a.Shape.rows, a.Shape.cols + b.Shape.cols);
            
            // For each row
            for (int i = 0; i < a.Shape.rows; i++)
            {
                // Add the elements of that row of the first matrix
                for (int j = 0; j < a.Shape.cols; j++)
                {
                    res[i, j] = a[i, j];
                }
                // Add the elements of that row of the second matrix
                for (int k = 0; k < b.Shape.cols; k++)
                {
                    res[i, a.Shape.cols + k] = b[i, k];
                }
            }
            return res;
        }
        
        protected bool Equals(Matrix other)
        {   // Returns true if the two matrices have the same reference or the same value
            return Data.SequenceEqual(other.Data) && Shape.Equals(other.Shape);
        }

        public override bool Equals(object obj)
        {   // Returns true if the two matrices have the same reference or same value
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            return obj.GetType() == this.GetType() && Equals((Matrix) obj);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Data, Shape);
        }
    }
}