using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

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
            set => Data[i * Shape.cols + j] = value;
        }

        public Matrix ApplyToElements(Func<float, float> expression)
        {
            return new Matrix(Data.Select(expression), Shape);
        }

        public Matrix Transpose()
        {
            // Returns a copied version of the transposed matrix
            Matrix temp = new Matrix(Shape.cols, Shape.rows);

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
        {
            // Regular matrix multiplication
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
        {
            // Element-wise multiplication
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for element-wise multiplication");
            }

            return new Matrix(a.Data.Zip(b.Data, (elemA, elemB) => elemA * elemB), a.Shape);
        }

        public Matrix HadamardMult(Matrix b)
        {
            // Element-wise multiplication
            if (Shape != b.Shape)
            {
                throw new InvalidOperationException("Matrices must be the same size for element-wise multiplication");
            }

            return new Matrix(Data.Zip(b.Data, (elemA, elemB) => elemA * elemB), Shape);
        }

        /// <summary>
        /// Returns the Kronecker product of a row vector "a" and column vector "b".
        /// The i-th row of the resulting matrix is the i-th element of "a" multiplied by "b" transpose.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>Matrix with shape i x j where i is the number of cols of matrix "a" and j is the number of rows of matrix "b"
        /// </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Matrix KroneckerVectorMult(Matrix a, Matrix b)
        {
            // Check whether a is a row vector and b a column vector
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

        public Matrix KroneckerVectorMult(Matrix b)
        {
            // Kronecker product of a row vector and column vector
            if (Shape.rows != 1 || b.Shape.cols != 1)
            {
                throw new InvalidOperationException(
                    "Kronecker product is only implemented between a row vector and column vector");
            }

            Matrix res = new Matrix(Shape.cols, b.Shape.rows);
            for (int i = 0; i < Shape.cols; i++)
            {
                for (int j = 0; j < b.Shape.rows; j++)
                {
                    res[i, j] = this[0, i] * b[j, 0];
                }
            }

            return res;
        }

        public static Matrix HorizontalConcat(Matrix a, Matrix b)
        {
            // Concatenate the matrices horizontally and returns a new matrix
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

        public static Matrix RandomMatrix(float maxWeight, int rows, int cols)
        {
            // Creates a matrix with random elements between -maxWeight and maxWeight

            float[] data = new float[rows * cols];

            Random randObj = new Random();

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float) (maxWeight * (randObj.NextDouble() * 2 - 1));
            }

            return new Matrix((rows, cols), data);
        }

        public override string ToString()
        {
            StringBuilder str = new StringBuilder();
            str.Append('[');

            for (int m = 0; m < Shape.rows - 1; m++)
            {
                str.Append('[');

                for (int n = 0; n < Shape.cols - 1; n++)
                {
                    str.Append($"{this[m, n]}, ");
                }

                str.Append($"{this[m, Shape.cols - 1]}], \n");
            }

            str.Append('[');
            for (int n = 0; n < Shape.cols - 1; n++)
            {
                str.Append($"{this[Shape.rows - 1, n]}, ");
            }

            str.Append($"{this[Shape.rows - 1, Shape.cols - 1]}]]");

            return str.ToString();
        }

        private bool Equals(Matrix other)
        {
            // Returns true if the two matrices have the same reference or the same value
            return Data.SequenceEqual(other.Data) && Shape.Equals(other.Shape);
        }

        public override bool Equals(object obj)
        {
            // Returns true if the two matrices have the same reference or same value
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