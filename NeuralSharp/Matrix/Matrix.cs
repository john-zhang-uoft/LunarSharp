using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;

namespace NeuralSharp
{
    public partial class Matrix
    {   
        /// <summary>
        /// The matrix class is implemented as 1D arrays for better performance compared to 2D arrays
        /// </summary>

        private readonly float[] _data;
        public (int rows, int cols) Shape { get; private set; }

        public Matrix(int nRows, int nCols)
        {
            _data = new float[nRows * nCols];
            Shape = (nRows, nCols);
        }

        public Matrix(Matrix a)
        {
            _data = a._data;
            Shape = a.Shape;
        }

        public Matrix(float[] data, (int rows, int cols) shape)
        {
            _data = data;
            Shape = shape;
        }

        public Matrix(IEnumerable<float> data, (int rows, int cols) shape)
        {
            _data = data.ToArray();
            Shape = shape;
        }

        public float this[int i, int j]
        {
            get => _data[i * Shape.cols + j];
            private set => _data[i * Shape.cols + j] = value;
        }

        public Matrix Transpose()
        {   // Returns a copied version of the transposed matrix
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

        
        // Operator overrides
        
        public static Matrix operator -(Matrix a)
        {
            return new Matrix(a._data.Select(i => -i), a.Shape);
        }

        public static Matrix operator +(Matrix a)
        {
            return new Matrix(a);
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Cannot add two matrices with different shape");
            }

            return new Matrix(a._data.Zip(b._data, (elemA, elemB) => elemA + elemB), a.Shape);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Cannot subtract two matrices with different shape");
            }

            return new Matrix(a._data.Zip(b._data, (elemA, elemB) => elemA - elemB), a.Shape);
        }

        public static Matrix operator *(Matrix a, int b)
        {
            return new Matrix(a._data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator *(Matrix a, float b)
        {
            return new Matrix(a._data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator /(Matrix a, int b)
        {
            return new Matrix(a._data.Select(i => i / b), a.Shape);
        }

        public static Matrix operator /(Matrix a, float b)
        {
            return new Matrix(a._data.Select(i => i / b), a.Shape);
        }

        public static Matrix operator *(int b, Matrix a)
        {
            return new Matrix(a._data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator *(float b, Matrix a)
        {
            return new Matrix(a._data.Select(i => i * b), a.Shape);
        }

        public static bool operator ==(Matrix a, Matrix b)
        {
            return (a._data == b._data && a.Shape == b.Shape);
        }

        public static bool operator !=(Matrix a, Matrix b)
        {
            return (a._data != b._data || a.Shape != b.Shape);
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
                    float dot_product = 0;

                    for (int m = 0; m < a.Shape.cols; m++)
                    {
                        dot_product += a[i, m] * b[m, j];
                    }

                    res[i, j] = dot_product;
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
            return new Matrix(a._data.Zip(b._data, (elemA, elemB) => elemA * elemB), a.Shape);
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
        {
            return Equals(_data, other._data) && Shape.Equals(other.Shape);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Matrix) obj);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(_data, Shape);
        }
    }
}