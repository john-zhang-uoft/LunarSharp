using System;
using System.Linq;

namespace NeuralSharp.Matrix
{
    public partial class Matrix
    { 
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
            return a?._data == b?._data && a?.Shape == b?.Shape;
        }

        public static bool operator !=(Matrix a, Matrix b)
        {
            return a?._data != b?._data || a?.Shape != b?.Shape;
        }
    }
}