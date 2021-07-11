using System;
using System.Linq;

namespace NeuralSharp
{
    public partial class Matrix
    { 
        // Operator overrides
        
        public static Matrix operator -(Matrix a)
        {
            return new Matrix(a.Data.Select(i => -i), a.Shape);
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

            return new Matrix(a.Data.Zip(b.Data, (elemA, elemB) => elemA + elemB), a.Shape);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Shape != b.Shape)
            {
                throw new InvalidOperationException("Cannot subtract two matrices with different shape");
            }

            return new Matrix(a.Data.Zip(b.Data, (elemA, elemB) => elemA - elemB), a.Shape);
        }

        public static Matrix operator *(Matrix a, int b)
        {
            return new Matrix(a.Data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator *(Matrix a, float b)
        {
            return new Matrix(a.Data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator /(Matrix a, int b)
        {
            return new Matrix(a.Data.Select(i => i / b), a.Shape);
        }

        public static Matrix operator /(Matrix a, float b)
        {
            return new Matrix(a.Data.Select(i => i / b), a.Shape);
        }

        public static Matrix operator *(int b, Matrix a)
        {
            return new Matrix(a.Data.Select(i => i * b), a.Shape);
        }

        public static Matrix operator *(float b, Matrix a)
        {
            return new Matrix(a.Data.Select(i => i * b), a.Shape);
        }

        public static bool operator ==(Matrix a, Matrix b)
        {   // Returns true if the two matrices have the same reference or the same value
            if (ReferenceEquals(null, a))
            {
                throw new NullReferenceException("The first matrix is null");
            }

            if (ReferenceEquals(null, b))
            {
                throw new NullReferenceException("The second matrix is null");
            }
            if (ReferenceEquals(a, b)) { return true; }
            return a.Data.SequenceEqual(b.Data) && a.Shape.Equals(b.Shape);
        }

        public static bool operator !=(Matrix a, Matrix b)
        {   // Returns true if the two matrices have the same reference or the same value
            if (ReferenceEquals(null, a))
            {
                throw new NullReferenceException("The first matrix is null");
            }

            if (ReferenceEquals(null, b))
            {
                throw new NullReferenceException("The second matrix is null");
            }
            if (ReferenceEquals(a, b)) { return false; }
            return !a.Data.SequenceEqual(b.Data) || !a.Shape.Equals(b.Shape);
        }
    }
}