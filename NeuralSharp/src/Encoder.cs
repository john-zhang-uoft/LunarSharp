using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public class Encoder<T>
    {
        public int NumClasses { get; private set; }
        private Dictionary<string, Matrix> _stringToMatrixMap { get; set; }
        private Dictionary<int, Matrix> _intToMatrixMap { get; set; }
        private Dictionary<T, Matrix> _toMatrixMap { get; set; }
        
        public Encoder()
        {
        }
        
        /// <summary>
        /// Fit the label encoder on a list of string class names.
        /// </summary>
        /// <param name="classList"></param>
        public void Configure(string[] classList)
        {
            classList = new HashSet<string>(classList).ToArray();
            NumClasses = classList.Length;

            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _stringToMatrixMap[classList[i]] = encoded;
            }
        }

        /// <summary>
        /// Fit the label encoder on a list of string class names. Check whether the specified number of classes
        /// if equal to the actual number of classes.
        /// </summary>
        /// <param name="classList"></param>
        public void Configure(string[] classList, int numClasses)
        {
            NumClasses = numClasses;

            classList = new HashSet<string>(classList).ToArray();

            if (NumClasses != classList.Length)
            {
                throw new InvalidDataException(
                    "Input numClasses is not equal to the number of classes found in dataset.");
            }
            
            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _stringToMatrixMap[classList[i]] = encoded;
            }
        }

        /// <summary>
        /// Fit the label encoder on a list of int class names.
        /// </summary>
        /// <param name="classList"></param>
        public void Configure(int[] classList)
        {
            classList = new HashSet<int>(classList).ToArray();
            NumClasses = classList.Length;

            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _intToMatrixMap[classList[i]] = encoded;
            }
        }
        
        /// <summary>
        /// Fit the label encoder on a list of int class names. Check whether the specified number of classes
        /// if equal to the actual number of classes.
        /// </summary>
        /// <param name="classList"></param>
        public void Configure(int[] classList, int numClasses)
        {
            NumClasses = numClasses;
            
            classList = new HashSet<int>(classList).ToArray();
            
            if (NumClasses != classList.Length)
            {
                throw new InvalidDataException(
                    "Input numClasses is not equal to the number of classes found in dataset.");
            }

            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _intToMatrixMap[classList[i]] = encoded;
            }
            
        }

        public void Configure(T[] classList)
        {
            classList = new HashSet<T>(classList).ToArray();
            NumClasses = classList.Length;

            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _toMatrixMap[classList[i]] = encoded;
            }
        }

        public void Configure(T[] classList, int numClasses)
        {
            classList = new HashSet<T>(classList).ToArray();
            NumClasses = numClasses;
            
            if (NumClasses != classList.Length)
            {
                throw new InvalidDataException(
                    "Input numClasses is not equal to the number of classes found in dataset.");
            }
            
            for (int i = 0; i < NumClasses; i++)
            {
                Matrix encoded = new Matrix(NumClasses, 1) {[i, 1] = 1};
                _toMatrixMap[classList[i]] = encoded;
            }
        }
        
        public Matrix[] Transform(int[] classList)
        {
            return classList.Select(i => _intToMatrixMap[i]).ToArray();
        }
        
        public Matrix[] Transform(string[] classList)
        {
            return classList.Select(i => _stringToMatrixMap[i]).ToArray();
        }

        public Matrix[] Transform(T[] classList)
        {
            return classList.Select(i => _toMatrixMap[i]).ToArray();
        }
    }
}