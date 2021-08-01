using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public class Encoder<T>
    {
        public int NumClasses { get; private set; }
        private Dictionary<T, Matrix> ToMatrixMap { get; set; }
        
        public Encoder()
        {
        }
        
        /// <summary>
        /// Fit the label encoder on a list of int class names.
        /// </summary>
        /// <param name="classList"></param>
        public void Configure(T[] classList)
        {
            ToMatrixMap = new Dictionary<T, Matrix>();
            classList = new HashSet<T>(classList).ToArray();
            NumClasses = classList.Length;
            
            for (int i = 0; i < NumClasses; i++)
            {
                float[] encodedData = new float[NumClasses];
                encodedData[i] = 1;
                
                ToMatrixMap[classList[i]] = new Matrix((NumClasses, 1), encodedData);
            }
        }

        /// <summary>
        /// Fit the label encoder on a list of int class names. Check whether the specified number of classes
        /// if equal to the actual number of classes.
        /// </summary>
        /// <param name="classList"></param>
        /// <param name="numClasses"></param>
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
                float[] encodedData = new float[NumClasses];
                encodedData[i] = 1;
                
                ToMatrixMap[classList[i]] = new Matrix((NumClasses, 1), encodedData);
            }
        }
        
        public Matrix[] Transform(T[] classList)
        {
            return classList.Select(i => ToMatrixMap[i]).ToArray();
        }
    }
}