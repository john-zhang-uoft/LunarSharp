using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public class Encoder<T>
    {
        public int NumClasses { get; private set; }
        public T[] Classes { get; private set; }
        private Dictionary<T, Matrix> ToMatrixMap { get; set; }
        private Dictionary<Matrix, T> ToClassMap { get; set; }

        public Encoder()
        {
            ToMatrixMap = new Dictionary<T, Matrix>();
            ToClassMap = new Dictionary<Matrix, T>();
        }

        /// <summary>
        /// Fit the label encoder on a list of int class names.
        /// </summary>
        /// <param name="classLabels"></param>
        public void Configure(T[] classLabels)
        {
            if (classLabels == null)
            {
                throw new ArgumentNullException(nameof(classLabels));
            }

            if (classLabels.Length == 0)
            {
                throw new InvalidDataException("Attempting to configure encoder with an empty array.");
            }

            Classes = new HashSet<T>(classLabels).ToArray();
            NumClasses = Classes.Length;

            for (int i = 0; i < NumClasses; i++)
            {
                float[] encodedData = new float[NumClasses];
                encodedData[i] = 1;

                ToMatrixMap[Classes[i]] = new Matrix((NumClasses, 1), encodedData);
                ToClassMap[ToMatrixMap[Classes[i]]] = Classes[i];
            }
        }

        /// <summary>
        /// Fit the label encoder on a list of int class names. Checks whether the specified number of classes
        /// if equal to the actual number of classes.
        /// </summary>
        /// <param name="classLabels"></param>
        /// <param name="numClasses"></param>
        public void Configure(T[] classLabels, int numClasses)
        {
            if (classLabels == null)
            {
                throw new ArgumentNullException(nameof(classLabels));
            }

            if (classLabels.Length == 0)
            {
                throw new InvalidDataException("Attempting to configure encoder with an empty array.");
            }

            Classes = new HashSet<T>(classLabels).ToArray();
            NumClasses = numClasses;

            if (NumClasses != Classes.Length)
            {
                throw new InvalidDataException(
                    "Input numClasses is not equal to the number of classes found in dataset.");
            }

            for (int i = 0; i < NumClasses; i++)
            {
                float[] encodedData = new float[NumClasses];
                encodedData[i] = 1;

                ToMatrixMap[Classes[i]] = new Matrix((NumClasses, 1), encodedData);
                ToClassMap[ToMatrixMap[Classes[i]]] = Classes[i];
            }
        }

        /// <summary>
        /// Returns one-hot encoded labels given a list of labels. The encoder must already be configured.
        /// </summary>
        /// <param name="classLabels"></param>
        /// <returns></returns>
        public Matrix[] Transform(IEnumerable<T> classLabels)
        {
            if (classLabels == null)
            {
                throw new ArgumentNullException(nameof(classLabels));
            }

            if (ToMatrixMap == null)
            {
                throw new InvalidOperationException("Cannot transform classes without first configuring encoder.");
            }

            return classLabels.Select(i => ToMatrixMap[i]).ToArray();
        }

        /// <summary>
        /// Returns one-hot encoded label of a given class label. The encoder must already be configured.
        /// </summary>
        /// <param name="classLabel"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Matrix Transform(T classLabel)
        {
            if (classLabel == null)
            {
                throw new ArgumentNullException(nameof(classLabel));
            }

            if (ToMatrixMap == null)
            {
                throw new InvalidOperationException("Cannot transform class without first configuring encoder.");
            }

            return ToMatrixMap[classLabel];
        }

        /// <summary>
        /// Configures encoder on data and returns transformed data.
        /// </summary>
        /// <param name="labels"></param>
        /// <returns></returns>
        public Matrix[] ConfigureAndTransform(T[] labels)
        {
            Configure(labels);
            return Transform(labels);
        }

        /// <summary>
        /// Configures encoder on class labels and returns one-hot encoded labels.
        /// </summary>
        /// <param name="labels"></param>
        /// <returns></returns>
        public static Matrix[] Encode(T[] labels)
        {
            Encoder<T> encoder = new Encoder<T>();
            encoder.Configure(labels);
            return encoder.Transform(labels);
        }

        /// <summary>
        /// Given a label of class probabilities, returns the previously encoded corresponding class with the highest probability.
        /// </summary>
        /// <param name="oneHotLabel"></param>
        /// <returns></returns>
        public T Decode(Matrix oneHotLabel)
        {
            if (oneHotLabel == null)
            {
                throw new ArgumentNullException(nameof(oneHotLabel));
            }

            if (ToClassMap == null)
            {
                throw new InvalidOperationException("Cannot decode label without first configuring encoder.");
            }

            return ToClassMap[oneHotLabel];
        }

        /// <summary>
        /// Given a label of class probabilities, returns previously encoded corresponding class with the highest probabilities.
        /// </summary>
        /// <param name="oneHotLabels"></param>
        /// <returns></returns>
        public T[] Decode(IEnumerable<Matrix> oneHotLabels)
        {
            if (oneHotLabels == null)
            {
                throw new ArgumentNullException(nameof(oneHotLabels));
            }

            if (ToClassMap == null)
            {
                throw new InvalidOperationException("Cannot decode labels without first configuring encoder.");
            }

            return oneHotLabels.Select(i => ToClassMap[i]).ToArray();
        }

        /// <summary>
        /// Converts a column vector of probabilities into a one-hot encoded label based on the largest probability. 
        /// </summary>
        /// <param name="probabilities"></param>
        /// <returns></returns>
        public static Matrix ProbabilitiesToOneHot(Matrix probabilities)
        {
            float curMax = probabilities[0, 0];
            int curMaxInd = 0;

            for (int i = 1; i < probabilities.Shape.rows; i++)
            {
                if (probabilities[i, 0] > curMax)
                {
                    curMax = probabilities[i, 0];
                    curMaxInd = i;
                }
            }

            float[] encodedData = new float[probabilities.Shape.rows];
            encodedData[curMaxInd] = 1;

            return new Matrix((probabilities.Shape.rows, 1), encodedData);
        }

        /// <summary>
        /// Converts an array of column vectors of probabilities into one-hot encoded labels based on the largest probability.
        /// </summary>
        /// <param name="probabilities"></param>
        /// <returns></returns>
        public static IEnumerable<Matrix> ProbabilitiesToOneHot(Matrix[] probabilities)
        {
            for (int i = 0; i < probabilities.Length; i++)
            {
                yield return ProbabilitiesToOneHot(probabilities[i]);
            }
        }
    }
}