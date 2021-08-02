using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public static class Sampling
    {
        private static readonly Random Rand = new Random();

        /// <summary>
        /// Randomly orders the elements in a list and does not return a new list.
        /// </summary>
        /// <param name="list"></param>
        /// <typeparam name="T"></typeparam>
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = Rand.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        /// <summary>
        /// Randomly orders the elements in a two lists such that the elements
        /// in those lists that match indices still match indices. Does not return a new list.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <typeparam name="T"></typeparam>
        public static void Shuffle<T>(T[] x, T[] y)
        {
            if (x.Length != y.Length)
            {
                throw new InvalidDataException("X and y must be the same size.");
            }

            int n = x.Length;

            while (n > 1)
            {
                n--;
                int k = Rand.Next(n + 1);
                T xValue = x[k];
                T yValue = y[k];

                x[k] = x[n];
                y[k] = y[n];

                x[n] = xValue;
                y[n] = yValue;
            }
        }

        /// <summary>
        /// Returns list of n randomly chosen elements in a random order.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="n"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static IEnumerable<T> Sample<T>(this IEnumerable<T> list, int n)
        {
            Random rand = new Random();

            return list.OrderBy(x => rand.Next()).Take(n);
        }

        /// <summary>
        /// Returns a training list and a validation list of elements with sizes based on validationFrac.
        /// </summary>
        /// <param name="list"></param>
        /// <param name="validationFrac"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static (List<T> train, List<T> val) TrainValSplit<T>(T[] list, float validationFrac)
        {
            int numVal = (int) Math.Round(list.Length * validationFrac); // number of items to select
            List<T> train = new List<T>();
            List<T> val = new List<T>();

            // Store current number of needed items and number of available ones left to select from 
            double needed = numVal;
            double available = list.Length;

            // Iterate through list
            // The probability that an element is selected is needed / available,
            // Guaranteeing that the required number of elements are selected in one pass through
            for (int i = 0; i < list.Length; i++)
            {
                // If the element is randomly selected
                if (Rand.NextDouble() < needed / available)
                {
                    val.Add(list[i]);
                    needed--;
                }
                else
                {
                    train.Add(list[i]);
                }

                available--;
            }

            return (train, val);
        }

        /// <summary>
        /// Returns a training list and a validation list of elements with sizes based on validationFrac.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="validationFrac"></param>
        /// <param name="x"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static (List<T> xTrain, List<T> yTrain, List<T> xVal, List<T> yVal) TrainValSplitList<T>(T[] x, T[] y,
            float validationFrac)
        {
            if (x.Length != y.Length)
            {
                throw new InvalidDataException("x and y are not the same length.");
            }

            List<T> xTrain = new List<T>();
            List<T> yTrain = new List<T>();

            List<T> xVal = new List<T>();
            List<T> yVal = new List<T>();

            // Store current number of needed items and number of available ones left to select from 

            double needed = (int) Math.Round(x.Length * validationFrac); // number of items to select;
            double available = x.Length;

            // Iterate through list
            // The probability that an element is selected is needed / available,
            // Guaranteeing that the required number of elements are selected in one pass through
            for (int i = 0; i < x.Length; i++)
            {
                // If the element is randomly selected
                if (Rand.NextDouble() < needed / available)
                {
                    xVal.Add(x[i]);
                    yVal.Add(y[i]);
                    needed--;
                }
                else
                {
                    xTrain.Add(x[i]);
                    yTrain.Add(y[i]);
                }

                available--;
            }

            return (xTrain, yTrain, xVal, yVal);
        }

        /// <summary>
        /// Returns a training list and a validation list of elements with sizes based on validationFrac.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="validationFrac"></param>
        /// <param name="x"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static (T[] xTrain, T[] yTrain, T[] xVal, T[] yVal) TrainValSplit<T>(T[] x, T[] y, float validationFrac)
        {
            if (x.Length != y.Length)
            {
                throw new InvalidDataException("x and y are not the same length.");
            }
            // Store current number of needed items and number of available ones left to select from 

            double neededVal = (int) Math.Round(x.Length * validationFrac); // number of items to select for val;
            double neededTrain = x.Length - neededVal;
            double available = x.Length;

            T[] xTrain = new T[(int) (x.Length - neededVal)];
            T[] yTrain = new T[(int) (x.Length - neededVal)];

            T[] xVal = new T[(int) neededVal];
            T[] yVal = new T[(int) neededVal];

            // Iterate through list
            // The probability that an element is selected is needed / available,
            // Guaranteeing that the required number of elements are selected in one pass through
            for (int i = 0; i < x.Length; i++)
            {
                // If the element is randomly selected
                if (Rand.NextDouble() < neededVal / available)
                {
                    xVal[^(int) neededVal] = x[i];
                    yVal[^(int) neededVal] = y[i];
                    neededVal--;
                }
                else
                {
                    xTrain[^(int) neededTrain] = x[i];
                    yTrain[^(int) neededTrain] = y[i];
                    neededTrain--;
                }

                available--;
            }

            return (xTrain, yTrain, xVal, yVal);
        }
    }
}