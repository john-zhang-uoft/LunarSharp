using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using static NeuralSharp.Sampling;

namespace NeuralSharp
{
    public partial class Model
    {
        /// <summary>
        /// Fit and validated the model on the dataset.
        /// </summary>
        /// <param name="x">Train features the model learns on.</param>
        /// <param name="y">Train labels the model learns on.</param>
        /// <param name="batchSize">Number of samples per mini-batch.</param>
        /// <param name="epochs">Number of epochs the model will be trained for.</param>
        /// <param name="verbose">Verbosity mode, i.e. the amount of updates printed during training.</param>
        /// <param name="alpha">Learning rate for weights.</param>
        /// <param name="gamma">Learning rate for biases.</param>
        /// <param name="callbacks">List of callbacks that are checked during training.</param>
        /// <param name="validationFrac">Fraction of the train data used for validation between 0 and 1.</param>
        /// <param name="validationSet">Validation data not trained on but evaluated for loss and metrics each epoch.</param>
        /// <param name="shuffle">Whether to randomly shuffle training data before each epoch.</param>
        /// <param name="classWeights">Array containing a weight for each class that weights the loss function.</param>
        /// <param name="datasetWeights">Array containing a weight for each training datapoint that weights the loss function.</param>
        public void Fit(Matrix[] x, Matrix[] y, int batchSize = 1, int epochs = 1, int verbose = 1,
            float alpha = 0.001f, float gamma = 0.001f, Callback[] callbacks = null, float validationFrac = 0,
            Matrix[] validationSet = null, bool shuffle = false, float[] classWeights = null,
            float[] datasetWeights = null)
        {
            // Check whether x and y are the same length
            if (x.Length != y.Length)
            {
                throw new InvalidDataException("X and Y are not the same shape.");
            }

            Console.WriteLine($"Starting training process with verbosity: {verbose}.");

            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                string message = $"Epoch: {e + 1}\t";

                if (shuffle)
                {
                    x.Shuffle();
                    y.Shuffle();
                }

                List<Matrix> xTrain = new List<Matrix>();
                List<Matrix> yTrain = new List<Matrix>();
                List<Matrix> xVal = new List<Matrix>();
                List<Matrix> yVal = new List<Matrix>();

                if (validationFrac > 0)
                {
                    (xTrain, yTrain, xVal, yVal) = TrainValSplitList(x, y, validationFrac);
                }
                else
                {
                    xTrain = x.ToList();
                    yTrain = y.ToList();
                }

                // For each batch
                for (int i = 0; i < xTrain.Count / batchSize; i++)
                {
                    // Reset gradients in each layer
                    foreach (Layer l in _layers)
                    {
                        l.ResetGradients();
                    }

                    // For each datapoint inside that batch
                    for (int j = 0; j < batchSize; j++)
                    {
                        ForwardPass(xTrain[i * batchSize + j]);
                        BackwardPass(xTrain[i * batchSize + j], yTrain[i * batchSize + j], alpha, gamma);
                    }

                    // Update gradient based on the mean gradient
                    foreach (Layer l in _layers)
                    {
                        l.UpdateParameters(batchSize, alpha, gamma);
                    }
                }

                // For each remaining datapoint
                if (xTrain.Count / batchSize * batchSize != xTrain.Count)
                {
                    // Reset gradients in each layer
                    foreach (Layer l in _layers)
                    {
                        l.ResetGradients();
                    }
                    
                    for (int i = (xTrain.Count / batchSize) * batchSize; i < xTrain.Count; i++)
                    {
                        ForwardPass(xTrain[i]);
                        BackwardPass(xTrain[i], yTrain[i], alpha, gamma);
                    }

                    // Update gradient based on the mean gradient
                    foreach (Layer l in _layers)
                    {
                        l.UpdateParameters(xTrain.Count - (xTrain.Count / batchSize * batchSize), alpha, gamma);
                    }
                }

                // Validate data
                float trainLoss = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    Predict(x[i]);

                    trainLoss += _lossFunction(_layers[^1].Neurons, y[i]);
                }

                trainLoss /= x.Length;

                message += $"Train loss = {trainLoss}";
                
                // Validation data
                if (validationFrac != 0)
                {
                    float valLoss = 0;

                    for (int i = 0; i < xVal.Count; i++)
                    {
                        Predict(xVal[i]);
                        valLoss += _lossFunction(_layers[^1].Neurons, yVal[i]);
                    }

                    valLoss /= xVal.Count;

                    message += $"Validation loss = {valLoss}";
                }
                
                Console.WriteLine(message);
            }
        }

        public void Fit2(Matrix[] x, Matrix[] y, int batchSize = 1, int epochs = 1, int verbose = 1,
            float alpha = 0.001f, float gamma = 0.001f, Callback[] callbacks = null, float validationFrac = 0,
            bool shuffle = false, float[] classWeights = null, float[] datasetWeights = null)
        {
            // Check whether x and y are the same length
            if (x.Length != y.Length)
            {
                throw new InvalidDataException("X and Y are not the same shape.");
            }

            if (validationFrac < 0)
            {
                throw new InvalidModelArgumentException("Validation fraction cannot be less than zero");
            }

            Console.WriteLine($"Starting training process with verbosity: {verbose}.");

            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                string message = $"Epoch: {e + 1}\t";

                if (shuffle)
                {
                    x.Shuffle();
                    y.Shuffle();
                }

                Matrix[] xTrain = new Matrix[] { };
                Matrix[] yTrain = new Matrix[] { };
                Matrix[] xVal = new Matrix[] { };
                Matrix[] yVal = new Matrix[] { };

                if (validationFrac > 0)
                {
                    (xTrain, yTrain, xVal, yVal) = TrainValSplit(x, y, validationFrac);
                }
                else
                {
                    xTrain = x;
                    yTrain = y;
                }

                // For each batch
                for (int i = 0; i < xTrain.Length / batchSize * batchSize; i += batchSize)
                {
                    TrainBatch(xTrain[i..(i + batchSize)],
                        yTrain[i..(i + batchSize)], alpha, gamma);
                }

                if (xTrain.Length / batchSize * batchSize != xTrain.Length)
                {
                    TrainBatch(xTrain[(xTrain.Length / batchSize * batchSize).. xTrain.Length],
                        yTrain[(xTrain.Length / batchSize * batchSize).. xTrain.Length], alpha, gamma);
                }
                
                // Validate train data
                float trainLoss = 0;
                for (int i = 0; i < xTrain.Length; i++)
                {
                    Predict(xTrain[i]);
                    trainLoss += _lossFunction(_layers[^1].Neurons, yTrain[i]);
                }

                trainLoss /= xTrain.Length;

                message += $"Train loss = {trainLoss}";

                // Validation data
                if (validationFrac != 0)
                {
                    float valLoss = 0;

                    for (int i = 0; i < xVal.Length; i++)
                    {
                        Predict(xVal[i]);
                        valLoss += _lossFunction(_layers[^1].Neurons, yVal[i]);
                    }

                    valLoss /= xVal.Length;

                    message += $"\tValidation loss = {valLoss}";
                }

                Console.WriteLine(message);
            }
        }
    }
}