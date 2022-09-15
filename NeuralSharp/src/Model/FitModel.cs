using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using static NeuralSharp.Sampling;

namespace NeuralSharp
{
    public partial class Model
    {
        public void Fit(Matrix[] x, Matrix[] y, int batchSize = 1, int epochs = 1, int verbose = 1,
            Callback[] callbacks = null, float validationFrac = 0, bool shuffle = false, float[] classWeights = null,
            float[] datasetWeights = null)
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

            // Initialize train log
            if (validationFrac > 0)
            {
                var trackingMetrics = new List<string> {"Training Loss", "Validation Loss"};
                foreach (Metric m in ModelMetrics)
                {
                    trackingMetrics.Add(m.ToString());
                }
                Log = new TrainLog(epochs, trackingMetrics);
            }
            else
            {
                Log = new TrainLog(epochs, new List<string> { "Training Loss" });
            }

            Matrix[] xTrain = Array.Empty<Matrix>();
            Matrix[] yTrain = Array.Empty<Matrix>();
            Matrix[] xVal = Array.Empty<Matrix>();
            Matrix[] yVal = Array.Empty<Matrix>();

            // Split into validation and training sets
            if (validationFrac > 0)
            {
                (xTrain, yTrain, xVal, yVal) = TrainValSplit(x, y, validationFrac);
            }
            else
            {
                // Make copies of x and y data to prevent the original data passed in from being rearranged

                xTrain = new Matrix[x.Length];
                yTrain = new Matrix[y.Length];

                Array.Copy(x, xTrain, x.Length);
                Array.Copy(y, yTrain, y.Length);
            }

            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                string message = $"Epoch: {e + 1}\t";

                if (shuffle)
                {
                    Shuffle(xTrain, yTrain);
                }

                // Validate train data
                float trainLoss = 0;

                // For each batch
                for (int i = 0; i < xTrain.Length / batchSize * batchSize; i += batchSize)
                {
                    trainLoss += TrainBatchCalcLoss(xTrain[i..(i + batchSize)],
                        yTrain[i..(i + batchSize)]) * batchSize;
                }

                if (xTrain.Length / batchSize * batchSize != xTrain.Length)
                {
                    trainLoss += TrainBatchCalcLoss(xTrain[(xTrain.Length / batchSize * batchSize).. xTrain.Length],
                                     yTrain[(xTrain.Length / batchSize * batchSize).. xTrain.Length])
                                 * (xTrain.Length - xTrain.Length / batchSize * batchSize);
                }

                trainLoss /= xTrain.Length;

                message += $"Train loss = {trainLoss}";

                
                // Validation

                if (validationFrac > 0)
                {
                    float valLoss = 0;
                    
                    for (int i = 0; i < xVal.Length; i++)
                    {
                        valLoss += LossFunction(Predict(xVal[i]), yVal[i]);
                    }
                    valLoss /= xVal.Length;
                    message += $"\tValidation loss = {valLoss}";
                    Log.LogEpoch(new float[] { trainLoss, valLoss });
                }

                Log.LogEpoch(new float[] { trainLoss });

                Console.WriteLine(message);
            }
        }

        /// <summary>
        /// Train model on a mini-batch of features and labels.
        /// </summary>
        /// <param name="xBatch"></param>
        /// <param name="yBatch"></param>
        /// <exception cref="InvalidDataException"></exception>
        public void TrainBatch(Matrix[] xBatch, Matrix[] yBatch)
        {
            if (xBatch.Length != yBatch.Length)
            {
                throw new InvalidDataException("X and Y batch are not the same size.");
            }

            if (xBatch.Length == 0 || yBatch.Length == 0)
            {
                throw new InvalidDataException("X and Y batch cannot be empty for training.");
            }

            // Reset gradients in each layer
            foreach (Layer l in Layers)
            {
                l.ResetGradients();
            }

            // For each datapoint inside that batch
            for (int j = 0; j < xBatch.Length; j++)
            {
                ForwardPass(xBatch[j]);
                BackwardPass(DerivativeLossFunction(Layers[^1].Neurons, yBatch[j]), xBatch[j]);
            }

            // Update gradient based on the mean gradient
            Optimizer.UpdateParameters(xBatch.Length);
        }

        /// <summary>
        /// Train model on a mini-batch of features and labels and output loss.
        /// </summary>
        /// <param name="xBatch"></param>
        /// <param name="yBatch"></param>
        /// <exception cref="InvalidDataException"></exception>
        public float TrainBatchCalcLoss(Matrix[] xBatch, Matrix[] yBatch)
        {
            float loss = 0;

            if (xBatch.Length != yBatch.Length)
            {
                throw new InvalidDataException("X and Y batch are not the same size.");
            }

            if (xBatch.Length == 0 || yBatch.Length == 0)
            {
                throw new InvalidDataException("X and Y batch cannot be empty for training.");
            }

            // Reset gradients in each layer
            foreach (Layer l in Layers)
            {
                l.ResetGradients();
            }

            // For each datapoint inside that batch
            for (int j = 0; j < xBatch.Length; j++)
            {
                ForwardPass(xBatch[j]);

                loss += LossFunction(Layers[^1].Neurons, yBatch[j]);

                BackwardPass(DerivativeLossFunction(Layers[^1].Neurons, yBatch[j]), xBatch[j]);
            }

            // Update gradient based on the mean gradient
            Optimizer.UpdateParameters(xBatch.Length);

            return loss / xBatch.Length;
        }
    }
}