using System;

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
            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                // For each datapoint
                for (int i = 0; i < x.Length; i++)
                {
                    ForwardPass(x[i]);
                    BackwardPass(x[i], y[i], alpha, gamma);
                }
                
                float trainLoss = 0;
                switch (_lossFunction)
                {
                    case Loss.MeanSquareDError:

                        for (int i = 0; i < x.Length; i++)
                        {
                            Predict(x[i]);
                            
                            trainLoss += Output.MeanSquaredError(_layers[^1].Neurons, y[i]);
                        }
                        break;
                    
                    default:
                        throw new NotImplementedException("Unimplemented loss function");
                }
                Console.WriteLine($"Train loss = {trainLoss}");
            }
        }

    }
}