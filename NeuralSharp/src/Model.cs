using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;

namespace NeuralSharp
{
    public partial class Model
    {
        public List<Layer> Layers;
        public Loss LossFunction;

        public Model(params Layer[] layers)
        {
            if (layers.Any(layer => layer == null))
            {
                throw new InvalidDataException("Layer cannot be null");
            }

            Layers = new List<Layer>(layers);
        }

        /// <summary>
        /// Adds a layer to the network. The first layer added to the model must have an input size parameter.
        /// </summary>
        /// <param name="layer"></param>
        /// <exception cref="InvalidDataException"></exception>
        public void Add(Layer layer)
        {
            if (layer == null)
            {
                throw new InvalidDataException("Layer cannot be null");
            }

            Layers.Add(layer);
        }

        /// <summary>
        /// Compile model to prepare for training.
        /// </summary>
        /// <param name="optimizer">Optimizer used during training.</param>
        /// <param name="loss">Loss function used during training.</param>
        /// <param name="metrics">List of metrics used to validate the dataset on during training and testing.</param>
        public void Compile(Optimizer optimizer, Loss loss, Metric[] metrics)
        {
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Connect(Layers[i - 1]);
                Layers[i].InitializeRandomWeights(1);
                Layers[i].InitializeRandomBiases(1);
            }

            Layers[0].InitializeRandomWeights(1);
            Layers[0].InitializeRandomBiases(1);

            LossFunction = loss;
        }

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
        /// <param name="validationData">Validation data not trained on but evaluated for loss and metrics each epoch.</param>
        /// <param name="shuffle">Whether to randomly shuffle training data before each epoch.</param>
        /// <param name="classWeights">Array containing a weight for each class that weights the loss function.</param>
        /// <param name="sampleWeights">Array containing a weight for each training datapoint that weights the loss function.</param>
        public void Fit(Matrix[] x, Matrix[] y, int batchSize = 1, int epochs = 1, int verbose = 1,
            float alpha = 0.001f, float gamma = 0.001f, Callback[] callbacks = null, float validationFrac = 0,
            Matrix[] validationData = null, bool shuffle = false, float[] classWeights = null,
            float[] sampleWeights = null)
        {
            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                // For each datapoint
                for (int i = 0; i < x.Length; i++)
                {
                    Layers[0].FeedForward(x[i]);

                    // Feedforward result through each other layer
                    for (int l = 1; l < Layers.Count; l++)
                    {
                        Layers[l].FeedForward(Layers[l - 1].Neurons);
                    }

                    // Back-propagate error

                    Layers[^1].BackPropagate(null, Layers[^2].Neurons, y[i], alpha, gamma);

                    for (int l = Layers.Count - 2; l >= 1; l--)
                    {
                        Layers[l].BackPropagate(Layers[l + 1], Layers[l - 1].Neurons, y[i], alpha, gamma);
                    }
                    
                    Layers[0].BackPropagate(Layers[1], x[i], y[i], alpha, gamma);
                }
                
                float trainLoss = 0;
                switch (LossFunction)
                {
                    case Loss.MeanSquareDError:

                        for (int i = 0; i < x.Length; i++)
                        {
                            Layers[0].FeedForward(x[i]);

                            // Feedforward result through each other layer
                            for (int l = 1; l < Layers.Count; l++)
                            {
                                Layers[l].FeedForward(Layers[l - 1].Neurons);
                            }
                            
                            trainLoss += Output.MeanSquaredError(Layers[^1].Neurons, y[i]);
                        }

                        break;
                    default:
                        throw new NotImplementedException("Unimplemented loss function");
                }
                
                Console.WriteLine($"Train loss = {trainLoss}");
                
            }
            
        }


        public void Save(string filePath)
        {
        }

        public Matrix Predict(Matrix input)
        {
            throw new NotImplementedException();
        }
    }
}