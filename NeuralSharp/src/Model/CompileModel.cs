using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public partial class Model
    {
        /// <summary>
        /// Compile model to prepare for training.
        /// </summary>
        /// <param name="optimizer">Optimizer used during training.</param>
        /// <param name="lossFunction">Loss function used during training.</param>
        /// <param name="metrics">List of metrics used to validate the dataset on during training and testing.</param>
        public void Compile(Optimizer optimizer, LossFunctions lossFunction, IEnumerable<Metric> metrics)
        {
            if (!_layers[0].IsValidInputShape())
            {
                throw new InvalidDataException("The input shape for the first layer was not provided or is invalid.");
            }
            
            for (int i = 0; i < _layers.Count; i++)
            {
                if (!_layers[i].IsValidOutputShape())
                {
                    throw new InvalidDataException($"The shape of layer {i + 1} was not provided or is invalid.");
                }
            }

            _metrics = new HashSet<Metric>(metrics).ToArray();
            
            for (int i = _layers.Count - 1; i > 0; i--)
            {
                _layers[i].Connect(_layers[i - 1]);
                _layers[i].InitializeRandomWeights(1);
                _layers[i].InitializeRandomBiases(1);
            }
            
            _layers[0].InitializeRandomWeights(1);
            _layers[0].InitializeRandomBiases(1);

            switch (lossFunction)
            {
                case LossFunctions.MeanSquaredError:
                    _lossFunction = Loss.MeanSquaredError;
                    _dLossFunction = Loss.DMeanSquaredError;
                    break;

                case LossFunctions.BinaryCrossEntropy:
                    _lossFunction = Loss.BinaryCrossEntropy;
                    _dLossFunction = Loss.DBinaryCrossEntropy;
                    break;
                
                default:
                    throw new InvalidModelArgumentException("Invalid loss function.");
            }
            
        }
        
    }
}