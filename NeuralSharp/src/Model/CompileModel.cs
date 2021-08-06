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

            for (int i = 1; i < _layers.Count; i++)
            {
                if (_layers[i] is Dropout)
                {
                    if (_layers[i - 1] is Dropout)
                    {
                        throw new InvalidModelArgumentException(
                            $"Cannot compile model with two dropout layers in a row ({i - 1}, {i}).");
                    }
                }
            }
            
            for (int i = 0; i < _layers.Count; i++)
            {
                if (_layers[i] is not Dropout)
                {
                    if (!_layers[i].IsValidOutputShape())
                    {
                        throw new InvalidDataException($"The shape of layer {i + 1} was not provided or is invalid.");
                    }  
                }

            }

            _metrics = new HashSet<Metric>(metrics).ToArray();
                
            ConnectLayers();
            
            InitializeParametersXavier();
            
            switch (lossFunction)
            {
                case LossFunctions.MeanSquaredError:
                    _lossFunction = Loss.MeanSquaredError;
                    _derivativeLossFunction = Loss.DMeanSquaredError;
                    break;

                case LossFunctions.BinaryCrossEntropy:
                    _lossFunction = Loss.BinaryCrossEntropy;
                    _derivativeLossFunction = Loss.DBinaryCrossEntropy;
                    break;
                
                default:
                    throw new InvalidModelArgumentException("Invalid loss function.");
            }
            
        }

        /// <summary>
        /// Connects the layers in the model together to construct weight matrices of the right size.
        /// </summary>
        /// <exception cref="InvalidDataException"></exception>
        public void ConnectLayers()
        {
            if (_layers == null || _layers.Count == 0)
            {
                throw new InvalidDataException("The model does not contain any layers");
            }

            for (int i = _layers.Count - 1; i > 0; i--)
            {
                _layers[i].ConnectDropout(_layers[i - 1]);    
            }
            
            for (int i = _layers.Count - 1; i > 0; i--)
            {
                _layers[i].Connect(_layers[i - 1]);
            }
        }
        
    }
}