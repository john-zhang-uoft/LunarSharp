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
        public void Compile(AbstractOptimizer optimizer, LossFunctions lossFunction, IEnumerable<Metric> metrics = null)
        {
            if (!Layers[0].IsValidInputShape())
            {
                throw new InvalidDataException("The input shape for the first layer was not provided or is invalid.");
            }

            for (int i = 1; i < Layers.Count; i++)
            {
                if (Layers[i] is not Dropout) continue;
                if (Layers[i - 1] is Dropout)
                {
                    throw new InvalidModelArgumentException(
                        $"Cannot compile model with two dropout layers in a row ({i - 1}, {i}).");
                }
            }
            
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] is Dropout) continue;
                if (!Layers[i].IsValidOutputShape())
                {
                    throw new InvalidDataException($"The shape of layer {i + 1} was not provided or is invalid.");
                }

            }

            ModelMetrics = new HashSet<Metric>(metrics ?? Array.Empty<Metric>()).ToArray();
                
            ConnectLayers();
            
            InitializeParametersXavier();
            
            switch (lossFunction)
            {
                case LossFunctions.MeanSquaredError:
                    LossFunction = Loss.MeanSquaredError;
                    DerivativeLossFunction = Loss.DMeanSquaredError;
                    break;

                case LossFunctions.BinaryCrossEntropy:
                    LossFunction = Loss.BinaryCrossEntropy;
                    DerivativeLossFunction = Loss.DBinaryCrossEntropy;
                    break;
                
                default:
                    throw new InvalidModelArgumentException("Invalid loss function.");
            }

            Optimizer = optimizer;
            Optimizer.ConnectToModel(this);
            Optimizer.Initialize();
        }

        public void Compile(AbstractOptimizer optimizer, LossFunctionDelegate lossFunction,
            IEnumerable<Metric> metrics = null)
        {
            if (!Layers[0].IsValidInputShape())
            {
                throw new InvalidDataException("The input shape for the first layer was not provided or is invalid.");
            }

            for (int i = 1; i < Layers.Count; i++)
            {
                if (Layers[i] is not Dropout) continue;
                if (Layers[i - 1] is Dropout)
                {
                    throw new InvalidModelArgumentException(
                        $"Cannot compile model with two dropout layers in a row ({i - 1}, {i}).");
                }
            }
            
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i] is Dropout) continue;
                if (!Layers[i].IsValidOutputShape())
                {
                    throw new InvalidDataException($"The shape of layer {i + 1} was not provided or is invalid.");
                }
            }
            
            ModelMetrics = new HashSet<Metric>(metrics ?? Array.Empty<Metric>()).ToArray();
                
            ConnectLayers();
            
            InitializeParametersXavier();

            LossFunction = (x,y) => lossFunction(x, y);
            Optimizer = optimizer;
            Optimizer.ConnectToModel(this);
            Optimizer.Initialize();
        }
        
        /// <summary>
        /// Delegate for custom loss function.
        /// </summary>
        public delegate float LossFunctionDelegate(Matrix output, Matrix trueLabel);
        
        /// <summary>
        /// Connects the layers in the model together to construct weight matrices of the right size.
        /// </summary>
        /// <exception cref="InvalidDataException"></exception>
        public void ConnectLayers()
        {
            if (Layers == null || Layers.Count == 0)
            {
                throw new InvalidDataException("The model does not contain any layers");
            }

            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].ConnectDropout(Layers[i - 1]);    
            }
            
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Connect(Layers[i - 1]);
            }
        }
        
    }
}