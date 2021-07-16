using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;

namespace NeuralSharp
{
    public partial class Model
    {
        private List<Layer> _layers;
        private Loss _lossFunction;

        public Model(params Layer[] layers)
        {
            if (layers.Any(layer => layer == null))
            {
                throw new InvalidDataException("Layer cannot be null");
            }

            _layers = new List<Layer>(layers);
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

            _layers.Add(layer);
        }
        
        public void Save(string filePath)
        {
        }

        public Matrix Predict(Matrix input)
        {
            _layers[0].FeedForward(input);

            // Feedforward result through each other layer
            for (int l = 1; l < _layers.Count; l++)
            {
                _layers[l].FeedForward(_layers[l - 1].Neurons);
            }

            return _layers[^1].Neurons;        }
    }
}