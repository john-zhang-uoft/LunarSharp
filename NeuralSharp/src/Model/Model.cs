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
        private Func<Matrix, Matrix, float> _lossFunction;
        private Func<Matrix, Matrix, Matrix> _derivativeLossFunction;
        private Metric[] _metrics;
        private TrainLog _log;
            
        public Model(params Layer[] layers)
        {
            if (layers.Any(layer => layer == null))
            {
                throw new InvalidDataException("Layer cannot be null.");
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
                throw new InvalidDataException("Layer cannot be null.");
            }

            _layers.Add(layer);
        }

        /// <summary>
        /// Feed an input through all the layers of the network.
        /// </summary>
        /// <param name="input"></param>
        private void ForwardPass(Matrix input)
        {
            _layers[0].FeedForward(input);

            // Feedforward result through each other layer
            for (int l = 1; l < _layers.Count; l++)
            {
                _layers[l].FeedForward(_layers[l - 1].Neurons);
            }
        }

        /// <summary>
        /// Feed an input through all the layers of the network.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"></param>
        /// <param name="alpha"></param>
        /// <param name="gamma"></param>
        private void BackwardPass(Matrix input, Matrix target, float alpha, float gamma)
        {
            // Backpropagation algorithm to calculate gradient with respect to neurons
            // then with respect to weights and biases and adjust parameters

            _layers[^1].BackPropagate(null, _layers[^2].Neurons, target, _derivativeLossFunction);

            for (int l = _layers.Count - 2; l >= 1; l--)
            {
                _layers[l].BackPropagate(_layers[l + 1], _layers[l - 1].Neurons, target, _derivativeLossFunction);
            }

            _layers[0].BackPropagate(_layers[1], input, target, _derivativeLossFunction);
        }
        
        public void Save(string filePath)
        {
        }

        public Matrix Predict(Matrix input)
        {
            ForwardPass(input);

            return _layers[^1].Neurons;
        }
    }
}