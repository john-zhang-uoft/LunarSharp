using System;
using System.Linq;

namespace NeuralSharp
{
    public partial class Model
    {
        /// <summary>
        /// Initializes weights and biases for layers that are already connected.
        /// </summary>
        public void InitializeParametersXavier()
        {
            if (_layers.Any(layer => !layer.IsValidInputShape() || !layer.IsValidOutputShape()))
            {
                throw new InvalidOperationException("Cannot initialize parameters of layers with invalid shapes.");
            }
            
            for (int i = 0; i < _layers.Count; i++)
            {
                float range = (float) Math.Sqrt(6f / (_layers[i].InputShape.Item1 + _layers[i].OutputShape.Item1));
                _layers[i].InitializeRandomWeights(range);
                _layers[i].InitializeZeroBiases();
            }
        }

        /// <summary>
        /// Initializes weights and biases for layers that are already connected.
        /// </summary>
        public void InitializeParametersHe()
        {
            if (_layers.Any(layer => !layer.IsValidInputShape() || !layer.IsValidOutputShape()))
            {
                throw new InvalidOperationException("Cannot initialize parameters of layers with invalid shapes.");
            }

            for (int i = 0; i < _layers.Count; i++)
            {
                float range = (float) Math.Sqrt(2f / _layers[i].InputShape.Item1);
                _layers[i].InitializeRandomWeights(range);
                _layers[i].InitializeZeroBiases();
            }
        }
    }
}