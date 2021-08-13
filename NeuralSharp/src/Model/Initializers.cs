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
            if (Layers.Any(layer => !layer.IsValidInputShape() || !layer.IsValidOutputShape()))
            {
                throw new InvalidOperationException("Cannot initialize parameters of layers with invalid shapes.");
            }
            
            for (int i = 0; i < Layers.Count; i++)
            {
                float range = (float) Math.Sqrt(6f / (Layers[i].InputShape.Item1 + Layers[i].OutputShape.Item1));
                Layers[i].InitializeRandomWeights(range);
                Layers[i].InitializeZeroBiases();
            }
        }

        /// <summary>
        /// Initializes weights and biases for layers that are already connected.
        /// </summary>
        public void InitializeParametersHe()
        {
            if (Layers.Any(layer => !layer.IsValidInputShape() || !layer.IsValidOutputShape()))
            {
                throw new InvalidOperationException("Cannot initialize parameters of layers with invalid shapes.");
            }

            for (int i = 0; i < Layers.Count; i++)
            {
                float range = (float) Math.Sqrt(2f / Layers[i].InputShape.Item1);
                Layers[i].InitializeRandomWeights(range);
                Layers[i].InitializeZeroBiases();
            }
        }
    }
}