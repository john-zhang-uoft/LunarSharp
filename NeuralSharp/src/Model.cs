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
        
        public Model(params Layer[] layers)
        {
            Layers = new List<Layer>(layers);

            for (int i = Layers.Count - 1; i > 0; i--)
            {
                Layers[i].Connect(Layers[i - 1]);
                Layers[i].InitializeRandomWeights(1);
                Layers[i].InitializeRandomBiases(1);
            }
            Layers[0].InitializeRandomWeights(1);
            Layers[0].InitializeRandomBiases(1);
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
            
            // If this is not the first layer of the network, connect the layer to the previous layer
            if (Layers.Count > 1)
            {
                Layers[^1].Connect(Layers[^2]);
            }
        }
        
        public Matrix Compile(Matrix input)
        {
            throw new NotImplementedException();
        }
        
        
        public Matrix Predict(Matrix input)
        {
            throw new NotImplementedException();
        }


        
    }
}