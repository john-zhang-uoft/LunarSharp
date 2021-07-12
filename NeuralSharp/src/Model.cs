using System;
using System.Collections.Generic;
using System.Net.Sockets;

namespace NeuralSharp
{
    public class Model
    {
        public List<Layer> Layers;
        
        public Model(params Layer[] layers)
        {
            Layers = new List<Layer>(layers);
        }

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        public Matrix Predict(Matrix input)
        {
            throw new NotImplementedException();
        }

        public Matrix Compile(Matrix input)
        {
            throw new NotImplementedException();
        }
        
    }
}