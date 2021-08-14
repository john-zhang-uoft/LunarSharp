using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;

namespace NeuralSharp
{
    public partial class Model
    {
        public List<Layer> Layers { get; private set; }
        private Func<Matrix, Matrix, float> _lossFunction;
        private Func<Matrix, Matrix, Matrix> _derivativeLossFunction;
        private Metric[] _metrics;
        private TrainLog _log;
        private AbstractOptimizer _optimizer;
        
        public Model(params Layer[] layers)
        {
            if (layers.Any(layer => layer == null))
            {
                throw new InvalidDataException("Layer cannot be null.");
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
                throw new InvalidDataException("Layer cannot be null.");
            }

            Layers.Add(layer);
        }

        /// <summary>
        /// Feed an input through all the layers of the network as the forward step during backpropagation.
        /// </summary>
        /// <param name="input"></param>
        private void ForwardPass(Matrix input)
        {
            Layers[0].FeedForward(input);

            // Feedforward result through each other layer
            for (int l = 1; l < Layers.Count; l++)
            {
                Layers[l].FeedForward(Layers[l - 1].Neurons);
            }
        }

        /// <summary>
        /// Feed an input through all the layers of the network.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="target"></param>
        private void BackwardPass(Matrix input, Matrix target)
        {
            // Backpropagation algorithm to calculate gradient with respect to neurons
            // then with respect to weights and biases and adjust parameters

            Layers[^1].BackPropagate(null, Layers[^2].Neurons, target, _derivativeLossFunction);

            for (int l = Layers.Count - 2; l >= 1; l--)
            {
                Layers[l].BackPropagate(Layers[l + 1], Layers[l - 1].Neurons, target, _derivativeLossFunction);
            }

            Layers[0].BackPropagate(Layers[1], input, target, _derivativeLossFunction);
        }
        
        public void Save(string filePath)
        {
        }
        
        /// <summary>
        /// Uses the full capabilities of the model to make a prediction given an input.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix Predict(Matrix input)
        {
            Matrix previousNeurons = input;
            
            // Feedforward result through each layer that is not a dropout layer
            foreach (Layer t in Layers.Where(t => t is not Dropout))
            {
                t.FeedForward(previousNeurons);
                previousNeurons = t.Neurons;
            }
            
            return Layers[^1].Neurons;
        }

        public float Evaluate(Matrix[] inputs, Matrix[] expectedOutputs, Metric[] metrics)
        {
            float accuracy = 0;
            string message = "";
            
            for (int i = 0; i < inputs.Length; i++)
            {
                if (Encoder<Matrix>.ProbabilitiesToOneHot(Predict(inputs[i])) == expectedOutputs[i])
                {
                    accuracy += 1;
                }
            }

            accuracy /= inputs.Length;
            message += $"Accuracy: {accuracy}";
            Console.WriteLine(message);
            return accuracy;
        }
    }
}