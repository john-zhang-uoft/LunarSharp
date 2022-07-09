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
        public Func<Matrix, Matrix, float> LossFunction;
        public Func<Matrix, Matrix, Matrix> DerivativeLossFunction;
        public Metric[] Metrics;
        public TrainLog Log;
        public AbstractOptimizer Optimizer;
        
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
        public void ForwardPass(Matrix input)
        {
            Layers[0].FeedForward(input);

            // Feedforward result through each other layer
            for (int l = 1; l < Layers.Count; l++)
            {
                Layers[l].FeedForward(Layers[l - 1].Neurons);
            }
        }

        /// <summary>
        /// Feed an input through all the layers of the network as the forward step during backpropagation and return output.
        /// </summary>
        /// <param name="input"></param>
        public Matrix ForwardPassReturn(Matrix input)
        {
            ForwardPass(input);
            return Layers[^1].Neurons;
        }
        
        /// <summary>
        /// Backpropagate through all the layers of the network.
        /// </summary>
        /// <param name="derivativeLossFunction"></param>
        /// <param name="input"></param>
        public void BackwardPass(Matrix derivativeLossFunction, Matrix input)
        {
            // Backpropagation algorithm to calculate gradient with respect to neurons
            // then with respect to weights and biases and adjust parameters
            Layers[^1].BackPropagateLastLayer(derivativeLossFunction, Layers[^2].Neurons);

            for (int l = Layers.Count - 2; l >= 1; l--)
            {
                Layers[l].BackPropagateNotLastLayer(Layers[l + 1], Layers[l - 1].Neurons);
            }
            Layers[0].BackPropagateNotLastLayer(Layers[1], input);
        }


        /// <summary>
        /// Backpropagate through all the layers of the network and update weights on some layers.
        /// </summary>
        /// <param name="derivativeLossFunction"></param>
        /// <param name="input"></param>
        /// <param name="layersToChangeParameters"></param>
        public void BackwardPassSelective(Matrix derivativeLossFunction, Matrix input, List<int> layersToChangeParameters)
        {
            // Backpropagation algorithm to calculate gradient with respect to neurons
            // then with respect to weights and biases and adjust parameters
            if (layersToChangeParameters.Contains(Layers.Count - 1))
            {
                Layers[^1].BackPropagateLastLayer(derivativeLossFunction, Layers[^2].Neurons);
            }
            else
            {
                Layers[^1].BackPropagateLastLayerNoUpdatingParameters(derivativeLossFunction, Layers[^2].Neurons);
            }

            for (int l = Layers.Count - 2; l >= 1; l--)
            {
                if (layersToChangeParameters.Contains(l))
                {
                    Layers[l].BackPropagateNotLastLayer(Layers[l + 1], Layers[l - 1].Neurons);
                }
                else
                {
                    Layers[l].BackPropagateNotLastLayerNoUpdatingParameters(Layers[l + 1], Layers[l - 1].Neurons);
                }

            }

            if (layersToChangeParameters.Contains(0))
            {
                Layers[0].BackPropagateNotLastLayer(Layers[1], input);
            }
            else
            {
                Layers[0].BackPropagateNotLastLayerNoUpdatingParameters(Layers[1], input);
            }
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