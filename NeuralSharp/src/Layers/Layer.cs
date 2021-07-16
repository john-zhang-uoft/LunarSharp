using System;

namespace NeuralSharp
{
    /// <summary>
    /// Abstract layer class used as a template for every type of layer.
    /// </summary>
    public abstract class Layer
    {
        public Matrix Neurons { get; protected set; }
        protected (int, int, int) InputShape;
        protected (int, int, int) OutputShape;
        
        /// <summary>
        /// Has i rows and j columns, where i is the number of neurons in this layer
        /// and j is the number of neurons in the previous layer.
        /// The [i, j]-th element is the weight between the j-th input neuron to the i-th neuron of this layer.
        /// </summary>
        public Matrix Weights { get; protected set; }
        
        /// <summary>
        /// Has i rows, where i is the number of neurons in this layer.
        /// </summary>
        public Matrix Biases { get; protected set; }
        
        /// <summary>
        /// Applied to each neuron in the layer to finish calculating the brightness of each neuron.
        /// </summary>
        public ActivationFunctions ActivationFunction { get; protected set; }

        /// <summary>
        /// Stores the gradient of the cost function with respect to each neuron for backpropagation
        /// </summary>
        public Matrix Gradient { get; protected set; }

        
        #region Constructors

        protected Layer() {}    // Empty constructor for inheritance
        protected Layer((int, int, int) outputShape)
        {
            OutputShape = outputShape;
        }

        #endregion


        public abstract void FeedForward(Matrix inputs);
        public abstract void BackPropagate(Layer nextLayer, Matrix previousLayerNeurons, Matrix target, float alpha, float gamma);
        public void Connect(Layer previousLayer)
        {
            InputShape = previousLayer.OutputShape;
        }

        public void InitializeRandomWeights(float range)
        {
            Weights = Matrix.RandomMatrix(range, OutputShape.Item1, InputShape.Item1);
        }

        public void InitializeRandomBiases(float range)
        {
            Biases = Matrix.RandomMatrix(range, OutputShape.Item1, 1);
        }

        /// <summary>
        /// Returns true if there are no negative values and there is at least one nonzero value in the InputShape.
        /// </summary>
        /// <returns></returns>
        public bool IsValidInputShape()
        {
            return (InputShape.Item1 >= 0 && InputShape.Item2 >= 0 && InputShape.Item3 >= 0)
                    && (InputShape.Item1 > 0 || InputShape.Item2 > 0 || InputShape.Item3 > 0);
        }
        
        /// <summary>
        /// Returns true if there are no negative values and there is at least one nonzero value in the OutputShape.
        /// </summary>
        /// <returns></returns>
        public bool IsValidShape()
        {
            return (OutputShape.Item1 >= 0 && OutputShape.Item2 >= 0 && OutputShape.Item3 >= 0)
                   && (OutputShape.Item1 > 0 || OutputShape.Item2 > 0 || OutputShape.Item3 > 0);        
        }
        
    }
}