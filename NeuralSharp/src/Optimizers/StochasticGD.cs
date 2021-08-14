using System;

namespace NeuralSharp
{
    public class StochasticGD : AbstractOptimizer
    {
        public float Momentum { get; private set; }
        public bool Nesterov { get; private set; }

        private Matrix[] _weightVelocity;
        private Matrix[] _biasVelocity;
        
        public StochasticGD(float alpha = 0.01f, float gamma = 0.01f, float momentum = 0.0f, bool nesterov = false)
        {
            Alpha = alpha;
            Beta = gamma;
            Momentum = momentum;
            Nesterov = nesterov;
        }

        /// <summary>
        /// Initializes velocity matrices for momentum gradient descent.
        /// </summary>
        public override void Initialize()
        {
            if (NNModel == null)
            {
                throw new InvalidOperationException(
                    "Cannot initialize StochasticGD optimizer without first connecting to a model.");
            }
            
            if (Momentum == 0)
            {
                return;
            }

            _weightVelocity = new Matrix[NNModel.Layers.Count];
            _biasVelocity = new Matrix[NNModel.Layers.Count];

            for (int i = 0; i < NNModel.Layers.Count; i++)
            {
                _weightVelocity[i] = new Matrix(NNModel.Layers[i].Weights.Shape.rows,
                    NNModel.Layers[i].Weights.Shape.cols);

                _biasVelocity[i] = new Matrix(NNModel.Layers[i].Biases.Shape.rows,
                    NNModel.Layers[i].Biases.Shape.cols);
            }
        }

        /// <summary>
        /// Updates model parameters.
        /// </summary>
        public override void UpdateParameters(int batchSize)
        {
            if (Momentum == 0)
            {
                for (int i = 0; i < NNModel.Layers.Count; i++)
                {
                    if (NNModel.Layers[i] is not Dense)
                    {
                        continue;
                    }
                    NNModel.Layers[i].Weights -= Alpha / batchSize * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Biases -= Beta / batchSize * NNModel.Layers[i].DeltaBias;
                }

                return;
            }

            if (Nesterov)
            {
                for (int i = 0; i < _weightVelocity.Length; i++)
                {
                    if (NNModel.Layers[i] is not Dense)
                    {
                        continue;
                    }
                    _weightVelocity[i] = Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Weights += 1 / batchSize * (Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight);

                    _biasVelocity[i] = Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias;
                    NNModel.Layers[i].Biases += 1 / batchSize * (Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias);
                }
            }
            else
            {
                for (int i = 0; i < _weightVelocity.Length; i++)
                {
                    if (NNModel.Layers[i] is not Dense)
                    {
                        continue;
                    }
                    _weightVelocity[i] = Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Weights += 1 / batchSize * _weightVelocity[i];

                    _biasVelocity[i] = Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias;
                    NNModel.Layers[i].Biases += 1 / batchSize * _biasVelocity[i];
                }
            }
        }
    }
}