using System;

namespace NeuralSharp
{
    public class StochasticGD : AbstractOptimizer
    {
        public float Momentum { get; private set; }
        public bool Nesterov { get; private set; }

        private Matrix[] _weightVelocity;
        private Matrix[] _biasVelocity;
        
        public StochasticGD(float alpha = 0.01f, float beta = 0.01f, float momentum = 0.0f, bool nesterov = false)
        {
            Alpha = alpha;
            Beta = beta;
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
        public override void UpdateParameters()
        {
            if (Momentum == 0)
            {
                for (int i = 0; i < _weightVelocity.Length; i++)
                {
                    NNModel.Layers[i].Weights -= Alpha * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Biases -= Beta * NNModel.Layers[i].DeltaBias;
                }

                return;
            }

            if (Nesterov)
            {
                for (int i = 0; i < _weightVelocity.Length; i++)
                {
                    _weightVelocity[i] = Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Weights += Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight;

                    _biasVelocity[i] = Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias;
                    NNModel.Layers[i].Biases += Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias;
                }
            }
            else
            {
                for (int i = 0; i < _weightVelocity.Length; i++)
                {
                    _weightVelocity[i] = Momentum * _weightVelocity[i] - Alpha * NNModel.Layers[i].DeltaWeight;
                    NNModel.Layers[i].Weights += _weightVelocity[i];

                    _biasVelocity[i] = Momentum * _biasVelocity[i] - Beta * NNModel.Layers[i].DeltaBias;
                    NNModel.Layers[i].Biases += _biasVelocity[i];
                }
            }
        }
        //////////////////////// MAKE SURE THAT BATCH SIZE IS INCLUDED IN THIS METHOD OR FIRST DIVIDE DELTA BY BATCH SIZE IN FITMODEL
    }
}