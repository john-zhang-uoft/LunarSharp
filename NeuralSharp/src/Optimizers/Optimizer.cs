namespace NeuralSharp
{
    public abstract class AbstractOptimizer
    {
        public Model NNModel { get; protected set; }
        public float Alpha { get; protected set; }
        public float Beta { get; protected set; }
        
        public void ConnectToModel(Model model)
        {
            NNModel = model;
        }

        public abstract void Initialize();

        public abstract void UpdateParameters(int batchSize);
        
    }
}