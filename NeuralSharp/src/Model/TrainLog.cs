using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralSharp
{
    public class TrainLog
    {
        public List<float[]> Log { get; private set; }
        public List<string> Metrics { get; private set; }
        
        public TrainLog(int numEpochs, List<string> metrics)
        {
            Log = new List<float[]>(numEpochs);
            Metrics = metrics;
        }

        public void LogEpoch(float[] data)
        {
            if (data.Length != Metrics.Count)
            {
                throw new InvalidDataException("Data points logged cannot exceed the number metrics.");
            }
            
            Log.Add(data);
        }
        
    }
}