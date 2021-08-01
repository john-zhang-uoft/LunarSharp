using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralSharp
{
    public class DataLoader
    {
        private readonly StreamReader _streamReader;

        private DataLoader(string filePath)
        {
            _streamReader = new StreamReader(filePath);
        }

        public static Matrix[] ReadCsv(string filePath, string separation, int numHeaderRows = 0, int skipFooter = 0)                        
        {
            DataLoader dataLoader = new DataLoader(filePath);

            List<Matrix> res = new List<Matrix>();

            using (dataLoader._streamReader)
            {
                int numLine = 0;
                
                for (int i = 0; i < numHeaderRows; i++)
                {
                    dataLoader._streamReader.ReadLine();
                    numLine++;
                }
                
                while (!dataLoader._streamReader.EndOfStream)
                {
                    numLine++;
                    string line = dataLoader._streamReader.ReadLine();
                    if (line == null)
                    {
                        throw new InvalidDataException($"Line {numLine} is null and cannot be read");
                    }
                    string[] values = line.Split(separation);
                    
                    res.Add(new Matrix(values.Select(float.Parse), (values.Length, 1)));
                    
                }
            }

            return res.ToArray();
        }
        
    }
}