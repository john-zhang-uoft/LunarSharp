using System.Diagnostics;
namespace NeuralSharp
{
    public class Program
    {
        public static int Main()
        {
            Matrix a = new Matrix(5, 10);
            Matrix b = new Matrix(5, 10);

            Debug.Assert(a + b == b);
            
            return 0;
        }
    }
}