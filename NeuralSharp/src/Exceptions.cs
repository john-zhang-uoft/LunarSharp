using System;

namespace NeuralSharp
{
    
    public class InvalidModelArgumentException : Exception
    {
        public InvalidModelArgumentException()
        {
            
        }
        
        public InvalidModelArgumentException(string message) : base(message)
        {
            
        }
        
        public InvalidModelArgumentException(string message, Exception inner) : base(message, inner)
        {
            
        }
    }
}