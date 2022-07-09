using System;

namespace NeuralSharp
{
    public class UnexpectedEnumValueException<T> : Exception
    {
        public UnexpectedEnumValueException(T value)
            : base("Value " + value + " of enum " + typeof(T).Name + " is not supported")
        {
        }
    }
    
    public class InvalidModelParameter<T> : Exception
    {
        public InvalidModelParameter(T value)
            : base(value + " is an invalid parameter for " + nameof(value))
        {
        }
    }
}