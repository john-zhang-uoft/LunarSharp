﻿using NeuralSharp;
using System.Diagnostics;

namespace NeuralSharp.Tests
{
    public class MatrixTests
    {
        int RunTests(string[] args)
        {
            Matrix a = new Matrix(5, 10);
            Matrix b = new Matrix(5, 10);

            Debug.Assert(a + b == b);
            
            return 0;
        }
    }
}