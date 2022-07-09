﻿using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.InteropServices;
using static NeuralSharp.Sampling;

namespace NeuralSharp.Tests
{
    public static class GenerativeAdversarial
    {
        public static void Example()
        {
            // Load Mnist dataset
            const string trainPath = @"C:\Users\johnz\RiderProjects\NeuralSharp2\NeuralSharp2\mnist_train.csv";
            var data = DataLoader.ReadCsv(trainPath, ",", numHeaderRows: 1);

            // Get images
            var (y, x) = Matrix.ExtractCol(data, 0);
            // Turn features into proper format
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = x[i].Transpose();
            }
            
            // Create discriminator model
            Model discriminator = new Model(
                new Dense(inputShape: 784, shape: 64, ActivationFunctions.ReLU),
                new Dense(shape: 16, ActivationFunctions.ReLU),
                new Dense(shape: 16, ActivationFunctions.ReLU),
                new Dense(shape: 1, ActivationFunctions.Sigmoid)
            );
            // Create dense generator model
            Model generator = new Model(
                new Dense(inputShape: 100, shape: 144, ActivationFunctions.ReLU),
                new Dense(shape: 256, ActivationFunctions.ReLU),
                new Dense(shape: 784, ActivationFunctions.Sigmoid),
                discriminator.Layers[0],
                discriminator.Layers[1],
                discriminator.Layers[2],
                discriminator.Layers[3]
            );
            var generatorLayers = new List<int>() { 0, 1, 2 };
            
            static float RealDiscriminatorLoss(Matrix realOutput)
            {
                return Loss.BinaryCrossEntropy(realOutput, Matrix.OnesLike(realOutput));
            }
            static Matrix RealDiscriminatorLossDerivative(Matrix realOutput)
            {
                return Loss.DBinaryCrossEntropy(realOutput, Matrix.OnesLike(realOutput));
            }
            
            static float FakeDiscriminatorLoss(Matrix fakeOutput)
            {
                return Loss.BinaryCrossEntropy(fakeOutput, Matrix.ZerosLike(fakeOutput));
            }
            static Matrix FakeDiscriminatorLossDerivative(Matrix fakeOutput)
            {
                return Loss.DBinaryCrossEntropy(fakeOutput, Matrix.ZerosLike(fakeOutput));
            }
            
            
            static float GeneratorLoss(Matrix fakeOutput)
            {
                return Loss.BinaryCrossEntropy(fakeOutput, Matrix.OnesLike(fakeOutput));
            }
            static Matrix GeneratorLossDerivative(Matrix fakeOutput)
            {
                return Loss.DBinaryCrossEntropy(fakeOutput, Matrix.OnesLike(fakeOutput));
            }

            discriminator.ConnectLayers();
            discriminator.InitializeParametersXavier();
            discriminator.Optimizer = new StochasticGD(alpha: 0.01f, gamma: 0.01f, momentum:0.5f, nesterov:true);
            discriminator.Optimizer.ConnectToModel(discriminator);
            discriminator.Optimizer.Initialize();
            
            generator.ConnectLayers();
            generator.InitializeParametersXavier();
            generator.Optimizer = new StochasticGD(alpha: 0.01f, gamma: 0.01f, momentum:0.5f, nesterov:true);
            generator.Optimizer.ConnectToModel(generator);
            generator.Optimizer.Initialize();
            
            void TrainStep(Matrix[] images, out float discLoss, out float genLoss)
            {
                discLoss = 0;
                genLoss = 0;
                foreach (Layer l in generator.Layers)
                {
                    l.ResetGradients();
                }
                foreach (Layer l in discriminator.Layers)
                {
                    l.ResetGradients();
                }
                for (int i = 0; i < images.Length; i++)
                {
                    Matrix randomNoise = Matrix.RandomMatrix(1, 100, 1);
                    generator.ForwardPass(randomNoise);
                    Matrix generatedImage = generator.Layers[2].Neurons;

                    Matrix fakeOutput = discriminator.Layers[^1].Neurons;
                    // Only update delta Weights and Biases for generator layers in this backward pass
                    generator.BackwardPassSelective(GeneratorLossDerivative(fakeOutput), randomNoise, generatorLayers);
                    
                    // Now train the discriminator on the generated image
                    discriminator.ForwardPass(generatedImage);
                    discriminator.BackwardPass(FakeDiscriminatorLossDerivative(fakeOutput), generatedImage);

                    // Train the discriminator on a real image
                    Matrix realOutput = discriminator.ForwardPassReturn(images[i]);
                    discriminator.BackwardPass(RealDiscriminatorLossDerivative(realOutput), images[i]);
                    
                    discLoss += FakeDiscriminatorLoss(fakeOutput) + RealDiscriminatorLoss(realOutput);
                    genLoss += GeneratorLoss(fakeOutput);
                    Console.WriteLine($"Discriminator Loss: {discLoss}");
                    Console.WriteLine($"Generator Loss: {genLoss}");
                }
                // I need to detach backpropagation from delta bias and weight in the future
                // Update only generator layer parameters with generator optimizer
                generator.Optimizer.UpdateParameters(images.Length, generatorLayers);
                discriminator.Optimizer.UpdateParameters(images.Length);
                discLoss /= images.Length;
                genLoss /= images.Length;
            }

            const int epochs = 10;
            const int batchSize = 64;

            // For each epoch
            for (int e = 0; e < epochs; e++)
            {
                string message = $"Epoch: {e + 1}\t";

                Shuffle(x, y);
                
                float epochDiscLoss = 0;
                float epochGenLoss = 0;
                // For each batch
                for (int i = 0; i < x.Length / batchSize * batchSize; i += batchSize)
                {
                    TrainStep(x[i..(i + batchSize)], out float discLoss, out float genLoss);
                    epochDiscLoss += discLoss;
                    epochGenLoss += genLoss;
                }

                if (x.Length / batchSize * batchSize != x.Length)
                {
                    TrainStep(x[(x.Length / batchSize * batchSize).. x.Length], out float discLoss, out float genLoss);
                    epochDiscLoss += discLoss;
                    epochGenLoss += genLoss;
                }

                epochDiscLoss /= x.Length;
                epochGenLoss /= x.Length;

                message += $"\n\nEpochDiscLoss loss = {epochDiscLoss}\n";
                message += $"EpochGenLoss loss = {epochGenLoss}\n";

                Console.WriteLine(message);
            }
        }
    }
}