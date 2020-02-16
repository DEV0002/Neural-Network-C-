using System;

namespace Neural_Network {
    class NN {
        NDArray[] Weights;
        NDArray[] Biases;
        NDArray[] Layers;
        public NN(int[] layer_sizes) {
            Weights = new NDArray[layer_sizes.Length-1];
            Biases = new NDArray[Weights.Length];
            Layers = new NDArray[layer_sizes.Length];
            for(int i = 1; i < layer_sizes.Length; i++) {
                Weights[i - 1] = NDArray.CreateRandom(new int[] { layer_sizes[i - 1], layer_sizes[i] });
                Biases[i-1] = NDArray.CreateColumnRandom(layer_sizes[i]);
            }
        }
        public NDArray Test(double[] a) {
            Layers[0] = NDArray.Create(new int[] { a.Length, 1 }, a);
            for(int i = 0; i < Biases.Length; i++)
                Layers[i+1] = NDArray.Sigmoid(Weights[i].Mul(Layers[i]).Add(Biases[i]));
            return Layers[Layers.Length - 1];
        }
    }
}
