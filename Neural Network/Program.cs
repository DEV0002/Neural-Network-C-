using System;
using System.Collections.Generic;
using System.Text;

namespace Neural_Network {
    class Program {
        static void Main(string[] args) {
            NN n = new NN(new int[] { 4, 2, 1 });
            n.Test(new double[] { 0, 1, 1, 0 });
            Console.ReadKey();
        }
    }
}
