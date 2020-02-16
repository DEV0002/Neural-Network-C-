using System;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace Neural_Network {
    public class NDArray {
        public double[] arr;
        public int[] shape;
        public static NDArray Create(int[] s, double i) {
            //Create Array with size s and values i
            NDArray nDArray = new NDArray();
            nDArray.shape = s;
            nDArray.arr = new double[s[0]*s[1]];
            for(int l = 0; l < nDArray.arr.Length; l++)
                nDArray.arr[l] = i;
            return nDArray;
        }
        public static NDArray Create(int[] s, double[] i) {
            //Create Array with size s and values in i
            NDArray nDArray = new NDArray();
            if(s[0] * s[1] != i.Length)
                return null;
            nDArray.shape = s;
            nDArray.arr = i;
            return nDArray;
        }
        public static NDArray CreateColumnRandom(int s) {
            //Create Column Array with size s and random values
            NDArray nDArray = new NDArray();
            nDArray.shape = new int[] { s, 1 };
            nDArray.arr = new double[s];
            for(int i = 0; i < s; i++)
                nDArray.arr[i] = new CryptoRandom().RandomValue;
            return nDArray;
        }
        public static NDArray CreateRandom(int[] s) {
            //Create Array with shape s and random values
            NDArray nDArray = new NDArray();
            nDArray.shape = s;
            nDArray.arr = new double[s[0] * s[1]];
            for(int l = 0; l < nDArray.arr.Length; l++)
                nDArray.arr[l] = new CryptoRandom().RandomValue;
            return nDArray;
        }
        public static NDArray Sigmoid(NDArray x) {
            NDArray y = x;
            for(int i = 0; i < x.arr.Length; i++)
                y.arr[i] = 1d / (1d+Math.Exp(-x.arr[i]));
            return y;
        }
        public void PrintArr() {
            for(int y = 0; y < shape[1]; y++) {
                for(int x = 0; x < shape[0]; x++)
                    Console.Write(arr[y * shape[1] + x] + ", ");
                Console.WriteLine();
            } 
        }
        public NDArray Mul(NDArray a) {
            if(shape[1] != a.shape[0]||shape[0]!=a.shape[1])
                return null;
            NDArray b = NDArray.Create(new int[] { shape[1], a.shape[0] }, 0);
            Parallel.For(0, b.shape[1], y => {
                for(int x = 0; x < b.shape[0]; x++) {
                    double temp = 0;
                    for(int i = 0; i < shape[0]; i++)
                        temp += arr[y * shape[0] + i] * a.arr[i * a.shape[0] + x];
                    b.arr[y * shape[1] + x] = temp;
                }
            });
            return b;
        }
        public NDArray Add(NDArray a) {
            if(shape[1] != a.shape[1] || shape[0] != a.shape[0])
                return null;
            NDArray b = NDArray.Create(a.shape, 0);
            Parallel.For(0, b.shape[1], y => {
                for(int x = 0; x < b.shape[0]; x++)
                    b.arr[y * shape[1] + x] = arr[y * shape[1] + x] + a.arr[y * a.shape[1] + x];
            }); 
            return b;
        }
    }

    class CryptoRandom {
        public double RandomValue {
            get; set;
        }
        public CryptoRandom() {
            using(RNGCryptoServiceProvider p = new RNGCryptoServiceProvider()) {
                Random r = new Random(p.GetHashCode());
                this.RandomValue = r.NextDouble();
            }
        }

    }
}
