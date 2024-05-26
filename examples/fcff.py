from lalagrad import Tensor


class FCFF:
    def __init__(self, layers):
        self.weights = [Tensor.rand(shape=(layers[i-1], layers[i])) for i in range(1, len(layers))]
        self.biases = [Tensor.rand(shape=(1, layers[i])) for i in range(1, len(layers))]
        
    def forward(self, x):
        for l, b in zip(self.weights, self.biases):
            x = x.matmul(l)
            x.Relu()
            x += b
        x.Softmax() 
        return x
    
if __name__ == "__main__":
    model = FCFF([3, 4, 2, 2])
    x = Tensor.rand(shape=(1, 3))
    
    print(model.forward(x).tolist())