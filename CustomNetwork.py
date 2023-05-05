import Classes.Altial as nn
import math
import random
import statistics

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = [[nn.Decimal(random.uniform(-0.1, 0.1)) for _ in range(in_features)] for _ in range(out_features)]
        self.biases = [[nn.Decimal(random.uniform(-0.1, 0.1))] for _ in range(out_features)]
        self.grad_weights = [[nn.Decimal(0) for _ in range(in_features)] for _ in range(out_features)]
        self.grad_bias = [[nn.Decimal(0)] for _ in range(out_features)]

    def update_parameters(self, new_weights, new_biases):
        updated_weights = []
        counter = 0
        for i in range(nn.list_shape(self.weights)[0]):
            row = []
            for z in range(nn.list_shape(self.weights)[1]):
                row.append(new_weights[counter])
                counter += 1
            updated_weights.append(row)

        self.weights = updated_weights
        

        updated_biases = []
        counter = 0
        for i in range(nn.list_shape(self.biases)[0]):
            row = []
            for z in range(nn.list_shape(self.biases)[1]):
                row.append(new_biases[counter])
                counter += 1
            updated_biases.append(row)

        self.biases = updated_biases

    def forward(self, x):
        # Store the input tensor for use in the backward pass
        self.input = x
        
        # Compute the linear transformation of the input tensor
        linear_output = []
        batchsize = len(x)
        
        for b in range(batchsize):
            row_output = []            
            for i in range(len(self.weights)):
                weighted_sum = 0                
                for j in range(len(x[b])):                  
                    weighted_sum += x[b][j] * self.weights[i][j]    
   
                row_output.append(weighted_sum + self.biases[i][0])
            linear_output.append(row_output)
        
        # Return the linear transformation output
        self.linear_output = linear_output
        return linear_output

    def backward(self, grad_output):
        grad_input = []
        
        weights_transposed = nn.decimal_matrix_transpose(self.weights)
        for b in range(len(grad_output)):
            batch_grad_input = []            
            for i in range(self.in_features):
                sum_weighted_sums = 0                
                for j in range(self.out_features):
                    sum_weighted_sums += weights_transposed[i][j] * grad_output[b][j]                   
                batch_grad_input.append(sum_weighted_sums)                
            grad_input.append(batch_grad_input)

        
        batch_size = len(self.input)
        grad_weights = []
        input_transposed = nn.decimal_matrix_transpose(self.input)
        for j in range(self.out_features):
            temp_row = []
            for i in range(self.in_features):
                temp_sum = []
                for b in range(batch_size): 
                    temp_sum.append(input_transposed[i][b] * grad_output[b][j])
                temp_row.append(nn.Decimal(sum(temp_sum)) / nn.Decimal(batch_size))  # Divide by batch_size
            grad_weights.append(temp_row)


        grad_bias = []
        for i in range(self.out_features):
            bias_sum = []
            for b in range(len(grad_output)):
                bias_sum.append(grad_output[b][i])
            bias_mean = nn.Decimal(sum(bias_sum)) / nn.Decimal(len(grad_output))
            grad_bias.append([bias_mean])


        #input()
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        returndata =   [grad_input, grad_weights, grad_bias]

        return returndata

    @property
    def parameters(self):
        return [nn.flatten_nested_structure(self.weights),nn.flatten_nested_structure(self.biases)]

class mixedNetwork(nn.NeuralNetwork):
    def __init__(self,insize,outsize,hiddennodes,linear=3):
        super().__init__()   

        #Network 1
        self.inputLayer = CustomLinear(insize, hiddennodes)
        self.layers.append(self.inputLayer)

        self.hidden1 = CustomLinear(hiddennodes, hiddennodes)
        self.layers.append(self.hidden1)

        self.out = CustomLinear(hiddennodes, outsize)
        self.layers.append(self.out)
        
        self.networks.append(self.layers)

    def forward(self,x):
        nn_input = self.inputLayer.forward(x)    
        
        nn_hidden = self.hidden1.forward(nn_input)  
     
        nn_out = self.out.forward(nn_hidden)
        return nn_out

