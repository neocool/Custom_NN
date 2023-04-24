from copy import deepcopy
import math
import random
from decimal import Decimal, getcontext
getcontext().prec = 150

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            #print("Layer:", layer.__class__.__name__)
            #print("Output shape:", len(x), len(x[0]))
        return x
    
    def update_parameters(self, optimizer_parameters):
        for i, layer in enumerate(self.layers):
            layer.weights = optimizer_parameters[i][0]
            layer.bias = optimizer_parameters[i][1]

    def backward(self, grad_output):        
        gradients = []
        grad_input, grad_weights, grad_bias = self.layers[-1].backward(grad_output)
        gradients.append((grad_weights, grad_bias))

        for layer_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[layer_idx]
  
            grad_input, grad_weights, grad_bias = layer.backward(grad_input)
            gradients.append((grad_weights, grad_bias))

            # Update grad_output for the next iteration
            grad_output = grad_input

        return gradients[::-1]  # Reverse the list to match the order of layers
    
class mixedNetwork(NeuralNetwork):
    def __init__(self,insize,outsize,hiddennodes):
        super().__init__()
        self.inputLayer1 = Linear(insize, hiddennodes*3)
        self.layers.append(self.inputLayer1)
        
        self.conv1 = Conv1D(3,3,3,1,1)
        self.layers.append(self.conv1)

        self.conv2 = Conv1D(3,3,3,1,1)
        self.layers.append(self.conv2)

        self.conv3 = Conv1D(3,3,3,1,1)
        self.layers.append(self.conv3)

        self.conv4 = Conv1D(3,3,3,1,1)
        self.layers.append(self.conv4)

        self.conv5 = Conv1D(3,3,3,1,1)
        self.layers.append(self.conv5)

        self.outLayer1 = Linear(hiddennodes*3, hiddennodes)
        self.layers.append(self.outLayer1)

        self.outLayer2 = Linear(hiddennodes, outsize)
        self.layers.append(self.outLayer2)

    def forward(self,x):
        nn_input1 = self.inputLayer1.forward(x)
        nn_input1 = pow(nn_input1,3)

        nn_input = []
        for example in nn_input1:
            featureList = []
            featuresNum = int(len(example)/ 3)
            featuresStart = 0
            featuresEnd = featuresStart +  featuresNum
            for i in range(3):                                
                featureList.append(example[featuresStart:featuresEnd])
                featuresStart = featuresStart +  featuresNum
                featuresEnd = featuresStart +  featuresNum
            nn_input.append(featureList)
               
        nn_conv1 = leaky_relu(self.conv1.forward(nn_input))
        nn_conv2 = leaky_relu(self.conv2.forward(nn_conv1))
        nn_conv3 = leaky_relu(self.conv3.forward(nn_conv2))
        nn_conv4 = self.conv4.forward(nn_conv3)
        nn_conv5 = leaky_relu(self.conv5.forward(nn_conv4))
        
        convOut = []
        for i in range(len(nn_conv5)):
            features = []
            for m in range(len(nn_conv5[0])):
                for item in nn_conv5[i][m]:
                    features.append(item)
            convOut.append(features)  
        
        outLayer1 = self.outLayer1.forward(convOut)
        outLayer2 = self.outLayer2.forward(outLayer1)  
        return outLayer2

class Module:
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError("Forward method should be implemented in the derived class")

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = [[Decimal(random.uniform(-0.1, 0.1)) for _ in range(in_features)] for _ in range(out_features)]
        self.bias = [[Decimal(random.uniform(-0.1, 0.1))] for _ in range(out_features)]

    def forward(self, x):
        # Store the input tensor for use in the backward pass
        self.input = x
        
        # Compute the linear transformation of the input tensor
        linear_output = []
        for b in range(len(x)):
            row_output = []
            for i in range(len(self.weights)):
                weighted_sum = 0
                for j in range(len(x[b])):                    
                    weighted_sum += x[b][j] * self.weights[i][j]
                row_output.append(weighted_sum + self.bias[i][0])
            linear_output.append(row_output)
        
        # Return the linear transformation output
        return linear_output

    def backward(self, grad_output):
        grad_input = [[sum([self.weights[j][i] * grad_output[b][j] for j in range(self.out_features)]) for i in range(self.in_features)] for b in range(len(grad_output))]       
        batch_size = len(self.input)

        grad_weights = []
        for j in range(self.out_features):
            temp_row = []
            for i in range(self.in_features):
                temp_sum = 0
                for b in range(batch_size):              
                    temp_sum += grad_output[b][j] * Decimal(str(self.input[b][i]))
                temp_row.append(temp_sum / batch_size)  # Divide by batch_size
            grad_weights.append(temp_row)

        grad_bias = [[Decimal(sum([grad_output[b][i] for b in range(len(grad_output))])) / batch_size] for i in range(self.out_features)]  # Divide by batch_size
        return grad_input, grad_weights, grad_bias

    @property
    def parameters(self):
        return [self.weights, self.bias]

class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = [[[Decimal(random.uniform(-0.1, 0.1)) for _ in range(kernel_size)] for _ in range(in_channels)] for _ in range(out_channels)]
        self.bias = [Decimal(random.uniform(-0.1, 0.1)) for _ in range(out_channels)]

    @property
    def parameters(self):
        return [self.weights, self.bias]
    
    def forward(self, x):
        self.input = x
        batch_size = len(x)
        in_features = len(x[0][0])

        # Apply padding to the input
        padded_input = []
        for b in range(batch_size):
            padded_input_for_b = []
            for c in range(self.in_channels):
                padded_input_for_c = []
                for i in range(-self.padding, len(x[b][c]) + self.padding):
                    if 0 <= i < len(x[b][c]):
                        padded_input_for_c.append(Decimal(x[b][c][i]))
                    else:
                        padded_input_for_c.append(Decimal(0))
                padded_input_for_b.append(padded_input_for_c)
            padded_input.append(padded_input_for_b)
        
       
        out_features = in_features + 2 * self.padding - self.kernel_size + 1
        output = [[[Decimal(0) for _ in range(out_features)] for _ in range(self.out_channels)] for _ in range(batch_size)]

        for example in range(batch_size): 
            # Iterate through each output channel           
            for out_channel in range(self.out_channels):
                # Slide the kernel along the feature axis and apply the convolution operation
                for feature in range(out_features):
                    # Take a slice of size kernel_size from the input features in each input channel
                    input_slice = [padded_input[example][channel][feature : feature + self.kernel_size] for channel in range(self.in_channels)]

                    # Multiply the corresponding values of the kernel and the input slice element-wise
                    multiplied_values = [[input_slice[channel][k] * Decimal(self.weights[out_channel][channel][k]) for k in range(self.kernel_size)] for channel in range(self.in_channels)]
                
                    # Sum up the multiplied values across all input channels and add the bias for the current output channel
                    row_sums = [sum(row) for row in multiplied_values]
                    total_sum = sum(row_sums)
                    output_feature = Decimal(total_sum) + Decimal(self.bias[out_channel])

                    # Store the output feature value in the output tensor
                    output[example][out_channel][feature] = output_feature
        
        
        # Use output tensor to calculate the convolution output
        conv_output = []
        for b in range(batch_size):
            out_channels = []  
            for k in range(self.out_channels):
                out_seq = []
                for i in range(out_features):
                    out_seq.append(output[b][k][i])
                out_channels.append(out_seq)  
            conv_output.append(out_channels)
        
        return conv_output
    
    def backward(self, grad_output):  
        grad_output_reshaped = []
        for i in range(len(grad_output)): 
            featuresNum = int(len(grad_output[i])/ self.out_channels)
            featuresStart = 0
            featuresEnd = featuresStart +  featuresNum
            
            tempList =[]
            for m in range(self.out_channels):
                feature_list = grad_output[i][featuresStart:featuresEnd]
                featuresStart = featuresStart +  featuresNum
                featuresEnd = featuresStart +  featuresNum
                
                tempList.append(feature_list)

            grad_output_reshaped.append(tempList)
        
        grad_output = grad_output_reshaped
        batch_size = len(grad_output)
        in_features = len(grad_output[0][0])

        # Add padding to grad_output
        padded_grad_output = []
        for b in range(batch_size):
            padded_grad_output_b = []
            for c in range(self.out_channels):
                padded_grad_output_c = [0] * self.padding + grad_output[b][c] + [0] * self.padding
                padded_grad_output_b.append(padded_grad_output_c)
            padded_grad_output.append(padded_grad_output_b)

        grad_input = []
        for b in range(batch_size):
            grad_input_b = []
            for c in range(self.in_channels):
                grad_input_c = []
                for i in range(len(self.input[0][c])):
                    temp_sum = Decimal(0)
                    for j in range(self.kernel_size):
                        for k in range(self.out_channels):
                            if 0 <= i - j < len(padded_grad_output[b][k]):
                                temp_sum += self.weights[k][c][j] * padded_grad_output[b][k][i - j]
                    grad_input_c.append(temp_sum)
                grad_input_b.extend(grad_input_c)  # Flatten inner lists
            grad_input.append(grad_input_b)


        batch_size = len(grad_output)
        
        grad_weights = []
        for k in range(self.out_channels):
            grad_weights_k = []
            for j in range(self.kernel_size):
                grad_weights_j = []
                for c in range(self.in_channels):
                    grad_weights_jc = Decimal(0)
                    for i in range(len(self.input) - self.kernel_size + 1):
                        grad_weights_jc += sum(decimalList_multiply_list(self.input[i][c], padded_grad_output[i+j][k]))
                    grad_weights_j.append(grad_weights_jc / batch_size)  # Divide by batch_size
                grad_weights_k.append(grad_weights_j)
            grad_weights.append(grad_weights_k)

        grad_bias = []
        for k in range(self.out_channels):
            grad_bias_k = Decimal(0)
            for i in range(len(self.input) - self.kernel_size + 1):
                grad_bias_k += sum(padded_grad_output[i][k])

            grad_bias.append([grad_bias_k / batch_size])  # Divide by batch_size

        return grad_input, grad_weights, grad_bias

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # Compute the dot product between Q and the transpose of K
        K_T = list(map(list, zip(*K)))  # Transpose K
        matmul_QK = decimal_matrix_multiply(Q, K_T)
        dk = Decimal(self.head_dim)
        scaled_attention_logits = [[qk / (dk.sqrt()) for qk in row] for row in matmul_QK]
        attention_weights = [[elem.exp() for elem in row] for row in scaled_attention_logits]
        sum_attention_weights = [sum(row) for row in attention_weights]
        attention_weights = [[weight / total for weight in row] for row, total in zip(attention_weights, sum_attention_weights)]
        return decimal_matrix_multiply(attention_weights, V)

    def split_heads(self, x):
        return [[x[b][head * self.head_dim : (head + 1) * self.head_dim] for head in range(self.num_heads)] for b in range(len(x))]


    def concat_heads(self, x):
        return [[Decimal(elem) for head in range(self.num_heads) for elem in x[b][head]] for b in range(len(x))]

    def forward(self, Q, K, V):
        Q = self.W_q.forward([[row[0] for row in example] for example in Q])
        K = self.W_k.forward([[row[0] for row in example] for example in K])
        V = self.W_v.forward([[row[0] for row in example] for example in V])

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention = []
        for b in range(len(Q)):
            batch_attention = self.scaled_dot_product_attention(Q[b], K[b], V[b])
            attention.append(batch_attention)

        attention = self.concat_heads(attention)
        return self.W_o.forward(attention)

    @property
    def parameters(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]

class PositionwiseFeedForwardNetwork(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = [[max(Decimal(0), elem) for elem in row] for row in x]  # ReLU
        return self.fc2.forward(x)

    @property
    def parameters(self):
        return [self.fc1.parameters, self.fc2.parameters]

class TransformerLayer(Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x):
        mha_output = self.mha.forward(x, x, x)
        print("x shape:", len(x), len(x[0]))
        print("mha_output shape:", len(mha_output), len(mha_output[0]))
        print("x element type:", type(x[0][0]))
        print("mha_output element type:", type(mha_output[0][0]))

        x_norm1 = self.norm1.forward(decimal_matrix_add(x, mha_output))

        ffn_output = self.ffn.forward(x_norm1)
        x_norm2 = self.norm2.forward(decimal_matrix_add(x_norm1, ffn_output))

        return x_norm2

    @property
    def parameters(self):
        return [self.mha.parameters, self.ffn.parameters]
  
class LayerNormalization(Module):
    def __init__(self, d_model, epsilon=Decimal(1e-6)):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = [[Decimal(1)] for _ in range(d_model)]
        self.beta = [[Decimal(0)] for _ in range(d_model)]

    def forward(self, x):
        mean = [Decimal(sum(row)) / Decimal(len(row)) for row in x]
        var = [Decimal(sum([(elem - mean_i) ** 2 for elem in row])) / Decimal(len(row)) for row, mean_i in zip(x, mean)]
        x_hat = [[(x[b][i] - mean[b]) / ((var[b] + self.epsilon).sqrt()) for i in range(len(x[0]))] for b in range(len(x))]
        return [[self.gamma[i][0] * x_hat[b][i] + self.beta[i][0] for i in range(len(x[0]))] for b in range(len(x))]


class Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method should be implemented in the derived class")

    def backward(self, y_pred, y_true):
        return [Decimal(1) if y_pred[i] > y_true[i] else Decimal(-1) for i in range(len(y_pred))]

class L1Loss(Loss):
    def forward(self, y_pred, y_true):
        batch_size = len(y_pred)
        total_loss = Decimal(0)
        for i in range(batch_size):
            total_loss += sum([abs(y_pred[i][j] - y_true[i][j]) for j in range(len(y_pred[i]))])
        
        return total_loss / batch_size

    def backward(self, y_pred, y_true):
        #print(f"y_pred: {y_pred}")
        #print(f"y_true: {y_true}")
        gradients =  [[Decimal('1') if y_pred[i][j] > y_true[i][j] else Decimal('-1') for j in range(len(y_pred[i]))] for i in range(len(y_pred))]

        #print(f"gradients: {gradients}")
        #input()
        return gradients

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self, grads):
        for grad, param in zip(grads, self.params):
            for k in param.keys():
                param[k] -= self.lr * grad[k]

class Adam(Optimizer):
    def __init__(self, params,lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params,lr)
        self.parameters = self.params
        self.beta1 = Decimal(str(beta1))
        self.beta2 = Decimal(str(beta2))
        self.epsilon = Decimal(str(epsilon))
        self.lr = Decimal(lr)
        self.m = create_zeros_structure(self.params)
        self.v = create_zeros_structure(self.params)

        self.t = 0

    def step(self, gradients):
        self.t += 1
        for i in range(len(self.parameters)):
            grad_layer_weights = gradients[i][0]
            param_layer_weights = self.parameters[i][0]            
            m_layer_weights = self.m[i][0]            
            v_layer_weights = self.v[i][0]

            # For biases
            grad_layer_biases = gradients[i][1]
            param_layer_biases = self.parameters[i][1]
            m_layer_biases = self.m[i][1]
            v_layer_biases = self.v[i][1]

            #items = count_elements(grad_layer_weights)
            for k in range(len(grad_layer_weights)):                
                grad_weights = grad_layer_weights[k][0]
                grad_biases = grad_layer_biases[k][0]

                if isinstance(grad_weights,list):
                    for z in range(len(grad_weights)):
                        # Update first and second moment estimates for weights
                        m_layer_weights[k][0][z] = self.beta1 * m_layer_weights[k][0][z] + (1 - self.beta1) * grad_weights[z]
                        v_layer_weights[k][0][z] = self.beta2 * v_layer_weights[k][0][z] + (1 - self.beta2) * (grad_weights[z] ** 2)

                        # Compute bias-corrected first and second moment estimates for weights
                        m_hat_weights = m_layer_weights[k][0][z] / (1 - self.beta1 ** self.t)
                        v_hat_weights = v_layer_weights[k][0][z] / (1 - self.beta2 ** self.t)

                        # Update weights
                        param_layer_weights[k][0][z] -= self.lr * m_hat_weights / ((v_hat_weights).sqrt() + self.epsilon)

                    # Update first and second moment estimates for biases
                    m_layer_biases[k] = self.beta1 * m_layer_biases[k] + (1 - self.beta1) * grad_biases
                    v_layer_biases[k] = self.beta2 * v_layer_biases[k] + (1 - self.beta2) * (grad_biases ** 2)

                    # Compute bias-corrected first and second moment estimates for biases
                    m_hat_biases = m_layer_biases[k]/ (1 - self.beta1 ** self.t)
                    v_hat_biases = v_layer_biases[k] / (1 - self.beta2 ** self.t)

                    # Update biases
                    param_layer_biases[k] -= self.lr * m_hat_biases / ((v_hat_biases).sqrt() + self.epsilon)
                
                else:
                    # Update first and second moment estimates for weights
                    m_layer_weights[k][0] = self.beta1 * m_layer_weights[k][0] + (1 - self.beta1) * grad_weights
                    v_layer_weights[k][0] = self.beta2 * v_layer_weights[k][0] + (1 - self.beta2) * (grad_weights ** 2)

                    # Compute bias-corrected first and second moment estimates for weights
                    m_hat_weights = m_layer_weights[k][0] / (1 - self.beta1 ** self.t)
                    v_hat_weights = v_layer_weights[k][0] / (1 - self.beta2 ** self.t)

                    # Update weights
                    param_layer_weights[k][0] -= self.lr * m_hat_weights / ((v_hat_weights).sqrt() + self.epsilon)

                    # Update first and second moment estimates for biases
                    m_layer_biases[k][0] = self.beta1 * m_layer_biases[k][0] + (1 - self.beta1) * grad_biases
                    v_layer_biases[k][0] = self.beta2 * v_layer_biases[k][0] + (1 - self.beta2) * (grad_biases ** 2)

                    # Compute bias-corrected first and second moment estimates for biases
                    m_hat_biases = m_layer_biases[k][0] / (1 - self.beta1 ** self.t)
                    v_hat_biases = v_layer_biases[k][0] / (1 - self.beta2 ** self.t)

                    # Update biases
                    param_layer_biases[k][0] -= self.lr * m_hat_biases / ((v_hat_biases).sqrt() + self.epsilon)
 

def decimal_multiply_list(decimal_value, decimal_list):
    return [decimal_value * item for item in decimal_list]

def decimalList_multiply_list(list1, list2):
    return [item1 * item2 for item1, item2 in zip(list1, list2)]

def nntensor(data, dtype=Decimal):
    if isinstance(data[0], (list, tuple)):
        return [[dtype(feature) for feature in item] for item in data]
    else:
        return [[dtype(item)] for item in data]

def pow(list1, exponents):
    if isinstance(exponents, list):
        if len(list1) != len(exponents):
            raise ValueError("Lists must have same shape")
        data = []
        for example,exponent in zip(list1, exponents):
            dataExample = []
            for val, exp in zip(example,exponent):
                dataExample.append(abs(val) ** abs(exp))

            data.append(dataExample)
        return data
    else:
        data = []
        for example in list1:
            data.append([val** Decimal(exponents) for val in example])
        return data

def list_shape(lst):
    if not isinstance(lst, list):
        return []

    shape = [len(lst)]
    if len(lst) > 0 and isinstance(lst[0], list):
        shape.extend(list_shape(lst[0]))

    return tuple(shape)

def leaky_relu(x, alpha=0.01):
    return [
        [
            [max(Decimal(alpha) * element, element) for element in channel]
            for channel in example
        ]
        for example in x
    ]

def count_elements(lst):
    if isinstance(lst, list):
        return sum(count_elements(sublist) for sublist in lst)
    else:
        return 1

def create_zeros_structure(params):
    if isinstance(params, list):
        return [create_zeros_structure(sublist) for sublist in params]
    else:
        return Decimal(0)

class mixedNetwork2(NeuralNetwork):
    def __init__(self,insize,outsize,hiddennodes):
        super().__init__()
        self.inputLayer1 = Linear(insize, hiddennodes*3)
        self.layers.append(self.inputLayer1)
        self.convlayersNum = 20
        self.convlayers = []
        for i in range(self.convlayersNum):
            self.convlayers.append(Conv1D(3,3,3,1,1))
            self.layers.append(self.convlayers[i])

        self.transformer1 = TransformerLayer(32,8,64)
        self.layers.append(self.transformer1)

        self.outLayer1 = Linear(hiddennodes*3, hiddennodes)
        self.layers.append(self.outLayer1)

        self.outLayer2 = Linear(hiddennodes, outsize)
        self.layers.append(self.outLayer2)

    def forward(self,x):
        nn_input1 = self.inputLayer1.forward(x)
        
        #print(list_shape(nn_input1))

        nn_input = []
        for example in nn_input1:
            featureList = []
            featuresNum = int(len(example)/ 3)
            featuresStart = 0
            featuresEnd = featuresStart +  featuresNum
            for i in range(3):                                
                featureList.append(example[featuresStart:featuresEnd])
                featuresStart = featuresStart +  featuresNum
                featuresEnd = featuresStart +  featuresNum
            nn_input.append(featureList)
        
        nn_conv = nn_input
        for i in range(self.convlayersNum):
            nn_conv = self.convlayers[i].forward(nn_conv)
        
        
        convOut = []
        for i in range(len(nn_conv)):
            features = []
            for m in range(len(nn_conv[0])):
                for item in nn_conv[i][m]:
                    features.append(item)
            convOut.append(features)
        
        # Convert the shape to (1, 300, 1)
        #convOut = [[[x] for x in inner_list] for inner_list in convOut]

        print(list_shape(convOut))
        input()
        nn_transfrom = self.transformer1.forward(nn_conv)
        
        outLayer1 = self.outLayer1.forward(convOut)
        outLayer2 = self.outLayer2.forward(outLayer1)  
        return outLayer2


def decimal_matrix_multiply(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    result = [[Decimal(0) for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def decimal_matrix_add(matrix1, matrix2):
    print(f"Matrix1 shape: {len(matrix1)} {len(matrix1[0])}")
    print(f"Matrix2 shape: {len(matrix2)} {len(matrix2[0])}")
    print(f"Expected result shape: {len(matrix1)} {len(matrix1[0])}")
    result = [[0 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            print(f"Adding {matrix1[i][j]} and {matrix2[i][j]}")
            result[i][j] = matrix1[i][j] + matrix2[i][j]
    return result
