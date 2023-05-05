from copy import deepcopy
import math
import random
from decimal import Decimal, getcontext
import statistics
getcontext().prec = 150

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.networks = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def update_parameters(self, optimizer_parameters):        
        for b in range(len(self.networks)):
            for i, layer in enumerate(self.networks[b]):
                layer.update_parameters(optimizer_parameters[b][i][0],optimizer_parameters[b][i][1])

    def backward(self, grad_output):        
        network_gradients = []
        for i in range(len(self.networks)):
            network_grad_output = grad_output
            gradients = []
            grad_input, grad_weights, grad_bias = self.networks[i][-1].backward(network_grad_output)
            gradients.append([flatten_nested_structure(grad_weights), flatten_nested_structure(grad_bias)])

            for layer_idx in reversed(range(len(self.networks[i]) - 1)):
                layer = self.networks[i][layer_idx]
    
                grad_input, grad_weights, grad_bias = layer.backward(grad_input)
                gradients.append([flatten_nested_structure(grad_weights), flatten_nested_structure(grad_bias)])

            network_gradients.append(gradients[::-1] )

        return network_gradients 
    
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
        self.biases = [[Decimal(random.uniform(-0.1, 0.1))] for _ in range(out_features)]
        self.grad_weights = [[Decimal(0) for _ in range(in_features)] for _ in range(out_features)]
        self.grad_bias = [[Decimal(0)] for _ in range(out_features)]

    def update_parameters(self, new_weights, new_biases):
        updated_weights = []
        counter = 0
        for i in range(list_shape(self.weights)[0]):
            row = []
            for z in range(list_shape(self.weights)[1]):
                row.append(new_weights[counter])
                counter += 1
            updated_weights.append(row)

        self.weights = updated_weights
        

        updated_biases = []
        counter = 0
        for i in range(list_shape(self.biases)[0]):
            row = []
            for z in range(list_shape(self.biases)[1]):
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
        
        weights_transposed = decimal_matrix_transpose(self.weights)
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
        input_transposed = decimal_matrix_transpose(self.input)
        for j in range(self.out_features):
            temp_row = []
            for i in range(self.in_features):
                temp_sum = []
                for b in range(batch_size): 
                    temp_sum.append( input_transposed[i][b] * grad_output[b][j])
                temp_row.append(statistics.median(temp_sum))  # Divide by batch_size
            grad_weights.append(temp_row)

        grad_bias = []
        for i in range(self.out_features):
            bias_sum = []
            for b in range(len(grad_output)):
                bias_sum.append(grad_output[b][i])
            bias_median = statistics.median(bias_sum)
            grad_bias.append([bias_median])

        #input()
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        returndata =   [grad_input, grad_weights, grad_bias]

        return returndata

    @property
    def parameters(self):
        return [flatten_nested_structure(self.weights),flatten_nested_structure(self.biases)]


class Conv1D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = [[[Decimal(random.uniform(-0.1, 0.1)) for _ in range(kernel_size)] for _ in range(in_channels)] for _ in range(out_channels)]        
        self.bias = [Decimal(random.uniform(-0.1, 0.1)) for _ in range(out_channels)]

    def unflatten_weights(self):
        weights = []
        weight_start = 0
        
        for out_channel in range(self.out_channels):
            weights_out = []
            for in_channel in range(self.in_channels):
                weights_in = self.weights[weight_start:weight_start + self.kernel_size]
                weights_out.append(weights_in)
                weight_start += self.kernel_size
            
            weights.append(weights_out)           

        self.weights = weights

    @property
    def parameters(self):
        return [flatten_nested_structure(self.weights), flatten_nested_structure(self.bias)]

    def forward(self, x):        
        if not isinstance(self.weights[0],list):
            self.unflatten_weights()
        
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

        grad_conv_weights  = flatten_nested_structure(grad_weights)
        grad_conv_biases  = flatten_nested_structure(grad_bias)

        return grad_input, grad_conv_weights , grad_conv_biases

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

    def scaled_dot_product_attention(self, q, k, v):
        q = [q]
        k = [k]
        v = [v]

        dk = Decimal(len(q[0]))

        QK = decimal_matrix_multiply(q, k)

        # Scale the QK matrix
        scaled_attention_logits = [[x / dk.sqrt() for x in row] for row in QK]

        # Apply the softmax function to the scaled matrix
        attention_weights = custom_softmax(scaled_attention_logits)

        # Multiply the attention weights matrix by the value matrix

        output = decimal_matrix_multiply(attention_weights, v)

        self.attention_weights = attention_weights
    
        return output

    def split_heads(self, x):
        return [[x[b][head * self.head_dim : (head + 1) * self.head_dim] for head in range(self.num_heads)] for b in range(len(x))]

    def split_grad_heads(self, x):
        calculated_num_heads = list_shape(x)[1] // self.head_dim
        
        result = [
            [
                [x[b][head * self.head_dim : (head + 1) * self.head_dim]]
                for head in range(calculated_num_heads)
            ]
            for b in range(len(x))
        ]
        return result

    def concat_heads(self, x):
        return [[head_elem for head in x[b] for head_elem in head] for b in range(len(x))]
    
    def unflatten(self, x, row_count, col_count):
        return [[x[row * col_count + col] for col in range(col_count)] for row in range(row_count)]

    def forward(self, Q_input, K_input, V_input):        
        self.W_q.parameters[0] = self.unflatten(self.allweights[0:self.d_model * self.d_model], self.d_model, self.d_model)
        self.W_k.parameters[0] = self.unflatten(self.allweights[self.d_model * self.d_model:2 * self.d_model * self.d_model], self.d_model, self.d_model)
        self.W_v.parameters[0] = self.unflatten(self.allweights[2 * self.d_model * self.d_model:3 * self.d_model * self.d_model], self.d_model, self.d_model)
        self.W_o.parameters[0] = self.unflatten(self.allweights[3 * self.d_model * self.d_model:4 * self.d_model * self.d_model], self.d_model, self.d_model)
        
        self.W_q.parameters[1] = self.allbiases[0:self.d_model]
        self.W_k.parameters[1] = self.allbiases[self.d_model :2 * self.d_model]
        self.W_v.parameters[1] = self.allbiases[2 * self.d_model :3 * self.d_model]
        self.W_o.parameters[1] = self.allbiases[3 * self.d_model :4 * self.d_model ]
        
        Q = self.W_q.forward([[row for row in example] for example in Q_input])
        K = self.W_k.forward([[row for row in example] for example in K_input])
        V = self.W_v.forward([[row for row in example] for example in V_input])

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Initialize empty attention list
        attention = [[] for _ in range(len(Q))]
        all_attention_weights = [[] for _ in range(len(Q))]

        for b in range(len(Q)):
            batch_attention_heads = []
            batch_attention_weights = []
            for i in range(self.num_heads):
                Q_head = Q[b][i]
                K_head = K[b][i]
                V_head = V[b][i]

                head_attention = self.scaled_dot_product_attention(Q_head, K_head, V_head)
                batch_attention_heads.append(head_attention[0])

                # Save attention_weights for each head
                batch_attention_weights.append(self.attention_weights)

            # Concatenate the attention heads
            attention[b] = [token for head in batch_attention_heads for token in head]
            all_attention_weights[b] = batch_attention_weights
        
        self.all_attention_weights = all_attention_weights
        w_o_forward = self.W_o.forward(attention)

        return w_o_forward

    def backward(self, grad_output, attention_weights):
        # Compute the gradients of the output of the linear layer with respect to the output of the attention layer.
        grad_input, _, _ = self.W_o.backward(grad_output)
        grad_attention = grad_input

        # Unconcatenate the heads
        grad_attention_heads = self.split_grad_heads(grad_attention)   
      
        # Initialize the gradients for Q, K, and V
        grad_Q = [[[] for _ in range(self.num_heads)] for _ in range(len(grad_attention_heads))]
        grad_K = [[[] for _ in range(self.num_heads)] for _ in range(len(grad_attention_heads))]
        grad_V = [[[] for _ in range(self.num_heads)] for _ in range(len(grad_attention_heads))]

        for b in range(len(grad_attention_heads)):
            for i in range(self.num_heads):
                Q_head_grad = decimal_matrix_multiply(decimal_matrix_transpose(grad_attention_heads[b][i]), attention_weights[b][i])
                K_head_grad = decimal_matrix_multiply(decimal_matrix_transpose(attention_weights[b][i]), grad_attention_heads[b][i])
                V_head_grad = decimal_matrix_multiply(attention_weights[b][i], grad_attention_heads[b][i])
                grad_Q[b][i] = Q_head_grad
                grad_K[b][i] = K_head_grad
                grad_V[b][i] = V_head_grad

        # Concatenate the heads back and compute the gradients for W_q, W_k, and W_v

        grad_Q_concat = self.concat_heads(grad_Q)
        grad_K_concat = self.concat_heads(grad_K)
        grad_V_concat = self.concat_heads(grad_V)

        
        grad_Q_concat = [[sum(grad_Q_concat[b][i]) for i in range(len(grad_Q_concat[b]))] for b in range(len(grad_Q_concat))]
        grad_K_concat = [[sum(grad_K_concat[b][i]) for i in range(len(grad_K_concat[b]))] for b in range(len(grad_K_concat))]
        grad_V_concat = [[grad_V_concat[b][i][0] for i in range(len(grad_V_concat[b])) for _ in range(self.head_dim)] for b in range(len(grad_V_concat))]

        W_q_grad_input, W_q_grad_weights, W_q_grad_bias = self.W_q.backward(grad_Q_concat)
        W_k_grad_input, W_k_grad_weights, W_k_grad_bias = self.W_k.backward(grad_K_concat)
        W_v_grad_input, W_v_grad_weights, W_v_grad_bias = self.W_v.backward(grad_V_concat)
        grad_input = decimal_matrix_add(W_q_grad_input, W_k_grad_input)
        grad_input = decimal_matrix_add(grad_input, W_v_grad_input)

        return grad_input, [W_q_grad_weights, W_k_grad_weights, W_v_grad_weights, self.W_o.grad_weights], [W_q_grad_bias, W_k_grad_bias, W_v_grad_bias, self.W_o.grad_bias]
 
    @property
    def parameters(self):
        return [self.W_q.parameters, self.W_k.parameters, self.W_v.parameters, self.W_o.parameters]
    
    @property
    def weights(self):
        return [self.W_q.parameters[0], self.W_k.parameters[0], self.W_v.parameters[0], self.W_o.parameters[0]]
    
    @property
    def biases(self):
        return [self.W_q.parameters[1], self.W_k.parameters[1], self.W_v.parameters[1], self.W_o.parameters[1]]
    
class PositionwiseFeedForwardNetwork(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
    
    def unflatten(self, x, row_count, col_count):
        return [[x[row * col_count + col] for col in range(col_count)] for row in range(row_count)]

    def forward(self, x):
        self.fc1.parameters[0] = self.unflatten(self.allweights[0:self.d_model * self.d_ff], self.d_model, self.d_model)
        self.fc2.parameters[0] = self.unflatten(self.allweights[self.d_model * self.d_ff:], self.d_model, self.d_model)
        
        self.fc1.parameters[1] = self.allbiases[0:self.d_ff]
        self.fc2.parameters[1] = self.allbiases[self.d_ff :]
        

        x = self.fc1.forward(x)
        x = [[max(Decimal(0), elem) for elem in row] for row in x]  # ReLU
        return self.fc2.forward(x)

    def backward(self, grad_output):

        grad_fc2 = self.fc2.backward(grad_output)
        grad_relu = [[1 if elem > 0 else 0 for elem in row] for row in self.fc1.linear_output]

        # Print the shapes of the matrices
        
        # Correct the matrix multiplication for grad_fc1
        grad_fc1 = decimal_matrix_multiply(decimal_matrix_transpose(self.fc2.weights),grad_fc2[0])
     

        grad_fc1 = decimal_matrix_multiply( grad_relu,grad_fc1)

        # Return the gradients for all layers
        returnData = [grad_fc1,[self.fc1.grad_weights, self.fc2.grad_weights], [self.fc1.grad_bias, self.fc2.grad_bias], grad_fc2]

        return returnData


    @property
    def biases(self):
        return [self.fc1.parameters[1], self.fc2.parameters[1]]
    @property
    def weights(self):
        return [self.fc1.parameters[0], self.fc2.parameters[0]]

class TransformerLayer(Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def update_parameters(self,weights,biases):
        self.parameters[0] = weights
        self.parameters[1] = biases
        mha_weights_start = 0
        mha_weights_end = 4*self.d_model*self.d_model
        mha_bias_start = 0
        mha_bias_end = 4*self.d_model
        self.mha.allweights = self.parameters[0][mha_weights_start:mha_weights_end]
        self.mha.allbiases = self.parameters[1][mha_bias_start:mha_bias_end]

        ffn_weights_start = mha_weights_end
        ffn_weights_end = ffn_weights_start + 2*self.d_model*self.d_model
        ffn_bias_start = mha_bias_end
        ffn_bias_end = ffn_bias_start + 2*self.d_model
        self.ffn.allweights = self.parameters[0][ffn_weights_start:ffn_weights_end]
        self.ffn.allbiases = self.parameters[1][ffn_bias_start:ffn_bias_end]

        # Update normalization layer's parameters
        norm1_gamma_start = ffn_weights_end
        norm1_gamma_end = norm1_gamma_start + self.d_model
        norm1_beta_start = ffn_bias_end
        norm1_beta_end = norm1_beta_start + self.d_model
        self.norm1.gamma = [self.parameters[0][norm1_gamma_start:norm1_gamma_end]]
        self.norm1.beta = [self.parameters[1][norm1_beta_start:norm1_beta_end]]

        norm2_gamma_start = norm1_gamma_end
        norm2_gamma_end = norm2_gamma_start + self.d_model
        norm2_beta_start = norm1_beta_end
        norm2_beta_end = norm2_beta_start + self.d_model
        self.norm2.gamma = [self.parameters[0][norm2_gamma_start:norm2_gamma_end]]
        self.norm2.beta = [self.parameters[1][norm2_beta_start:norm2_beta_end]]

    def forward(self, x):
        self.update_parameters(self.parameters[0],self.parameters[1])
        self.x = x
        mha_output = self.mha.forward(x, x, x)

        x_norm1 = self.norm1.forward(decimal_matrix_add(x, mha_output))

        ffn_output = self.ffn.forward(x_norm1)
        x_norm2 = self.norm2.forward(decimal_matrix_add(x_norm1, ffn_output))
        self.all_attention_weights = self.mha.all_attention_weights        
        
        return x_norm2
    
    def backward(self, grad_output):
        # Calculate gradients of ffn_output with respect to loss
        grad_x_norm2, grad_norm2_gamma, grad_norm2_beta = self.norm2.backward(grad_output)

        grad_x_ffn, grad_ffn_weights, grad_ffn_biases, grad_fc2 = self.ffn.backward(grad_x_norm2)

        # Calculate gradients of x_norm1 with respect to loss
        grad_x_norm1, grad_norm1_gamma, grad_norm1_beta = self.norm1.backward(grad_x_ffn)

        # Calculate gradients of x_norm1 with respect to loss              
        grad_x, grad_mha_weights, grad_mha_bias = self.mha.backward(grad_x_norm1, self.all_attention_weights)

        # Calculate gradients of x with respect to loss
        grad_input = grad_x

        grad_transformer_weights  = flatten_nested_structure([grad_mha_weights, grad_ffn_weights,grad_norm1_gamma,grad_norm2_gamma])
        grad_transformer_biases  = flatten_nested_structure([grad_mha_bias, grad_ffn_biases,grad_norm1_beta,grad_norm2_beta])

        return grad_input, grad_transformer_weights, grad_transformer_biases

    @property
    def parameters(self):
        return [flatten_nested_structure([self.mha.weights, self.ffn.weights,self.norm1.gamma,self.norm2.gamma]),
                flatten_nested_structure([self.mha.biases, self.ffn.biases,self.norm1.beta,self.norm2.beta])]

class LayerNormalization(Module):
    def __init__(self, d_model, epsilon=Decimal(1e-6)):
        super().__init__()
        self.d_model = d_model
        self.epsilon = Decimal(epsilon)
        self.gamma = [[Decimal(1) for _ in range(d_model)]]
        self.beta = [[Decimal(0) for _ in range(d_model)]]

    def forward(self, x):       
        self.x = x
        self.mean = [Decimal(sum(row)) / Decimal(len(row)) for row in x]
        self.var = [Decimal(sum([(elem - mean_i) ** 2 for elem in row])) / Decimal(len(row)) for row, mean_i in zip(x, self.mean)]
        self.x_hat = [[(x[b][i] - self.mean[b]) / ((self.var[b] + self.epsilon).sqrt()) for i in range(len(x[0]))] for b in range(len(x))]

        return [[self.gamma[0][i] * self.x_hat[b][i] + self.beta[0][i] for i in range(len(x[0]))] for b in range(len(x))]

    def backward(self, grad_output):
        d_gamma = [[sum([grad_output[b][i] * self.x_hat[b][i] for b in range(len(grad_output))]) for i in range(len(grad_output[0]))]]
        d_beta = [[sum([grad_output[b][i] for b in range(len(grad_output))]) for i in range(len(grad_output[0]))]]

        # Compute the gradients for the input
        d_xhat = [[grad_output[b][i] * self.gamma[0][i] for i in range(len(grad_output[0]))] for b in range(len(grad_output))]
        d_var = [sum([d_xhat[b][i] * (self.mean[b] - elem) * Decimal(-0.5) * ((self.var[b] + self.epsilon) ** Decimal(-1.5)) for i, elem in enumerate(row)]) for b, row in enumerate(grad_output)]
        d_mean = [sum([-d_xhat[b][i] / ((self.var[b] + self.epsilon) ** Decimal(0.5)) for i in range(len(d_xhat[0]))]) + \
                d_var[b] * (-2) * Decimal(len(grad_output[0])) * sum([(elem - self.mean[b]) for elem in row]) / Decimal(len(grad_output[0])) for b, row in enumerate(grad_output)]
        d_x = [[d_xhat[b][i] / ((self.var[b] + self.epsilon) ** Decimal(0.5)) + \
                d_var[b] * (2) * (self.x[b][i] - self
                                  .mean[b]) / Decimal(len(grad_output[0])) + \
                d_mean[b] / Decimal(len(grad_output[0])) for i, elem in enumerate(row)] for b, row in enumerate(grad_output)]

        return d_x, d_gamma, d_beta



    @property
    def parameters(self):
        return [self.gamma, self.beta]

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
            for j in range(len(y_pred[i])):
                diff = y_pred[i][j] - y_true[i][j]
                abs_diff = abs(diff)
                total_loss += abs_diff

        return total_loss / batch_size

    def backward(self, y_pred, y_true):
        gradients = []
        #loop over examples in batch
        for i in range(len(y_pred)):
            row_gradients = []
            # loop over labels in example           
            for j in range(len(y_pred[i])):
                if y_pred[i][j] > y_true[i][j]:
                    grad = Decimal('1')
                elif y_pred[i][j] == y_true[i][j]:
                    grad = Decimal('0')
                else:
                    grad = Decimal('-1')
                row_gradients.append(grad)
            gradients.append(row_gradients)
        return gradients
    
class BinaryCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        batch_size = len(y_pred)
        total_loss = Decimal(0)

        for i in range(batch_size):
            for j in range(len(y_pred[i])):
                total_loss -= y_true[i][j] * Decimal(math.log(y_pred[i][j])) + (1 - y_true[i][j]) * Decimal(math.log(1 - y_pred[i][j]))

        return total_loss / batch_size
    
    def backward(self, y_pred, y_true):
        batch_size = len(y_pred)
        gradients = []

        for i in range(batch_size):
            row_gradients = []
            for j in range(len(y_pred[i])):
                grad = (y_pred[i][j] - y_true[i][j]) / (y_pred[i][j] * (1 - y_pred[i][j]))
                row_gradients.append(grad)
            gradients.append(row_gradients)
        return gradients

class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        batch_size = len(y_pred)
        total_loss = Decimal(0)
        for i in range(batch_size):
            total_loss += sum([(y_pred[i][j] - y_true[i][j])**2 for j in range(len(y_pred[i]))])
        
        return total_loss / batch_size

    def backward(self, y_pred, y_true):
        gradients = [[2 * (y_pred[i][j] - y_true[i][j]) for j in range(len(y_pred[i]))] for i in range(len(y_pred))]
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
    # Reviewed fully. there are no issues.
    def __init__(self, params,lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params,lr)
        self.parameters = params
        self.beta1 = Decimal(beta1)
        self.beta2 = Decimal(beta2)
        self.epsilon = Decimal(str(epsilon))
        self.lr = Decimal(lr)
        self.m = create_zeros_structure(self.parameters)        
        self.v = create_zeros_structure(self.parameters)

        self.t = 0

    def step(self, input_gradients):         
        self.t += 1
        for b in range(len(self.parameters)):
            network = self.parameters[b]
            gradients = input_gradients[b]  
 
            for i in range(len(network)):
                grad_layer_weights = deepcopy(gradients[i][0])
                param_layer_weights = deepcopy(network[i][0])
 
                # For biases
                grad_layer_biases = deepcopy(gradients[i][1])
                param_layer_biases = deepcopy(network[i][1])

                for k in range(len(self.m[b][i][0])):                    
                    self.m[b][i][0][k] = self.beta1 * self.m[b][i][0][k] + (1 - self.beta1) * grad_layer_weights[k]                  
                    self.v[b][i][0][k] = self.beta2 * self.v[b][i][0][k] + (1 - self.beta2) * (grad_layer_weights[k] ** 2)

                    # Compute bias-corrected first and second moment estimates for weights
                    m_hat_weights = self.m[b][i][0][k] / (1 - self.beta1 ** self.t)
                    v_hat_weights = self.v[b][i][0][k] / (1 - self.beta2 ** self.t)

                    # Update weights
                    param_layer_weights[k] = param_layer_weights[k] - (self.lr * m_hat_weights) / ((v_hat_weights).sqrt() + self.epsilon)                  

                for k in range(len(grad_layer_biases)):
                    # Update first and second moment estimates for biases
                    self.m[b][i][1][k] = self.beta1 * self.m[b][i][1][k] + (1 - self.beta1) * grad_layer_biases[k]
                    self.v[b][i][1][k] = self.beta2 * self.v[b][i][1][k] + (1 - self.beta2) * (grad_layer_biases[k] ** 2)

                    # Compute bias-corrected first and second moment estimates for biases
                    m_hat_biases = self.m[b][i][1][k] / (1 - self.beta1 ** self.t)
                    v_hat_biases = self.v[b][i][1][k] / (1 - self.beta2 ** self.t)

                    # Update biases
                    param_layer_biases[k] = param_layer_biases[k] -  (self.lr * m_hat_biases) / ((v_hat_biases).sqrt() + self.epsilon)
                
                self.parameters[b][i][0] = deepcopy(param_layer_weights)
                self.parameters[b][i][1] = deepcopy(param_layer_biases)

class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.parameters = params
        self.lr = Decimal(lr)

    def step(self, input_gradients):
        for b in range(len(self.parameters)):
            network = self.parameters[b]
            gradients = input_gradients[b]
            for i in range(len(network)):
                grad_layer_weights = gradients[i][0]
                param_layer_weights = network[i][0]

                # For biases
                grad_layer_biases = gradients[i][1]
                param_layer_biases = network[i][1]

                for k in range(len(grad_layer_weights)):                
                    # Update weights
                    param_layer_weights[k][0] -= self.lr * grad_layer_weights[k][0]

                for k in range(len(grad_layer_biases)):
                    # Update biases
                    param_layer_biases[k][0] -= self.lr * grad_layer_biases[k][0]
                
                network[i][0] = param_layer_weights
                network[i][1] = param_layer_biases

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

def leaky_relu_2d(x, alpha=0.01):
    return [
        [max(Decimal(alpha) * element, element) for element in row]
        for row in x
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

def decimal_matrix_multiply(matrix1, matrix2):
    if len(matrix1) == 0 or len(matrix2) == 0:
        return []

    if isinstance(matrix1[0][0], list):  # Check if matrix1 is a 3D tensor
        result = [
            [
                [
                    sum(matrix1[b][i][k] * matrix2[b][k][j] for k in range(len(matrix2[b])))
                    for j in range(len(matrix2[b][0]))
                ]
                for i in range(len(matrix1[b]))
            ]
            for b in range(len(matrix1))
        ]
    else:
        rows1, cols1 = len(matrix1), len(matrix1[0])
        rows2, cols2 = len(matrix2), len(matrix2[0])

        if cols1 == 1 and rows1 == rows2:  # Check for the (2,1) by (2,1) case
            result = [[sum(matrix1[i][0] * matrix2[j][0] for i in range(rows1))] for j in range(rows2)]
        else:
            result = [
                [
                    sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2)))
                    for j in range(len(matrix2[0]))
                ]
                for i in range(len(matrix1))
            ]
    return result

def decimal_matrix_add(matrix1, matrix2):
    result = [[0 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] + matrix2[i][j]
    return result

def decimal_matrix_transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def transpose(matrix):
    if isinstance(matrix, list) and isinstance(matrix[0], Decimal):  # Handle the case when the input is a single list of decimals
        return [[matrix[i]] for i in range(len(matrix))]
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def custom_softmax(matrix):
    softmax_matrix = []
    for x in matrix:
        e_x = [Decimal(math.exp(xi)) for xi in x]
        sum_e_x = sum(e_x)
        softmax_row = [num / sum_e_x for num in e_x]
        softmax_matrix.append(softmax_row)
    return softmax_matrix

def flatten_nested_structure(structure):
    result = []

    def _flatten(struct):
        for item in struct:
            if isinstance(item, list):
                _flatten(item)
            else:
                result.append(item)

    _flatten(structure)
    return result

def tetration(base_list, height_list):
    if isinstance(base_list, Decimal) and isinstance(height_list, int):
        return _tetration_single(base_list, height_list)

    if isinstance(height_list, list):
        if len(base_list) != len(height_list):
            raise ValueError("Lists must have the same shape")
        data = []
        for base, height in zip(base_list, height_list):
            data_example = []
            for b, h in zip(base, height):
                data_example.append(_tetration_single(b, h))
            data.append(data_example)
        return data
    else:
        data = []
        for base in base_list:
            data.append([_tetration_single(b, height_list) for b in base])
        return data

def _tetration_single(base, height):
    int_height = int(height)  # Cast height to integer
    if int_height <= 0:
        return Decimal(1)
    result = base
    for _ in range(int_height - 1):
        result = base ** result
    return result

def sigmoid(x):
    if isinstance(x, list):
        return [sigmoid(item) for item in x]
    return 1 / (1 + Decimal(math.exp(-x)))

def sin_decimal(x):
    if isinstance(x, list):
        return [sin_decimal(elem) for elem in x]
    else:
        x = Decimal(x) % (2 * Decimal(math.pi))
        n, term, total = 0, x, x

        while abs(term) > Decimal(1e-28):
            n += 1
            term *= -x * x / ((2 * n) * (2 * n + 1))
            total += term

        return total

def log_decimal(x, base=Decimal(math.e)):
    if x <= 0:
        raise ValueError("logarithm is undefined for non-positive values")
    
    # Use natural logarithm for the base e (default)
    if base == Decimal(math.e):
        n, term, total = 0, (x - 1), (x - 1)

        while abs(term) > Decimal(1e-28):
            n += 1
            term *= - (x - 1) / n
            total += term

        return total

    # Calculate logarithm for other bases using the change of base formula
    else:
        return log_decimal(x) / log_decimal(base)

    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2

    if n % 2 == 0:  # Even number of elements
        median = (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:  # Odd number of elements
        median = sorted_numbers[mid]

    return median
