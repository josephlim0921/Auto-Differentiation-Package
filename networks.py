import numpy as np

class Network():
    def __init__(self, input_size):
        self.neuron_sizes = [input_size] #need to pass input size when creating the network everything else can be defined later
        self.weights = []
        self.biases = []
        self.nonlinearities = []
        self.z_list = []
        self.nonlinearity_grads = []
        self.weights_grads = []
        self.biases_grads = []

    def modify_parameters(self, weights, biases):
        self.weights = weights #In order to have the same weights for assignment comparison we add a manual function to ovveride initilization 
        self.biases = biases
        return

    def add_layer(self, neuron_size, nonlinearity):
        self.weights = [] #Weights will need to find a new shape as layers are added, existing weights have minimal value when adding new layers so re-initilizing the weights is done here to reduce numerical issues
        self.biases = []
        self.neuron_sizes.append(neuron_size)
        self.nonlinearities.append(nonlinearity)
        for i in range(len(self.neuron_sizes) - 1):
            if self.nonlinearities[i] == "relu":
                # He initialization
                normal_std = np.sqrt(2 / self.neuron_sizes[i])
                self.weights.append(np.random.normal(0, normal_std, (self.neuron_sizes[i], self.neuron_sizes[i+1])))
            else:
                # Glorot initialization
                uniform_bound = np.sqrt(6 / (self.neuron_sizes[i] + self.neuron_sizes[i + 1]))
                self.weights.append(np.random.uniform(-uniform_bound, uniform_bound, (self.neuron_sizes[i], self.neuron_sizes[i+1])))
            self.biases.append(np.zeros(self.neuron_sizes[i + 1])) #Making an empty set of biases to initilize the network, ensuring the shape is consistant

    def forward(self, x):
        self.z_list.append(x) # Storing the input
        for i in range(len(self.neuron_sizes) - 1): #Looping through the network forwards
            x = np.dot(x, self.weights[i]) + self.biases[i] #Calculating the next layer inputs as vectors to save on computation [np.dot is used here as while it is not what is reccomended by the numpy documentation, it is capable of passing vectors to matmul and handeling scalars]
            if len(x.shape) == 1:
                x = x.reshape(-1, 1) #Handling scalars before applying non-linearities elementwise
            if self.nonlinearities[i] == "relu":
                x = np.maximum(x, 0)
                self.nonlinearity_grads.append(x>0) #Storing the gradients of the non-linearity 
            elif self.nonlinearities[i] == "sigmoid":
                x = 1 / (1 + np.exp(-x))
                self.nonlinearity_grads.append(x * (1 - x)) #Storing the gradients of the non-linearity 
            elif self.nonlinearities[i] == "linear":
                self.nonlinearity_grads.append(np.ones_like(x)) #Storing the gradients of the non-linearity 
            self.z_list.append(x) #Storing the activations for future gradient calculation
        self.y_pred = x #x here is a vector allowing for multi-dimensional outputs
        return x

    def backward(self, y_true):
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1) #Handling scalars
        if len(self.y_pred.shape) == 1:
            self.y_pred = self.y_pred.reshape(-1, 1) #Handling scalars
        dirac = self.y_pred - y_true #As per lecture notes for the loss functions we will be handling the loss dirac is yhat-y
        self.biases_grads.insert(0, np.mean(dirac, axis=0)) #Inserting into the list in reverse to handle the backwards nature, averaging over the runs when updating the gradient as per instructions
        self.weights_grads.insert(0, np.dot(dirac.T, self.z_list[-2]).T / len(y_true)) #The inner product containing the runs dimension acts as a sum of the gradients and allows for simple averaging
        for i in range(len(self.weights) - 1): #Repeating the above for all future layers 
            layer_index = len(self.weights) - i - 2
            dirac = np.multiply(self.nonlinearity_grads[layer_index], np.dot(dirac, self.weights[layer_index+1].T)) #Using elementwise multiplication to translate the non-linearities to the rest of the calculations
            self.biases_grads.insert(0, np.mean(dirac, axis=0))
            self.weights_grads.insert(0, np.dot(dirac.T, self.z_list[layer_index]).T / len(y_true))
        return

    def zero_grads(self):
        self.weights_grads = [] #Setting gradient and tools for gradient to zero
        self.biases_grads = []
        self.z_list = []
        self.nonlinearity_grads = []
        return

    def update_parameters(self, lr):
        for w, b, dw, db in zip(self.weights, self.biases, self.weights_grads, self.biases_grads):
            w -= lr * dw #Subtracting learning rate times graident from weights and biases
            b -= lr * db
        return
    
def regression_loss(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1) #Handling scalars
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1) #Handling scalars
    return 0.5*np.sum((y_true - y_pred) ** 2) #The loss for regression tasks, which is the sum of squared residuals with a 0.5 factor

def classification_loss(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1) #Handling scalars
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1) #Handling scalars

    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #Limiting the value of y_pred within a specific range. This is to tackle the problem of log(0) being negative infinity
    
    loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) #Binary Cross Entropy Loss
    return -np.sum(loss)