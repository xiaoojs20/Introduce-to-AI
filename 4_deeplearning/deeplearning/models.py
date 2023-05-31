import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import backend

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DigitClassificationModel(nn.Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """
    def __init__(self):
        super(DigitClassificationModel, self).__init__()
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        """
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64*5*5, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a PyTorch tensor with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a PyTorch tensor with shape (batch_size x 1 x 28 x 28)
        Output:
            A PyTorch tensor with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten the feature map

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def train_model(self, data_train, data_val=None):
        """
        Trains the model.
        You should define the loss (in torch.nn) and optimizer (in torch.optim) here.
        The train dataset and the validation dataset is defined as follows:
        
        transform = T.Compose([
               T.ToTensor(),
               T.Normalize((0.1307,), (0.3081,))
            ])
        data_train = dset.MNIST('./data/mnist', train=True, download=True,
                           transform=transform)
        data_val = None
        """
        # You may modify the attributes of DataLoader here (e.g. modify the batch_size).
        loader_train = DataLoader(data_train, batch_size=64, shuffle=True)
        self.train()
        # Define proper optimizers: suggest to use SGD with learning rate 0.05, monemtum 0.9, weight decay 5e-4.
        "*** YOUR CODE HERE ***"
        """
        torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False)
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=self.parameters(), lr=0.02, momentum=0.92)
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(loader_train):
                data, target = data.to(device), target.to(device) # recommend to add this line of code
                optimizer.zero_grad()
                output = self.forward(data)
                # Define cross entropy loss here.
                "*** YOUR CODE HERE ***"
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                accuracy = torch.mean((torch.max(output, dim=1)[1] == target).float())
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.2f}%'.format(
                        epoch, batch_idx, len(loader_train),
                        100. * batch_idx / len(loader_train), loss.item(), 100.*accuracy.item()))
         

class RegressionModel(nn.Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        n_feature = 1
        n_hidden = 20
        n_output = 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: PyTorch tensor with shape (batch_size x 1)
        Returns:
            PyTorch tensor with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        return self.net(x)

    def train_model(self, data_train, data_val=None):
        """
        Trains the model.
        You may define the loss (in torch.nn) and optimizer (in torch.optim) here.
        The train dataset and the validation dataset is defined as follows:
        
        x = torch.linspace(-2 * np.pi, 2 * np.pi, 2048).view(-1, 1) # shape (2048, 1)
        y = torch.sin(x) # shape (2048, 1)
        data_train = TensorDataset(x, y)
        x = torch.linspace(-2 * np.pi, 2 * np.pi, 200).view(-1, 1)
        y = torch.sin(x)
        data_val = TensorDataset(x, y)
        """
        # DataLoader should be created with the provided dataset. 
        # You can modify the DataLoader attributes here (e.g. batch_size)
        loader_train = DataLoader(data_train, batch_size=128, shuffle=True)
        loader_val = DataLoader(data_val, batch_size=128, shuffle=False)
        
        # Train with the DataLoader
        "*** YOUR CODE HERE ***"
        epochs = 100
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        
        for epoch in range(epochs):
            # train
            self.train()
            for batch_idx, (data, target) in enumerate(loader_train):
                data, target = data.to(device), target.to(device) # recommend to add this line of code
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validation
            if loader_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_x, val_y in loader_val:
                        val_output = self.forward(val_x)
                        val_loss += criterion(val_output, val_y)
                    val_loss /= len(loader_val)
                print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        print("Training complete.")                      
                        
class DigitAttackModel(object):
    """
    A model for attacking a handwritten digit classification model.

    Each handwritten digit is a 28x28 pixel grayscale image. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to genrate adversarial examples of a given handwritten digit.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.model = LeNet()
        self.model.load_state_dict(torch.load('./data/mnist_lenet.pt'))
        self.model = self.model.to(device)
        self.model.eval()

    def attack(self, x, target, epsilon):
        """
        Perfroming adversarial attacks with FGSM.
        
        The adversarial examples should be small perturbation of the original data x,
        but the predictions should not be the target label.
        You may use loss.backward() to compute the gradients, and use x.grad to obtain
        the gradients regrading to x.

        Inputs:
            x: a PyTorch tensor with shape (batch_size x 1 x 28 x 28)
            the elements of x must be in the interval [0,1].
            target: a PyTorch tensor with shape (batchsize,). The label of x ranges 0-9.
            epsilon: max perturbation on each pixel.
        Output:
            A PyTorch tensor with shape (batch_size x 1 x 28 x 28) which is the 
            adversarial examples of x. the elements of the adversarial examples must 
            be in the inverval [0,1].
        """
        self.model.eval()
        x = x.clone().requires_grad_() 
        x_input = (x - 0.1307) / 0.3081 # Normalize data
        # generate the adversarial examples and store them in x_adv
        "*** YOUR CODE HERE ***"
        criterion = nn.CrossEntropyLoss()
        output = self.model(x_input)
        self.model.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        sign_data_grad = x.grad.sign()
        x_adv = x + epsilon * sign_data_grad
        return torch.clamp(x_adv, 0, 1)


class LanguageIDModel(nn.Module):
    """
    A model for language identification at a single-word granularity.

    (You may use nn.RNN or nn.GRU or nn.LSTM here in this problem. please refer to the 
    official documentation for more details.
    We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        super(LanguageIDModel, self).__init__()
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 128
        self.gru = nn.GRU(input_size=self.num_chars, hidden_size=self.hidden_size, num_layers=5)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=len(self.languages))

    def forward(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be PyTroch tensor with (L, batch_size, self.num_chars), 
        where every row in the last axis is a one-hot vector encoding of a character. 
        For example, if we have a batch of 8 three-letter words where the last word is 
        "cat", then xs[1,6?7] will be a one-hot vector that contains a 1 at position 0. 
        Here the index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the 
        `xs` into a PyTorch tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: PyTroch tensor with (L, batch_size, self.num_chars)
        Returns:
            A PyTorch tensor shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        L = xs.size(0)
        batch_size = xs.size(1)
        
        # Pack the input sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(xs, lengths=[L] * batch_size, enforce_sorted=False)

        # Forward pass through GRU layer
        packed_output, hidden = self.gru(packed_input)

        # Unpack the output sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # Get the last output of the GRU for each sequence
        last_hidden = hidden[-1]

        # Apply linear layer
        logits = self.fc(last_hidden)

        output, hidden = self.gru(xs)
        
        # print(output.size())
        # print(hidden.size())
        # print('\n')
        # Get the last output of the LSTM for each sequence
        last_hidden = hidden[-1, :, :]

        # Apply linear layer
        logits = self.fc(last_hidden)
        # print(last_hidden.size())
        # logits = logits.reshape(batch_size, -1)
        return logits
        L = xs.size(0)
        batch_size = xs.size(1)
        hidden0 = torch.zeros(L,batch_size,self.hidden_size)
        self.embed = nn.Embedding(num_embeddings=batch_size, embedding_dim=self.num_chars)
        xs=torch.LongTensor(xs)
        embeded = self.embed(xs)
        output, hidden = self.gru(embeded, hidden0)
        print(output.size())
        # output = output[-1,:,:]
        print(hidden.size())
        logits = self.fc(hidden[-1])
        # print(logits.size())
        # print('\n')
        return logits


    def train_model(self, loader_train, loader_val):
        """
        Trains the model.
        
        The train loader and the validation loader are provided.
        """
        "*** YOUR CODE HERE ***"
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        num_epochs = 10
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for xs, ys in loader_train:
                optimizer.zero_grad()
                output = self.forward(xs)
                loss = criterion(output, ys)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader_train)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xs, ys in loader_val:
                    output = self.forward(xs)
                    _, predicted = torch.max(output.data, 1)
                    total += ys.size(0)
                    correct += (predicted == ys).sum().item()
            
            accuracy = correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")
        print("Training complete.")           

class DeepQModel(nn.Module):
    """
    Deep Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """

    def __init__(self):
        super(DeepQModel, self).__init__()
        self.data_loader = backend.CartPoleLoader(self)

        self.num_actions = 2
        self.state_size = 4

        # Initialize the model parameters and the optimizer here.
        "*** YOUR CODE HERE ***"
        self.optimizer = None

    def forward(self, states, Q_target=None):
        """
        TODO: Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the tensor of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a PyTorch scalar
        which computes the training loss between your current Q-value
        predictions and these target values, using mse loss.

        Inputs:
            states: a (batch_size x 4) PyTorch tensor
            Q_target: a (batch_size x 2) PyTorch tensor, or None
        Output:
            (if Q_target is not None) The loss for optimizing the network
            (if Q_target is None) A (batch_size x 2) PyTorch tensor of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
        else:
            "*** YOUR CODE HERE ***"

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) PyTorch tensor or numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        if np.random.rand() < eps:
            return int(np.random.choice(self.num_actions))
        else:
            scores = self.forward(state)
            return torch.argmax(scores).item()
            
    def train_model(self):
        for x, y in self.data_loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            self.optimizer.step()