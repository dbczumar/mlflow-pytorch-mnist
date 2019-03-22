import argparse

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env

import cloudpickle

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Train an RNN model for MNIST classification in PyTorch')
parser.add_argument('--num-hidden-layers', '-l', type=int, default=2)
parser.add_argument('--batch-size', '-b', type=int, default=100)
parser.add_argument('--epochs', '-e', type=int, default=4)
parser.add_argument('--learning-rate', '-r', type=float, default=0.01)
parser.add_argument('--checkpoint-path', '-c', type=str)

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_classes = 10
num_layers = args.num_hidden_layers
batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.learning_rate

mlflow.log_param("num_hidden_layers", num_layers)
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", num_epochs)
mlflow.log_param("learning_rate", learning_rate)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./mnist/',
                                          train=False,
					  download=True,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # Softmax
        out = self.softmax(out)
        return out

def train_model(model):
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("neg_log_loss", loss.item())

            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
if args.checkpoint_path:
    model.load_state_dict(torch.load(args.checkpoint_path))
else:
    train_model(model)
    torch.save(model.state_dict(), "model.ckpt")

mlflow.pytorch.log_model(pytorch_model=model, artifact_path="torch-rnn-model")

conda_env = _mlflow_conda_env(
    path="conda.yaml", 
    additional_conda_channels=[
        "pytorch",
    ],
    additional_conda_deps=[
        "pytorch={}".format(torch.__version__),
        "torchvision={}".format(torchvision.__version__),
    ],
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__)
    ])

class MnistTorchRNN(PythonModel):

    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(context.artifacts["torch-rnn-model"])
        self.model.to('cpu')
        self.model.eval()

    def predict(self, context, input_df):
        import numpy as np
        with torch.no_grad():
            input_tensor = torch.from_numpy(
                input_df.values.reshape(-1, 28, 28).astype(np.float32)).to('cpu')
            return self.model(input_tensor).numpy()

mlflow.pyfunc.log_model(
    artifact_path="pyfunc-rnn",
    artifacts={
        "torch-rnn-model": mlflow.get_artifact_uri("torch-rnn-model")
    },
    python_model=MnistTorchRNN(),
    conda_env=conda_env)

print(mlflow.active_run().info.run_uuid)
