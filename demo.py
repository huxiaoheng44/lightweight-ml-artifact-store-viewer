import torch
import torch.nn as nn
import torch.optim as optim
from LoggerAndViewer.dataset.DataManager import DataManager
from torch.utils.data import DataLoader, TensorDataset
from LoggerAndViewer.log.log import Logger
from LoggerAndViewer.visualization.backend import VisualBackend
import numpy as np
from torch.nn.functional import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

# MODEL A
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x
# MODEL B
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Generate some synthetic data for training and testing
def generate_data(num_samples, input_dim):
    data = np.random.rand(num_samples, input_dim)
    return torch.tensor(data, dtype=torch.float32)

def calculate_completeness(data, output):
    # Assumes data and output are torch tensors of the same shape
    # Calculating mean cosine similarity across the batch
    cos_sim = cosine_similarity(data.view(data.shape[0], -1), output.view(output.shape[0], -1), dim=1)
    return cos_sim.mean().item()



def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0]
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def train_model(model, dataloader, criterion, optimizer, num_epochs, logger, data_manager, experiment_name="", model_name="",meta_data={}):
    for epoch in range(num_epochs):
        total_loss = 0
        total_completeness = 0
        for batch in dataloader:
            data = batch[0]
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_completeness += calculate_completeness(data, output)

        average_loss = total_loss / len(dataloader)
        average_completeness = total_completeness / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Completeness: {average_completeness:.4f}')

        artifact_record = data_manager.getArtifactTemplate()

        artifact_record.update({
            'MSE': average_loss,
            'ModelName': model_name,
            'Completeness': average_completeness,
            'ExperimentName': experiment_name,
            'TrainingIteration': epoch + 1,
            'Metadata': meta_data,
            'loss': average_loss,
        })
        logger.logArtifact(artifact_record)


def train_and_evaluate(model_class, hyperparams, logger, data_manager, experiment_name="", model_name="",meta_data={}):
    model = model_class(hyperparams['input_dim'], hyperparams['hidden_dim'])
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.MSELoss()

    # Generate synthetic data
    dataset = generate_data(hyperparams['num_samples'], hyperparams['input_dim'])
    dataloader = DataLoader(dataset=TensorDataset(dataset), batch_size=hyperparams['batch_size'], shuffle=True)

    train_model(model, dataloader, criterion, optimizer, hyperparams['num_epochs'], logger, data_manager,
                experiment_name, model_name, meta_data)
    mse_loss = evaluate_model(model, dataloader, criterion)
    return mse_loss


def main():
    hyperparams = {
        'input_dim': 20,
        'hidden_dim': 10,
        'num_samples': 100,
        'batch_size': 10,
        'learning_rate': 0.01,
        'num_epochs': 10
    }

    # Define meta data manager
    data_manager = DataManager()
    # Read artifacts from file and generate an artifact template
    data_manager.readArtifactsFromFile("./Artifacts.json")

    # Train and evaluate Autoencoder
    meta_data = {"CODE": "1234", "release-date": "2020-01-01"}
    logger_autoencoder = Logger(log_dir='autoencoder-2')
    experiment_name_autoencoder = "AutoencoderExperiment-2"
    model_name_autoencoder = "Autoencoder-2"
    mse_loss_autoencoder = train_and_evaluate(Autoencoder, hyperparams, logger_autoencoder, data_manager, experiment_name_autoencoder, model_name_autoencoder,meta_data)
    logger_autoencoder.close()

    # Train and evaluate MLP
    meta_data = {"CODE": "4321", "release-date": "2020-02-02"}
    logger_mlp = Logger(log_dir='mlp-2')
    experiment_name_mlp = "MLPExperiment-2"
    model_name_mlp = "MLP-2"
    mse_loss_mlp = train_and_evaluate(MLP, hyperparams, logger_mlp, data_manager, experiment_name_mlp, model_name_mlp, meta_data)
    logger_mlp.close()
# Main
if __name__ == "__main__":

    # main()
    visualizer = VisualBackend()
    visualizer.launch()






