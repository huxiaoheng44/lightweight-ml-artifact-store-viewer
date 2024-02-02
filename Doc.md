
# LoggerAndViewer Documentation
- [DataManager Module](#datamanager-module)
  - [Overview](#overview)
  - [Features](#features)
  - [Core Functions](#core-functions)
  - [Usage Example](#usage-example)
- [Logger Module](#logger-module)
  - [Overview](#overview-1)
  - [Features](#features-1)
  - [Core Functions](#core-functions-1)
  - [Usage Example](#usage-example-1)
- [HandleStorage Module](#handlestorage-module)
  - [Overview](#overview-2)
  - [Features](#features-2)
  - [Core Functions](#core-functions-2)
  - [Usage Example](#usage-example-2)
- [VisualBackend Module](#visualbackend-module)
  - [Overview](#overview-3)
  - [Features](#features-3)
  - [Core Endpoints](#core-endpoints)
  - [Core Functions](#core-functions-3)
  - [Usage Example](#usage-example-3)
- [Example Usage](#example-usage)


# DataManager Module 

## Overview
The `DataManager` module is designed for persistent storage and operations on artifacts within a MongoDB database. It provides functionality for connecting to the database, adding, retrieving, and searching for artifacts, as well as specific metadata operations. This module plays a crucial role in the LoggerAndViewer system, facilitating the management of model training artifacts.

## Features

- **Database Connection**: Establishes a connection to a MongoDB database using a provided URI.
- **Artifact Retrieval**: Allows retrieval of all artifacts stored in the database.
- **Artifact Search**: Supports searching for artifacts based on specific queries.
- **Metadata Search**: Enables searching for metadata keys using regular expressions.
- **Data Addition**: Facilitates the addition of data to any specified collection within the database.
- **File Handling**: Provides the ability to add or update file records in the database.
- **Default Value Generation**: Generates default values for various data types, aiding in artifact record creation.
- **Artifact Reading and Template Generation**: Supports reading artifact data from files and generating artifact templates with default values.

## Core Functions

### `__init__(self, uri)`
Constructor that initializes the database connection using the provided MongoDB URI. It automatically connects to the specified database and sets up the collection for metadata storage.

### `getAllArtifacts(self)`
Retrieves all artifacts from the `artifacts` collection in the database and prints the count of retrieved artifacts.

### `searchArtifacts(self, query)`
Searches for artifacts in the `artifacts` collection based on a provided query dictionary.

### `searchMetadataByKey(self, key_pattern)`
Searches for metadata within artifacts using a key pattern, utilizing regular expressions for flexible matching.

### `addData(self, collection_name, data)`
Adds a new document to the specified collection within the database and returns the inserted document's ID.

### `addFile(self, file_json)`
Adds or updates a file record in the `files` collection based on the provided JSON object representing the file.

### `generate_default_value(self, type_str)`
Generates default values for a specified type, supporting basic data types like float, integer, string, and complex types like list, dict, set, etc.

### `readArtifactsFromFile(self, file_path)`
Reads artifact data from a specified file path and initializes an artifact record with default values based on the file's content.

### `getArtifactTemplate(self)`
Returns a copy of the current artifact template, which can be used to instantiate new artifact records.

## Usage Example

```python
# Initialize DataManager with the default MongoDB URI
data_manager = DataManager()

# Retrieve all artifacts
artifacts = data_manager.getAllArtifacts()

# Search for specific artifacts
search_results = data_manager.searchArtifacts({"type": "model"})

# Add new artifact data
new_artifact_id = data_manager.addData("artifacts", {"name": "New Model", "type": "model"})

# Generate a default artifact template
artifact_template = data_manager.getArtifactTemplate()

```


# Logger Module 

## Overview
The `Logger` module is integral to the LoggerAndViewer system, providing comprehensive logging capabilities for model training processes. It facilitates the recording of artifacts, including scalar values, images, and metadata, into both local storage and a MongoDB database via the `DataManager` module. Additionally, it supports visualization through TensorBoard, enhancing the analysis and monitoring of model training.

## Features

- **TensorBoard Integration**: Utilizes `SummaryWriter` from `torch.utils.tensorboard` for logging scalar values and images, allowing for real-time monitoring and visualization in TensorBoard.
- **Artifact Logging**: Records various types of training artifacts, including scalar metrics and images, into a structured format suitable for analysis and review.
- **CSV Logging**: Creates and appends to CSV files for detailed record-keeping of artifacts.
- **Database Interaction**: Leverages the `DataManager` module for storing artifacts in a MongoDB database, enabling persistent storage and later retrieval.
- **Image Logging**: Supports logging of image artifacts, including preprocessing for different image formats and storage both locally and in the database.
- **Log Management**: Maintains a directory structure for organized storage of logs and provides functionalities for closing the logger and retrieving log types.

## Core Functions

### `__init__(self, log_dir="")`
Initializes the logger with a specific directory for storing logs and sets up the `SummaryWriter` for TensorBoard integration.

### `logArtifact(self, artifact_record)`
Logs a given artifact record, which can include scalar metrics, to both TensorBoard and a MongoDB database. It also generates a CSV file for the artifact, organizing data based on the model and experiment.

### `logArtifactWithImage(self, artifact_record, image)`
Extends `logArtifact` by adding the capability to log image artifacts. It supports image inputs as either PyTorch tensors or NumPy arrays, handling necessary conversions and saving the image locally and its metadata in the database.

### `_record_log_type(self, log_type)`
Privately used to record and categorize the type of logs being captured, organizing them within the specified log directory.

### `getLogTypes(self)`
Returns a dictionary of log types and their corresponding paths, facilitating easier retrieval and organization of logged data.

### `close(self)`
Closes the `SummaryWriter`, ensuring that all pending logs are flushed to disk and properly finalized.

## Usage Example

```python
# Initialize Logger with a designated log directory
logger = Logger(log_dir="my_experiment_logs")

# Log scalar artifacts
artifact_record = {
    "ModelName": "MyModel",
    "ExperimentName": "Experiment1",
    "Accuracy": 0.95,
    "Loss": 0.05,
    "iteration": 100
}
logger.logArtifact(artifact_record)

# Log image artifacts
# Assuming images_tensor is a batch of images or a single image tensor
logger.logArtifactWithImage(artifact_record, images)


# Close the logger when done
logger.close()
```



# HandleStorage Module 

## Overview
The `handleStorage.py` module provides a set of functions to interact with Amazon Web Services (AWS) S3 storage. It includes capabilities to configure AWS sessions, upload and download files to/from S3 buckets, validate AWS credentials, and list files and buckets in S3. This module simplifies the process of integrating AWS S3 operations into Python applications.

## Features

- **AWS Session Configuration**: Configures the AWS session with specific credentials and region.
- **File Upload to S3**: Uploads files to specified S3 buckets.
- **File Download from S3**: Downloads files from specified S3 buckets.
- **AWS Credentials Validation**: Validates the configured AWS IAM credentials.
- **Check AWS Credentials Configuration**: Checks if AWS credentials are properly configured.
- **List Files in S3 Bucket**: Retrieves a list of files from a specified S3 bucket.
- **List S3 Buckets**: Lists all S3 buckets available to the configured AWS account.
- **List Files in Bucket**: Lists all files in a specified S3 bucket.

## Core Functions

### `configure_aws_session(aws_access_key_id, aws_secret_access_key, aws_session_token=None, region_name='us-east-1')`
Configures the AWS session using provided access key, secret access key, optional session token, and region name.

### `upload_to_s3(file_name, bucket, object_name=None)`
Uploads a file to an S3 bucket. If `object_name` is not specified, `file_name` is used as the object name.

### `download_from_s3(bucket, object_name, file_name=None)`
Downloads a file from an S3 bucket. If `file_name` is not provided, `object_name` is used as the file name for the downloaded content.

### `validate_aws_credentials(bucket)`
Validates AWS IAM credentials by attempting to list objects in the specified bucket. Returns `True` if credentials are valid and the bucket is accessible.

### `aws_credentials_configured()`
Checks if AWS credentials are configured by attempting to retrieve the AWS caller identity.

### `get_files_list_from_s3(bucket_name)`
Retrieves the list of files from the specified S3 bucket and returns a list of file names.

### `list_s3_buckets()`
Lists all S3 buckets available to the configured AWS account and returns a list of bucket names.

### `list_files_in_bucket(bucket_name)`
Lists all files in a specified S3 bucket and returns a list of object keys (file names).

## Usage Example

```python
# Configure AWS session
configure_aws_session('your_access_key_id', 'your_secret_access_key', region_name='your_region_name')

# Upload a file to S3
upload_success = upload_to_s3('path/to/your/file.txt', 'your_bucket_name')

# Download a file from S3
download_success = download_from_s3('your_bucket_name', 'your_object_name', 'path/to/save/file.txt')

# Validate AWS credentials
credentials_valid = validate_aws_credentials('your_bucket_name')

# List files in an S3 bucket
files_list = get_files_list_from_s3('your_bucket_name')

# List all S3 buckets
buckets_list = list_s3_buckets()

# List files in a specific S3 bucket
files_in_bucket = list_files_in_bucket('your_bucket_name')
```

# VisualBackend Module 

## Overview
The `VisualBackend` module establishes a Flask-based web server that serves as the backend for a visualization and management interface for model training artifacts. It integrates functionalities such as artifact retrieval, search, and visualization via generated charts, alongside initiating TensorBoard for further data analysis.

## Features

- **Flask Web Server**: Implements a web server that provides a user interface for artifact visualization and management.
- **Artifact Management**: Interfaces with the `DataManager` module to fetch, search, and group artifacts stored in a MongoDB database.
- **Dynamic Chart Generation**: Dynamically generates numeric and image charts based on the artifacts' data, aiding in the analysis of model performance.
- **TensorBoard Integration**: Offers functionality to start TensorBoard directly from the interface, facilitating detailed analysis of logs.
- **Artifact Search and Grouping**: Allows users to search for artifacts by various fields and group them for comparison.

## Core Endpoints

### `/` (Index Route)
- **Method**: `GET`, `POST`
- **Functionality**: Serves the main page of the web interface, displaying grouped artifacts based on the specified criteria. Supports searching and grouping of artifacts dynamically.

### `/start-tensorboard`
- **Method**: `GET`
- **Functionality**: Initiates a TensorBoard instance, making it accessible on a specified port (default: 6006) for log visualization.

### `/generate-charts`
- **Method**: `POST`
- **Functionality**: Receives a list of file paths to CSV files containing artifact data, generates combined charts for numeric data and individual images for visualization, and returns the paths to these generated charts.

## Core Functions

### `__init__(self)`
Initializes the Flask app, configures logging, and sets up routes for the web server. It also initializes a `DataManager` instance for database operations.

### `generate_combined_charts(self, csv_paths, static_dir='static/visualizationImages')`
Generates both numeric and image charts from a given list of CSV file paths. This involves reading the CSV files, extracting numeric data and image information, and plotting this data using matplotlib. The function returns a list of paths to the generated chart images.

### `generate_numeric_charts(self, numeric_data, iteration_columns, static_dir)`
Takes numeric data extracted from CSV files and generates line charts for each metric across iterations, saving the charts to the specified static directory.

### `generate_image_charts(self, image_data, static_dir)`
Generates and saves charts consisting of training images, facilitating visual comparison of model outputs or inputs over time.

### `launch(self)`
Starts the Flask web server in debug mode, making the web interface accessible for interaction.

### `groupArtifactsBy(self, field, artifacts)`
Groups artifacts by a specified field, such as model name or experiment name, to organize data for display on the web interface.

## Usage Example

To start the web server and make the visualization interface available, instantiate the `VisualBackend` class and call the `launch` method:

```python
if __name__ == '__main__':
    visual_backend = VisualBackend()
    visual_backend.launch()
```


# Example Usage
```python
# Import necessary modules from LoggerAndViewer package
from LoggerAndViewer.dataset.DataManager import DataManager
from LoggerAndViewer.log.log import Logger
from LoggerAndViewer.visualization.backend import VisualBackend
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define your model here - example using an Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x

# Function to generate synthetic data
def generate_data(num_samples, input_dim):
    data = np.random.rand(num_samples, input_dim)
    return torch.tensor(data, dtype=torch.float32)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs, logger, data_manager, experiment_name, model_name):
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        # After each epoch, log the necessary artifacts
        artifact_record = data_manager.getArtifactTemplate()
        artifact_record.update({
            'MSE': loss.item(),
            'ModelName': model_name,
            'ExperimentName': experiment_name,
            'TrainingIteration': epoch + 1,
            # Add additional metadata as needed
        })
        logger.logArtifact(artifact_record)

# Main script
if __name__ == "__main__":
    # Initialize the data manager and read artifact template
    data_manager = DataManager()
    data_manager.readArtifactsFromFile("./Artifacts.json")

    # Initialize logger with directory, experiment, and model names
    logger = Logger(log_dir='autoencoder_experiment')
    experiment_name = "AutoencoderExperiment"
    model_name = "AutoencoderModel"

    # Define hyperparameters and model
    input_dim = 20
    hidden_dim = 10
    num_samples = 100
    batch_size = 10
    learning_rate = 0.01
    num_epochs = 5

    model = Autoencoder(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Generate synthetic data and create DataLoader
    dataset = generate_data(num_samples, input_dim)
    dataloader = DataLoader(dataset=TensorDataset(dataset, dataset), batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs, logger, data_manager, experiment_name, model_name)

    # Close the logger after training is complete
    logger.close()

    # Launch the visualization backend to view the logged artifacts
    visualizer = VisualBackend()
    visualizer.launch()

```