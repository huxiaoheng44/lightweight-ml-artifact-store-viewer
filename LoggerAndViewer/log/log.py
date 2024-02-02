from ..dataset.DataManager import DataManager
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class Logger:
    def __init__(self, log_dir=""):
        log_dir = os.path.join('logs', log_dir)
        self.writer = SummaryWriter(log_dir)
        self.data_manager = DataManager()
        self.log_types = {}  # Dictionary to store log types and their paths
        self.log_dir = log_dir

    def logArtifact(self, artifact_record):
        collection_name = "artifacts"
        iteration_key = None
        iteration_value = None

        # Find the iteration key and value
        for key in artifact_record.keys():
            if 'iteration' in key.lower():
                iteration_key = key
                iteration_value = artifact_record[key]
                break

        model_name = artifact_record.get('ModelName', 'undefined')

        # Define headers for the CSV file
        headers = [key for key in artifact_record if key not in ['FileList', 'Metadata']]

        # Create or append to a CSV file specific to the experiment and model
        experiment_name = artifact_record.get('ExperimentName', 'default')
        artifact_id = str(artifact_record.get('ArtifactID', ''))[-5:]  # Convert to string and get the last 5 characters
        csv_filename = f"{experiment_name}_{artifact_id}.csv"
        csv_dir = os.path.join(self.log_dir, "data")
        os.makedirs(csv_dir, exist_ok=True)  # Create the directory if it doesn't exist
        csv_path = os.path.join(csv_dir, csv_filename)

        # Check if CSV file exists, if not create it with headers
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(headers)

            # Write the data row
            row = [artifact_record.get(key) for key in headers]
            writer.writerow(row)


        # Log the contents of artifact_record to TensorBoard
        for key, value in artifact_record.items():
            if key != iteration_key and isinstance(value, (int, float)):
                tag = f"{model_name}/{key}" if model_name else key
                if iteration_value is not None:
                    self.writer.add_scalar(tag, value, iteration_value)
                else:
                    self.writer.add_scalar(tag, value)

        # Add the CSV file path to artifact_record's FileList
        artifact_record['FileList'] = [csv_path]

        # Check if Metadata field exists and is a dictionary, then convert it
        if 'Metadata' in artifact_record and isinstance(artifact_record['Metadata'], dict):
            metadata_dict = artifact_record['Metadata']
            metadata_list = [{k: v} for k, v in metadata_dict.items()]
            artifact_record['Metadata'] = metadata_list

        # Add artifact_record to the database
        self.data_manager.addData(collection_name, artifact_record)

    def logArtifactWithImage(self, artifact_record, image):
        # Check if image is a batch of images and select the first one
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:  # Batch of images
                image = image[0]  # Select the first image from the batch
            elif image.ndim != 3 or image.shape[0] not in [1, 3, 4]:
                raise ValueError("The tensor must have 3 dimensions with C in [1, 3, 4].")

        # Check if image is a NumPy array
        if isinstance(image, np.ndarray):
            # Check if the image has 3 dimensions (H, W, C)
            if image.ndim == 3:
                # Check if the channels are in the last dimension
                if image.shape[2] in [1, 3, 4]:
                    # Convert numpy array to PyTorch Tensor, adjust channel order if needed
                    image = torch.from_numpy(image).permute(2, 0, 1)  # Convert from HxWxC to CxHxW
                else:
                    raise ValueError("Unexpected channel dimension in the numpy array.")
            elif image.ndim == 2:
                # If it's a grayscale image, add a channel dimension
                image = torch.from_numpy(image).unsqueeze(0)  # Convert from HxW to 1xHxW
            else:
                raise ValueError("The numpy array must be either HxWxC or HxW.")

        # Check if image is a PyTorch tensor and has 3 dimensions (C, H, W)
        elif isinstance(image, torch.Tensor):
            if image.ndim != 3 or image.shape[0] not in [1, 3, 4]:
                raise ValueError("The tensor must have 3 dimensions with C in [1, 3, 4].")
        else:
            raise TypeError("The image must be a numpy array or a PyTorch tensor.")

        # Extract iteration information for tagging
        iteration_value = None
        for key in artifact_record.keys():
            if 'iteration' in key.lower():
                iteration_value = artifact_record[key]
                break

        # Extract model name for tagging
        model_name = artifact_record.get('ModelName', 'undefined')

        # Extract ExperimentName for tagging
        experiment_name = artifact_record.get('ExperimentName', 'default')

        # Generate image tag using model name only (without iteration value)
        artifact_id = str(artifact_record.get('ArtifactID', ''))[-5:]
        image_tag = f"{model_name}_{artifact_id}_image"

        # Log the image to TensorBoard with the generated tag and iteration value as global_step
        iteration_value = artifact_record.get('TrainingIteration', None)
        self.writer.add_image(image_tag, image, global_step=iteration_value)

        # Convert the tensor to a PIL Image and save it
        pil_image = to_pil_image(image)
        image_path = os.path.join(self.log_dir, "image", f"{image_tag}_{iteration_value}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        pil_image.save(image_path)

        # Add the image related information to artifact_record
        artifact_record['ImageInfo'] = {'ImagePath': image_path, 'Tag': image_tag}

        # Log scalar artifacts
        self.logArtifact(artifact_record)

    def _record_log_type(self, log_type):
        # Record log type and path
        if log_type not in self.log_types:
            self.log_types[log_type] = self.writer.log_dir

    def getLogTypes(self):
        # Get the dictionary of log types and their paths
        return self.log_types

    def close(self):
        self.writer.close()
