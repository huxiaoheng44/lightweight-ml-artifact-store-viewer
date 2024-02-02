# Main code file

from torch import optim, nn
from torchvision import datasets, transforms
from Models import LinearGenerator, ConvolutionalGenerator, Encoder
from LoggerAndViewer.dataset.DataManager import DataManager
from torch.utils.data import DataLoader, TensorDataset
from LoggerAndViewer.log.log import Logger
from LoggerAndViewer.visualization.backend import VisualBackend

# Function to load MNIST dataset
def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing the images to (-1, 1)
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Function to train the model
def train_model(encoder, generator, gen_optimizer, enc_optimizer, dataloader, num_epochs, logger, data_manager, experiment_name="", model_name="", meta_data={}):
    # Training loop
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in dataloader:
            # Flattening MNIST images and generating noise vectors
            images_flat = images.view(images.size(0), -1)
            noise = encoder(images_flat)



            # Generating images using the generator
            fake_images = generator(noise)

            # Check if fake_images is not flat and flatten if necessary
            if fake_images.dim() == 4:  # Check if the output is 4D
                fake_images = fake_images.view(images.size(0), -1)  # Flatten the images

            loss = criterion(fake_images, images_flat)


            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()
            loss.backward()
            enc_optimizer.step()
            gen_optimizer.step()

            # Accumulating loss
            total_loss += loss.item()

        # generate a new artifact record for each epoch !!! super important
        artifact_record = data_manager.getArtifactTemplate()
        artifact_record.update({
            'MSE': total_loss,
            'ModelName': model_name,
            'ExperimentName': experiment_name,
            'TrainingIteration': epoch + 1,
            'Metadata': meta_data,
            'loss': total_loss,
        })

        # input the image as a tensor
        # Reshape the output to 2D images if using LinearGenerator
        if fake_images.dim() == 2:  # Check if the output is flat
            fake_images = fake_images.view(-1, 1, 32, 32)  # Reshape to [N, C, H, W] format
        logger.logArtifactWithImage(artifact_record, fake_images)

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")



    return average_loss  # Returning average loss


# Main function
def main():
    batch_size = 64  # Batch size for MNIST dataset
    noise_dim = 100  # Dimension of noise vector

    # Parameters for the generators
    linear_image_dim = 32*32  # Assuming generating 28x28 images
    conv_image_channels = 1   # Assuming generating single-channel images

    # Load MNIST data
    mnist_dataloader = load_mnist_data(batch_size)

    # Define meta data manager
    data_manager = DataManager()
    # Read artifacts from file and generate an artifact template
    data_manager.readArtifactsFromFile("./Artifacts.json")

    # Instantiating models
    encoder = Encoder(input_dim=32*32, output_dim=noise_dim)
    linear_generator = LinearGenerator(noise_dim=noise_dim, image_dim=linear_image_dim)
    conv_generator = ConvolutionalGenerator(noise_dim=noise_dim, image_channels=conv_image_channels)

    # Optimizers
    linear_optimizer = optim.Adam(linear_generator.parameters(), lr=0.0002)
    conv_optimizer = optim.Adam(conv_generator.parameters(), lr=0.0002)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0002)

    # Required field for linear_generator
    meta_data = {"Type": "Linear_ImageGenerator", "release-date": "2024-02-01"}
    logger_linear_generator = Logger(log_dir='linear_generator_2')
    experiment_name_linear_generator = "linear_generator_experiment-2"
    model_name_linear_generator = "linear_generator"

    # Required field for conv_generator
    meta_data = {"Type": "Conv_ImageGenerator", "release-date": "2024-02-01"}
    logger_conv_generator = Logger(log_dir='conv_generator_2')
    experiment_name_conv_generator = "conv_generator_experiment-2"
    model_name_conv_generator = "conv_generator"



    # Training models
    train_model(encoder, linear_generator, linear_optimizer, encoder_optimizer, mnist_dataloader, num_epochs=10, logger=logger_linear_generator, data_manager=data_manager, experiment_name=experiment_name_linear_generator, model_name=model_name_linear_generator, meta_data=meta_data)
    train_model(encoder, conv_generator, conv_optimizer, encoder_optimizer, mnist_dataloader, num_epochs=10, logger=logger_conv_generator, data_manager=data_manager, experiment_name=experiment_name_conv_generator, model_name=model_name_conv_generator, meta_data=meta_data)

if __name__ == "__main__":
#    main()
    visualizer = VisualBackend()
    visualizer.launch()
