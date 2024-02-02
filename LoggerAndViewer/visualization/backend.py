import json
import logging
from flask import Flask, render_template, request, jsonify
from ..dataset.DataManager import DataManager
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os
import uuid

class VisualBackend:
    def __init__(self):
        self.app = Flask(__name__, static_folder='../../static')
        # used for dealing with database
        self.meta_data_manager = DataManager()

        logging.basicConfig(level=logging.INFO)

        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            group_by = request.args.get('group_by', 'ModelName')
            search_field = request.args.get('search_field')
            search_value = request.args.get('search_value')
            search_active = False  # Flag to indicate if search is active

            if search_field and search_value:

                if search_field == 'Metadata':
                    search_data = self.meta_data_manager.searchMetadataByKey(search_value)
                else:
                    search_data = self.meta_data_manager.searchArtifactByFieldValue(search_field, search_value)
                    print(search_data)
                grouped_data = self.groupArtifactsBy(group_by, search_data)  # Use search data for grouping
                search_active = True
            else:
                all_data = self.meta_data_manager.getAllArtifacts()
                grouped_data = self.groupArtifactsBy(group_by, all_data)

            return render_template('index.html', grouped_data=grouped_data, group_by=group_by,
                                   search_active=search_active)

        @self.app.route('/start-tensorboard')
        def start_tensorboard():
            try:
                # Start TensorBoard
                subprocess.Popen(["tensorboard", "--logdir=logs", "--port=6006"])
                return "TensorBoard is starting on port 6006."
            except Exception as e:
                return f"An error occurred: {e}"

        @self.app.route('/generate-charts', methods=['POST'])
        def generate_charts():
            try:
                data = json.loads(request.data)
                file_paths = data.get('files', [])
                logging.info(f"Received file paths: {file_paths}")

                # Call the function to generate combined charts
                print("Generating combined charts...")
                chart_paths = self.generate_combined_charts(file_paths)

                # Since the function already returns paths, we don't need to check success here.
                return jsonify(chart_paths)
            except Exception as e:
                logging.error(f"Error in generate_charts: {e}")
                return jsonify({"error": str(e)}), 500


    def generate_combined_charts(self, csv_paths, static_dir='static/visualizationImages'):
        numeric_data = {}
        iteration_columns = {}
        image_data = {}

        # make sure the static directory exists
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # traverse all csv files
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
                continue

            # read csv file
            data = pd.read_csv(csv_path)
            model_name = os.path.splitext(os.path.basename(csv_path))[0]

            # find the iteration column
            iteration_column = [col for col in data.columns if 'iteration' in col.lower()][0]
            iteration_columns[model_name] = iteration_column
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]) and column != iteration_column:
                    if column not in numeric_data:
                        numeric_data[column] = {}
                    numeric_data[column][model_name] = data[iteration_column].astype(int), data[column]
                elif column == 'ImageInfo':
                    image_data[model_name] = [
                        json.loads(row.replace("'", "\"")) for row in data[column]
                    ]

        # generate charts
        # chart_paths = []
        # for column, model_data in numeric_data.items():
        #     plt.figure(figsize=(10, 6))
        #     for model_name, (iterations, values) in model_data.items():
        #         plt.plot(iterations, values, marker='o', linestyle='-', label=model_name)
        #
        #     plt.xlabel('Training Iteration')
        #     plt.ylabel(column)
        #     plt.title(f'{column} Over Iterations')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     # generate a unique file name
        #     chart_file_name = f'chart_{column}_{uuid.uuid4().hex[:8]}.png'
        #     chart_path = os.path.join(static_dir, chart_file_name)
        #     plt.savefig(chart_path)
        #     plt.close()
        #
        #     # Append the web-accessible path
        #     web_accessible_path = '/' + os.path.join(static_dir, chart_file_name).replace('\\', '/')
        #     chart_paths.append(web_accessible_path)
        #
        # print(f"Generated chart paths: {chart_paths}")
        # return chart_paths

        # generate numeric charts
        chart_paths = self.generate_numeric_charts(numeric_data, iteration_columns, static_dir)

        # generate image charts
        print("image_data: ", image_data)
        chart_paths.extend(self.generate_image_charts(image_data, static_dir))

        return chart_paths

    def generate_numeric_charts(self, numeric_data, iteration_columns, static_dir):

        # generate charts
        print(f"Generating numeric charts")
        chart_paths = []
        for column, model_data in numeric_data.items():
            plt.figure(figsize=(10, 6))
            for model_name, (iterations, values) in model_data.items():
                plt.plot(iterations, values, marker='o', linestyle='-', label=model_name)

            plt.xlabel('Training Iteration')
            plt.ylabel(column)
            plt.title(f'{column} Over Iterations')
            plt.legend()
            plt.grid(True)

            chart_file_name = f'chart_{column}_{uuid.uuid4().hex[:8]}.png'
            chart_path = os.path.join(static_dir, chart_file_name)
            plt.savefig(chart_path)
            plt.close()

            web_accessible_path = '/' + os.path.join(static_dir, chart_file_name).replace('\\', '/')
            chart_paths.append(web_accessible_path)

        # print(f"Generated chart paths: {chart_paths}")
        return chart_paths

    def generate_image_charts(self, image_data, static_dir):
        chart_paths = []
        # maximum number of images per chart
        max_images_per_chart = 10

        # generate charts
        for model_name, model_images in image_data.items():
            for i in range(0, len(model_images), max_images_per_chart):
                num_images = min(max_images_per_chart, len(model_images) - i)
                fig, axs = plt.subplots(1, num_images, figsize=(20, 3))
                fig.suptitle(f'Images for {model_name}', fontsize=16)

                batch_images = model_images[i:i + num_images]

                for ax, image_info in zip(axs, batch_images):
                    image_path = image_info['ImagePath']
                    image_tag = image_info['Tag']

                    # read the image
                    full_image_path = os.path.join(image_path)  # use the full path
                    try:
                        img = plt.imread(full_image_path)
                        ax.imshow(img)
                        ax.set_title(f'{image_tag}')
                        ax.axis('off')  # hide the axis
                    except FileNotFoundError:
                        print(f"Image not found: {full_image_path}")
                        ax.set_title('Image not found')
                        ax.axis('off')

                # adjust the layout
                plt.tight_layout()

                # generate a unique file name
                chart_file_name = f'chart_images_{model_name}_{uuid.uuid4().hex[:8]}.png'
                chart_path = os.path.join(static_dir, chart_file_name)
                plt.savefig(chart_path)
                plt.close()

                # Append the web-accessible path
                web_accessible_path = '/' + os.path.join(static_dir, chart_file_name).replace('\\', '/')
                chart_paths.append(web_accessible_path)

        return chart_paths

    def launch(self):
        self.app.run(debug=False)

    def groupArtifactsBy(self, field, artifacts):
        grouped_artifacts = {}

        for artifact in artifacts:
            key = artifact.get(field)
            if key not in grouped_artifacts:
                grouped_artifacts[key] = []
            grouped_artifacts[key].append(artifact)

        return list(grouped_artifacts.values())

