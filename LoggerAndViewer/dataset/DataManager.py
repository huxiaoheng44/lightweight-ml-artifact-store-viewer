import json
import re

from bson import ObjectId
from pymongo import MongoClient,errors
from pymongo.errors import ConnectionFailure, PyMongoError, OperationFailure, ConfigurationError


class DataManager:
    def __init__(self,

                 uri="mongodb+srv://huxiaoheng:hxh19981225@xiaoheng.nymd030.mongodb.net/?retryWrites=true&w=majority"):
        try:
            self.client = MongoClient(uri)
            self.client.admin.command('ping')
            self.db = self.client['PMSD']
            self.collection = self.db['metadata']
            self.artifact_record = {}
            # print("Connected to MongoDB.")
        except ConnectionFailure:
            print("Failed to connect to MongoDB. Please check your connection details.")
        except PyMongoError as e:
            print(f"An error occurred while connecting to MongoDB: {e}")


    def getAllArtifacts(self):
        """Retrieve all artifacts from the 'artifacts' collection."""
        try:
            self.collection = self.db['artifacts']
            artifacts = list(self.collection.find({}))
            print(f"Retrieved {len(artifacts)} artifacts.")
            return artifacts
        except PyMongoError as e:
            print(f"An error occurred while retrieving artifacts: {e}")
            return []

    def searchArtifacts(self, query):
        """Search for artifacts in the 'artifacts' collection based on a query."""
        try:
            self.collection = self.db['artifacts']
            results = list(self.collection.find(query))
            print(f"Artifacts found: {len(results)}")
            return results
        except PyMongoError as e:
            print(f"An error occurred while searching for artifacts: {e}")


    def searchMetadataByKey(self, key_pattern):
        try:
            regex = re.compile(key_pattern, re.IGNORECASE)

            query = {"Metadata.key": {"$regex": regex}}

            return list(self.db['artifacts'].find(query))
        except errors.PyMongoError as e:
            print(f"An error occurred while searching for metadata keys: {e}")
            return []



    def addData(self, collection_name, data):
        """Add data to the specified collection."""
        try:
            # Select the collection based on the provided collection name
            self.collection = self.db[collection_name]

            # Insert the new data
            result = self.collection.insert_one(data)
            print(f"Data added with ID: {result.inserted_id} in collection: {collection_name}")
            return result.inserted_id
        except PyMongoError as e:
            print(f"An error occurred while adding metadata to collection: {collection_name}: {e}")
            return None

    def addFile(self, file_json):
        """Add or update a file in the 'files' collection."""
        try:
            # Switch to the 'files' collection in the database.
            self.collection = self.db['files']
            file_query = {"file": file_json["file"]}
            existing_record = self.collection.find_one(file_query)

            if existing_record:

                updated_values = {"$set": {"filetype": file_json["type"]}}
                result = self.collection.update_one(file_query, updated_values)
                print(f"Filetype updated for file: {file_json.file}")
                return result.modified_count
            else:

                result = self.collection.insert_one(file_json)
                print(f"File added with ID: {result.inserted_id}")
                return result.inserted_id
        except PyMongoError as e:
            print(f"An error occurred while adding/updating file: {e}")


    def generate_default_value(self, type_str):
        # Generate a default value based on the type string
        default_values = {
            'float': 0.0,
            'integer': 0,
            'string': '',
            'boolean': False,  # Boolean type
            'list': [],  # Empty list
            'dict': {},  # Empty dictionary
            'none': None,  # NoneType
            # Complex types: initializing as empty structures
            'tuple': (),  # Empty tuple
            'set': set(),  # Empty set
            # For more complex structures, you can define a simple instantiation
            # or a function that generates the default structure
        }
        return default_values.get(type_str.lower(), "Unsupported type")

    def readArtifactsFromFile(self, file_path):
        # Try to read the content of the file
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                artifacts_type_data = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {file_path}")
            print("Error details:", e)  # Print the detailed exception
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        # Initialize the 'artifacts' collection if it doesn't exist
        if 'artifacts' not in self.db.list_collection_names():
            self.db.create_collection('artifacts')

        # Construct the artifact record with default values based on the types
        self.artifact_record = {key: self.generate_default_value(value) for key, value in artifacts_type_data.items()}
        self.artifact_record['ArtifactID'] = ObjectId()  # Generate a new ObjectId for the artifact
        self.artifact_record['ModelName'] = "undefined" # Default ModelName
        self.artifact_record['FileList'] = []  # This can be populated as needed
        self.artifact_record['Metadata'] = []  # Initialize Metadata as an empty list

        return self.artifact_record

    def searchArtifactByFieldValue(self, field, value):
        """Search for artifacts in the 'artifacts' collection based on a field and a value (fuzzy search)."""
        try:
            query = {field: re.compile(value, re.IGNORECASE)}
            result =  self.searchArtifacts(query)
            return result
        except errors.PyMongoError as e:
            print(f"An error occurred while searching for artifacts by field value: {e}")
            return []

    def getArtifactTemplate(self):
        return self.artifact_record.copy()


