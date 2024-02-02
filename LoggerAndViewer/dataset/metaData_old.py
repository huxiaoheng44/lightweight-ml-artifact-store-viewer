from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError, OperationFailure, ConfigurationError
    
class MetaDataManager:
    def __init__(self, uri="mongodb+srv://huxiaoheng:hxh19981225@xiaoheng.nymd030.mongodb.net/?retryWrites=true&w=majority"):
        try:
            self.client = MongoClient(uri)
            self.client.admin.command('ping')  
            self.db = self.client['PMSD']
            self.collection = self.db['metadata']
            print("Connected to MongoDB.")
        except ConnectionFailure:
            print("Failed to connect to MongoDB. Please check your connection details.")
        except PyMongoError as e:
            print(f"An error occurred while connecting to MongoDB: {e}")

    def initMetaData(self, data):
        """Add metadata to the collection."""
        try:
            self.collection = self.db['metadata']
            result = self.collection.insert_one(data)
            print(f"MetaData added with ID: {result.inserted_id}")
            return result.inserted_id
        except PyMongoError as e:
            print(f"An error occurred while adding metadata: {e}")

    def deleteMetaData(self, query):
        """Delete metadata from the collection."""
        try:
            self.collection = self.db['metadata']
            result = self.collection.delete_many(query)
            print(f"MetaData deleted count: {result.deleted_count}")
            return result.deleted_count
        except PyMongoError as e:
            print(f"An error occurred while deleting metadata: {e}")

    def queryMetaData(self, query):
        """Query metadata from the collection."""
        try:
            self.collection = self.db['metadata']
            results = list(self.collection.find(query))
            print(f"MetaData queried count: {len(results)}")
            return results
        except PyMongoError as e:
            print(f"An error occurred while querying metadata: {e}")
            
    def addMetaData(self, data):
        """Add or update metadata in the collection based on the 'file' field."""
        try:
            self.collection = self.db['metadata']
            file_query = {"file": data["file"]}
            existing_record = self.collection.find_one(file_query)

            if existing_record:

                updated_metadata = existing_record.get("metadata", [])
                updated_metadata.extend(data["metadata"]) 
                updated_values = {"$set": {"metadata": updated_metadata}}
                result = self.collection.update_one(file_query, updated_values)
                print(f"MetaData updated for file: {data['file']}")
                return result.modified_count
            else:
                
                result = self.collection.insert_one(data)
                print(f"MetaData added with ID: {result.inserted_id}")
                return result.inserted_id
        except PyMongoError as e:
            print(f"An error occurred while adding/updating metadata: {e}")
            
            
    def addFile(self, file_json):
        """Add or update a file in the 'files' collection."""
        try:
            #Switch to the 'files' collection in the database.
            self.collection = self.db['files']
            file_query = {"file":  file_json["file"]}
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
         
            
    def deleteFile(self, file):
        """Delete a file from the 'files' collection."""
        try:
            self.collection = self.db['files']
            result = self.collection.delete_many({"file": file})
            print(f"File deleted count: {result.deleted_count}")
            return result.deleted_count
        except PyMongoError as e:
            print(f"An error occurred while deleting file: {e}")
            
            
    def getAllFiles(self):
        """Retrieve all files from the 'files' collection."""
        try:
            self.collection = self.db['files']

            files = list(self.collection.find({}))
            print(f"Retrieved {len(files)} files.")
            return files
        except PyMongoError as e:
            print(f"An error occurred while retrieving files: {e}")
            return []  



