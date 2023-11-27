import os
import shutil
from multiprocessing import Manager


class MemoryCache:
    def __init__(self):
        # Create threadsafe dict as cache
        manager = Manager()
        self.cache = manager.dict()
    
    def __contains__(self, key):
        return self.cache.__contains__(key)
    
    def __getitem__(self, key):
        return self.cache[key]
    
    def __setitem__(self, key, value):
        self.cache[key] = value


class DiskCache:
    def __init__(self, cache_path, cleanup=True):
        self.cache_path = cache_path
        self.cleanup = cleanup
        # Create cache dir
        os.makedirs(self.cache_path)
    
    def __del__(self):
        if self.cleanup:
            # Remove cache dir
            shutil.rmtree(self.cache_path)
    
    def __contains__(self, key):
        filepath = self.get_filepath(key)
        return os.path.exists(filepath)
    
    def __getitem__(self, key):
        filepath = self.get_filepath(key)
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def __setitem__(self, key, value):
        filepath = self.get_filepath(key)
        with open(filepath, 'rw') as f:
            return pickle.dump(value, f)
    
    def get_filepath(self, key):
        return os.path.join(self.cache_path, f'{key}.pkl')
