from LoggerAndViewer.gui.gui import TrainingLogApp
from LoggerAndViewer.dataset.DataManager import MetaDataManager
from LoggerAndViewer.visualization.viwer import Viewer
import numpy as np
import tkinter as tk

def run():
    # viewer = Viewer()
    # viewer.add_scalar('loss', 0.1, 1)
    # viewer.add_image('example_image', np.random.rand(3, 100, 100), 2)
    # viewer.add_text('example_text', 'Hello, TensorBoard!', 3)
    # viewer.add_histogram('example_histogram', np.random.randn(1000), 4)
    # viewer.close()
    
    
    # manager = MetaDataManager()
    # manager.addMetaData({"name": "example", "value": 42})
    
    
    root = tk.Tk()
    app = TrainingLogApp(root)
    root.mainloop()

if __name__ == '__main__':
    run()
