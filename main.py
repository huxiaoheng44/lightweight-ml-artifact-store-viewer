from LoggerAndViewer.gui.gui import TrainingLogApp
import tkinter as tk

def run():
    root = tk.Tk()
    app = TrainingLogApp(root)
    root.mainloop()

if __name__ == '__main__':
    run()
