import tkinter as tk
from tkinter import messagebox, filedialog
from ..utils.handleStorage import upload_to_s3
from ..utils.handleStorage import aws_credentials_configured
from ..utils.handleStorage import list_files_in_bucket
from ..utils.handleStorage import list_s3_buckets
from ..utils.handleStorage import download_from_s3
from ..utils.handleStorage import configure_aws_session


class TrainingLogApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Login')
        if not aws_credentials_configured():
            self.prompt_aws_credentials()
        else:
            self.create_main_window()


    def center_window(self, width, height, window=None):
        """ Center window on the screen with the given width and height. """
        if window is None:
            window = self.master
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f'{width}x{height}+{x}+{y}')

    # def create_login_frame(self):
    #     """ Create the login frame. """
    #     self.frame_login = tk.Frame(self.master)
    #     self.frame_login.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    #     tk.Label(self.frame_login, text="Username:").grid(row=0, column=0, padx=10, pady=5)
    #     self.entry_username = tk.Entry(self.frame_login)
    #     self.entry_username.grid(row=0, column=1, padx=10, pady=5)

    #     tk.Label(self.frame_login, text="Password:").grid(row=1, column=0, padx=10, pady=5)
    #     self.entry_password = tk.Entry(self.frame_login, show="*")
    #     self.entry_password.grid(row=1, column=1, padx=10, pady=5)

    #     tk.Button(self.frame_login, text="Login", command=self.attempt_login).grid(row=2, column=0, columnspan=2,
    #                                                                                pady=10)

    def validate_login(self, username, password):
        """ Validate the login credentials. """
        return username == "admin" and password == "admin"

    def attempt_login(self):
        """ Attempt to login to the application. """
        username = self.entry_username.get()
        password = self.entry_password.get()
        if self.validate_login(username, password):
            self.master.withdraw()  # Hide the login window
            self.create_main_window()
        else:
            messagebox.showerror("Error", "Incorrect username or password!")

    def create_main_window(self):
        self.master.withdraw()
        self.main_window = tk.Toplevel(self.master)
        self.main_window.title('Training Log Keeper')
        self.center_window(500, 400, self.main_window)
        
        # Configure columns to distribute space evenly
        self.main_window.grid_columnconfigure(0, weight=1)
        self.main_window.grid_columnconfigure(1, weight=1)
        
        # Button for selecting a log file
        self.button_select_file = tk.Button(self.main_window, text="Select Log File", command=self.select_log_file)
        self.button_select_file.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Button for selecting a log folder
        self.button_select_folder = tk.Button(self.main_window, text="Select Log Folder", command=self.select_folder)
        self.button_select_folder.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Button for selecting file from S3
        self.button_select_s3_file = tk.Button(self.main_window, text="Select File from S3", command=self.select_s3_file)
        self.button_select_s3_file.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Listbox to display selected files and folders
        self.listbox_paths = tk.Listbox(self.main_window)
        self.listbox_paths.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        # Button to add metadata to the selected file or folder
        self.button_add_metadata = tk.Button(self.main_window, text="Add Metadata", command=self.add_metadata)
        self.button_add_metadata.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Button to upload the selected file or folder to S3
        self.button_upload_s3 = tk.Button(self.main_window, text="Upload to S3", command=self.upload_to_s3)
        self.button_upload_s3.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Button to add IAM configuration
        self.button_iam_configure = tk.Button(self.main_window, text="Add IAM Configuration", command=self.prompt_aws_credentials)
        self.button_iam_configure.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Adjust the grid to make the listbox expand
        self.main_window.grid_rowconfigure(2, weight=1)



    def select_folder(self):
        """ Open a directory dialog and update the listbox with the selected folder. """
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.listbox_folders.insert(tk.END, folder_path)

    def select_log_file(self):
        """ Open a file dialog and update the listbox with the selected file. """
        file_path = filedialog.askopenfilename()
        if file_path:
            self.listbox_files.insert(tk.END, file_path)
            
            
    def select_s3_file(self):
        self.s3_window = tk.Toplevel(self.master)
        self.s3_window.title('Select S3 File')
        self.center_window(300, 400, self.s3_window)

        # Configure grid layout
        self.s3_window.grid_rowconfigure(0, weight=1)
        self.s3_window.grid_rowconfigure(1, weight=0)
        self.s3_window.grid_columnconfigure(0, weight=1)

        # Listbox to display buckets with a scrollbar
        self.scrollbar_buckets = tk.Scrollbar(self.s3_window, orient="vertical")
        self.listbox_buckets = tk.Listbox(self.s3_window, yscrollcommand=self.scrollbar_buckets.set)
        self.scrollbar_buckets.config(command=self.listbox_buckets.yview)
        self.scrollbar_buckets.grid(row=0, column=1, sticky='ns')
        self.listbox_buckets.grid(row=0, column=0, sticky='nsew')
        self.listbox_buckets.bind('<<ListboxSelect>>', self.on_bucket_selected)

        # Listbox to display files in the selected bucket with a scrollbar
        self.scrollbar_files = tk.Scrollbar(self.s3_window, orient="vertical")
        self.listbox_s3_files = tk.Listbox(self.s3_window, yscrollcommand=self.scrollbar_files.set)
        self.scrollbar_files.config(command=self.listbox_s3_files.yview)
        self.scrollbar_files.grid(row=2, column=1, sticky='ns')
        self.listbox_s3_files.grid(row=2, column=0, sticky='nsew')

        # Button to download the selected file
        self.button_download = tk.Button(self.s3_window, text="Download", command=self.download_file)
        self.button_download.grid(row=3, column=0, columnspan=2, sticky='ew')

        self.load_buckets()

        
    def on_bucket_selected(self, event):

        widget = event.widget
        selection = widget.curselection()
        if selection:
            index = selection[0]
            bucket_name = widget.get(index)
            self.current_bucket = widget.get(index)
            self.listbox_s3_files.delete(0, tk.END)
            try:

                files = list_files_in_bucket(bucket_name)

                for file in files:
                    self.listbox_s3_files.insert(tk.END, file)
            except Exception as e:
                messagebox.showerror("Error", f"Can not read file list from {bucket_name}: {e}")
        
    
    def download_file(self):
        selected_indices = self.listbox_s3_files.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a file to download.")
            return
        selected_file = self.listbox_s3_files.get(selected_indices[0])
        

        download_directory = filedialog.askdirectory()
        if not download_directory:
            messagebox.showwarning("Warning", "Download cancelled.")
            return
        

        try:
            download_path = f"{download_directory}/{selected_file}"
            download_from_s3(self.current_bucket, selected_file, download_path)
            messagebox.showinfo("Success", f"File downloaded to {download_path}")
            # Add the downloaded file's path to the main window's Listbox
            self.listbox_paths.insert(tk.END, download_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download file: {e}")
    
    
    def load_buckets(self):

        try:
            buckets = list_s3_buckets()
            for bucket in buckets:
                self.listbox_buckets.insert(tk.END, bucket)
        except Exception as e:
            messagebox.showerror("Error", f"Can not get bucket list from S3: {e}")

    def add_metadata_to_file(self):
        """ Add metadata to the selected log file. """
        selected_indices = self.listbox_files.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a file.")
            return

        selected_file = self.listbox_files.get(selected_indices[0])
        self.add_metadata(selected_file)

    def add_metadata_to_folder(self):
        """ Add metadata to the selected folder. """
        selected_indices = self.listbox_folders.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a folder.")
            return

        selected_folder = self.listbox_folders.get(selected_indices[0])
        # Implement the logic to add metadata here.
        # ...



    def upload_to_s3(self):
        """ Upload the selected folder to S3. """
        selected_indices = self.listbox_folders.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a folder.")
            return

        selected_folder = self.listbox_folders.get(selected_indices[0])
        # Implement the logic to upload the folder to S3 here.
        # ...

    def add_metadata(self):
        """ Add metadata key-value pair to the dictionary and listbox. """
        key = self.entry_key.get()
        value = self.entry_value.get()
        if key and value:  # Ensure that neither key nor value is empty
            self.metadata[key] = value
            self.listbox_metadata.insert(tk.END, f"{key}: {value}")
            self.entry_key.delete(0, tk.END)
            self.entry_value.delete(0, tk.END)
        else:
            messagebox.showwarning("Warning", "Both key and value are required.")

    def on_close_main_window(self):
        """ Handle the closing of the main window. """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.main_window.destroy()
            self.master.destroy()
            
            
    def prompt_aws_credentials(self):
        """ Prompt the user to enter AWS credentials. """
        self.credentials_window = tk.Toplevel(self.master)
        self.credentials_window.title('AWS Credentials')
        self.center_window(300, 150, self.credentials_window) 


        current_access_key, current_secret_key = self.get_current_aws_credentials()

        tk.Label(self.credentials_window, text="AWS Access Key ID:").grid(row=0, column=0)
        self.entry_access_key = tk.Entry(self.credentials_window)
        self.entry_access_key.insert(0, current_access_key) 
        self.entry_access_key.grid(row=0, column=1)

        tk.Label(self.credentials_window, text="AWS Secret Access Key:").grid(row=1, column=0)
        self.entry_secret_key = tk.Entry(self.credentials_window, show='*')
        self.entry_secret_key.insert(0, current_secret_key) 
        self.entry_secret_key.grid(row=1, column=1)

    def get_current_aws_credentials(self):

        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials.access_key, credentials.secret_key
        except Exception:
            return "", "" 


    def set_aws_credentials(self):
        """ Set AWS credentials and close the credentials window. """
        access_key = self.entry_access_key.get()
        secret_key = self.entry_secret_key.get()
        if access_key and secret_key:
            configure_aws_session(access_key, secret_key)
            self.credentials_window.destroy()
            self.create_main_window() 
        else:
            messagebox.showwarning("Warning", "Both Access Key and Secret Key are required.")

            
            
    def prompt_aws_credentials(self):
        """ Prompt the user to enter AWS credentials. """
        self.credentials_window = tk.Toplevel(self.master)
        self.credentials_window.title('AWS Credentials')

        tk.Label(self.credentials_window, text="AWS Access Key ID:").grid(row=0, column=0)
        self.entry_access_key = tk.Entry(self.credentials_window)
        self.entry_access_key.grid(row=0, column=1)

        tk.Label(self.credentials_window, text="AWS Secret Access Key:").grid(row=1, column=0)
        self.entry_secret_key = tk.Entry(self.credentials_window, show='*')
        self.entry_secret_key.grid(row=1, column=1)

        tk.Button(self.credentials_window, text="Submit", command=self.set_aws_credentials).grid(row=2, column=0, columnspan=2)
        tk.Button(self.credentials_window, text="Cancel", command=self.credentials_window.destroy).grid(row=2, column=1, columnspan=2)




