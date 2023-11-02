import tkinter as tk
from tkinter import ttk
import asyncio
import os

class ProgressBarApp:
    def __init__(self, total_files, total_folders, folder_image_counts):
        self.root = tk.Tk()
        self.root.title("Processing Images")

        # Overall progress bar
        self.total_progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.total_progress.grid(row=0, column=0, padx=10, pady=10)
        self.total_progress["maximum"] = total_files
        self.total_label = tk.Label(self.root, text=f"Overall Progress: 0/{total_files}")
        self.total_label.grid(row=0, column=1, padx=10, pady=10)

        # Current folder file progress bar
        self.current_folder_progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.current_folder_progress.grid(row=1, column=0, padx=10, pady=10)
        self.current_folder_label = tk.Label(self.root, text="Current Folder: 0/0")
        self.current_folder_label.grid(row=1, column=1, padx=10, pady=10)

        # Folder progress bar
        self.folder_progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.folder_progress.grid(row=2, column=0, padx=10, pady=10)
        self.folder_progress["maximum"] = total_folders
        self.folder_label = tk.Label(self.root, text=f"Folder Progress: 0/{total_folders}")
        self.folder_label.grid(row=2, column=1, padx=10, pady=10)

        # TODO: Fix Stop Buttons
        # # Stop buttons
        # self.stop_immediately = False
        # self.stop_after_folder = False
        # self.stop_button = tk.Button(self.root, text="Stop Immediately", command=self.stop_now)
        # self.stop_button.grid(row=3, column=0, padx=10, pady=10)
        # self.stop_after_folder_button = tk.Button(self.root, text="Stop After Current Folder", command=self.stop_after_current)
        # self.stop_after_folder_button.grid(row=3, column=1, padx=10, pady=10)
        
        # current folder name label
        self.current_folder_name_label = tk.Label(self.root, text="Current Folder: None")
        self.current_folder_name_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        # last processed file label
        self.last_processed_file_label = tk.Label(self.root, text="Last Processed File: None")
        self.last_processed_file_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="w")


    def close(self):
        self.root.destroy()

    def stop_now(self):
        self.stop_immediately = True

    def stop_after_current(self):
        self.stop_after_folder = True

    def update_current_folder_name(self, folder_path):
        folder_name = os.path.basename(folder_path)
        self.current_folder_name_label.config(text=f"Current Folder: {folder_name}")

    def update_last_processed_file(self, file_path):
        file_name = os.path.basename(file_path)
        self.last_processed_file_label.config(text=f"Last Processed File: {file_name}")

        
    def update_progress_data(self, progress_data, folder_image_counts):
        # Update overall progress bar
        total_files_processed = progress_data['items_processed']
        self.update_total_progress(total_files_processed)

        # Update current folder progress (Bar #2)
        self.update_current_folder_progress(progress_data, folder_image_counts)

        # Update folder progress (Bar #3)
        self.update_folders_progress(progress_data)
        
        # Update the current folder name and last processed file labels
        self.update_current_folder_name(progress_data['current_folder'])
        self.update_last_processed_file(progress_data['last_processed_file'])

    def update_total_progress(self, value):
        value = int(value)
        self.total_progress["value"] = value
        max_val = self.total_progress["maximum"]
        self.total_label.config(text=f"Overall Progress: {value}/{max_val}")

    def update_folders_progress(self, progress_data):
        folders_completed = progress_data['folders_completed']
        max_val = self.folder_progress["maximum"]
        self.folder_progress["value"] = folders_completed
        # Corrected the label for folder progress
        self.folder_label.config(text=f"Folder Progress: {folders_completed}/{max_val}")

    def update_current_folder_progress(self, progress_data, folder_image_counts):
        current_folder = progress_data['current_folder']
        items_processed_current_folder = progress_data['items_processed_current_folder']
        max_value = folder_image_counts.get(current_folder, 0)
        self.current_folder_progress["value"] = items_processed_current_folder
        self.current_folder_progress["maximum"] = max_value
        # Corrected the label for current folder progress
        self.current_folder_label.config(text=f"Current Folder: {items_processed_current_folder}/{max_value}")

    def run(self):
        self.root.mainloop()