import requests
import base64
from PIL import Image
import io
import os
import argparse
import tkinter as tk
from tkinter import filedialog
import httpx
import asyncio
import threading
from UI_Progress import ProgressBarApp
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

HOST = config['HOST']
URI = config['URI_template'].format(HOST=HOST)
max_image_size = config['max_image_size']
caption_start_template = config['caption_start_template']
max_concurrent_requests = config['max_concurrent_requests']
httpx_timeout_value = config['httpx_timeout']
API_Payload = config['API_Payload']
prompt_template = '\n'.join(API_Payload['prompt'])
del API_Payload['prompt']

semaphore = asyncio.Semaphore(max_concurrent_requests)

progress_data = {
    'items_processed': 0,
    'items_processed_current_folder': 0,
    'current_folder': '',
    'folders_completed': 0,
    'last_processed_file': ''
}

async def process_folder(folder_path, payload, semaphore, update_queue):
    file_names = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    for file in file_names:
        if shutdown_flag.is_set():
            break
        image_path = os.path.join(folder_path, file)
        await run(payload, image_path, os.path.basename(folder_path), semaphore, update_queue, folder_path)
    progress_data['folders_completed'] += 1

async def run_async_folders(folder_image_counts, payload, semaphore, update_queue):
    for folder, _ in folder_image_counts.items():
        await process_folder(folder, payload, semaphore, update_queue)
    
def count_files_and_folders(master_folder):
    folder_image_counts = {}
    
    subfolders = [f.path for f in os.scandir(master_folder) if f.is_dir()]
    for folder in subfolders:
        image_count = sum(1 for item in os.listdir(folder) if item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        folder_image_counts[folder] = image_count

    folder_image_counts = {k: v for k, v in folder_image_counts.items() if v > 0}

    total_folders = len(folder_image_counts)
    total_files = sum(folder_image_counts.values())

    return total_files, total_folders, folder_image_counts

def image_to_base64(image_path):
    img = Image.open(image_path)
    
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
        
    width, height = img.size
    aspect_ratio = float(height) / float(width)
        
    if width > height:
        new_width = max_image_size
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = max_image_size
        new_width = int(new_height / aspect_ratio)

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img_resized.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_image
    
def save_result_to_file(image_path, result):
    txt_file_path = os.path.splitext(image_path)[0] + '.txt'
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(result)

async def run(payload, image_path, folder_name, semaphore, update_queue, folder_path):
    async with semaphore:
        this_payload = payload.copy()

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_start = caption_start_template.replace('@folder_name', folder_name).replace('@image_name', image_name)
        prompt = prompt_template.replace("@folder_name", folder_name).replace("@image_name", image_name)
        
        base64_image = image_to_base64(image_path)
        this_payload["messages"] = [
            {
                "role": "user",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            {
                "role": "user",
                "content": prompt + caption_start,
            },
        ]
        timeout = httpx.Timeout(httpx_timeout_value)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(URI, json=this_payload)

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            save_result_to_file(image_path, caption_start+result)
            print("\n", image_path, "\n", caption_start+result)
            progress_data['items_processed'] += 1
            progress_data['items_processed_current_folder'] += 1
            progress_data['last_processed_file'] = image_path
            if progress_data['current_folder'] != folder_path:
                progress_data['current_folder'] = folder_path
                progress_data['items_processed_current_folder'] = 1
            progressBar.update_progress_data(progress_data, folder_image_counts)

def get_folder_from_gui():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected
    
def process_queue():
    while not update_queue.empty():
        image_path, folder_name = update_queue.get_nowait()
        progressBar.update_total_progress(progressBar.total_progress["value"] + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images in subfolders of a master folder.')
    parser.add_argument('--folder', type=str, help='Path to the master folder containing subfolders with images.')

    args = parser.parse_args()

    if args.folder:
        master_folder_path = args.folder
    else:
        master_folder_path = get_folder_from_gui()

    if not master_folder_path:
        print("No folder selected. Exiting.")
        exit()
    
    total_files, total_folders, folder_image_counts = count_files_and_folders(master_folder_path)
    print("Total image files:", total_files)
    print("Total folders with images:", total_folders)
    progressBar = ProgressBarApp(total_files, total_folders, folder_image_counts)

    update_queue = asyncio.Queue()
    all_tasks_done = threading.Event()
    shutdown_flag = threading.Event()

    def start_async_processing():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_async_folders(folder_image_counts, API_Payload, semaphore, update_queue))
            loop.run_until_complete(update_queue.join())
        except KeyboardInterrupt:
            print("Caught keyboard interrupt. Shutting down.")
            shutdown_flag.set()
        finally:
            loop.close()
            all_tasks_done.set()
            progressBar.close()

    processing_thread = threading.Thread(target=start_async_processing)
    processing_thread.start()

    def process_queue_thread():
        while not all_tasks_done.is_set() or not update_queue.empty():
            process_queue()
        progressBar.close()

    queue_thread = threading.Thread(target=process_queue_thread)
    queue_thread.start()
    progressBar.run()