import argparse
import base64
import datetime
import httpx
import io
import json
import os
import requests
import threading
import tkinter as tk
import torch
from tkinter import filedialog
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from UI_Progress import ProgressBarApp

# Load Configuration from File
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Debug Mode Configuration
debug_mode = False

# Global variables for Qwen model and tokenizer
qwen_model = None
qwen_tokenizer = None

# API Configuration
api_model = config.get('model', 'ooba')
api_key = config.get('key', '')
HOST = config.get('HOST', '')
URI = config.get('URI_template', '').format(HOST=HOST, model=api_model, API_KEY=api_key)
prompt_template = config.get('prompt', '')
API_Payload = config.get('API_Payload', {})

# Image Processing Configuration
max_image_size = config.get('max_image_size', 1024)

# Text Template Configuration
caption_start_template = config.get('caption_start_template', '')

# Network and Concurrency Configuration
max_concurrent_requests = config.get('max_concurrent_requests', 1)
httpx_timeout_value = config.get('httpx_timeout', 120.0)

# Semaphore for Controlling Concurrency
semaphore = threading.Semaphore(max_concurrent_requests)

# Progress Tracking Data
progress_data = {
    'items_processed': 0,
    'items_processed_current_folder': 0,
    'current_folder': '',
    'folders_completed': 0,
    'last_processed_file': ''
}

def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_print(message):
    if debug_mode:
        print(f"{current_time()} - {message}")
        
def initialize_qwen_model():
    global qwen_model, qwen_tokenizer

    torch.manual_seed(1234)
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    qwen_model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def process_folder(folder_path, semaphore):
    file_names = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    for file in file_names:
        if shutdown_flag.is_set():
            break
        image_path = os.path.join(folder_path, file)
        run(image_path, os.path.basename(folder_path), semaphore, folder_path)
    progress_data['folders_completed'] += 1

def run_folders(folder_image_counts, semaphore):
    for folder, _ in folder_image_counts.items():
        process_folder(folder, semaphore)

def process_image_qwen(image_path, prompt_text):
    global qwen_model, qwen_tokenizer
    query = qwen_tokenizer.from_list_format([
        {'image': image_path},
        {'text': prompt_text},
    ])
    response, history = qwen_model.chat(qwen_tokenizer, query=query, history=None)
    return response


def count_files_and_folders(master_folder):
    debug_print("Starting file and folder count")
    folder_image_counts = {}
    
    image_count_in_master = sum(1 for item in os.listdir(master_folder) if item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    if image_count_in_master > 0:
        folder_image_counts[master_folder] = image_count_in_master

    subfolders = [f.path for f in os.scandir(master_folder) if f.is_dir()]
    for folder in subfolders:
        image_count = sum(1 for item in os.listdir(folder) if item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        folder_image_counts[folder] = image_count

    folder_image_counts = {k: v for k, v in folder_image_counts.items() if v > 0}

    total_folders = len(folder_image_counts)
    total_files = sum(folder_image_counts.values())

    debug_print("Finished file and folder count")
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
    debug_print(f"Resized image dimensions: {img_resized.size[0]}x{img_resized.size[1]}")
    buffered = io.BytesIO()
    img_resized.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_image

def save_result_to_file(image_path, result):
    txt_file_path = os.path.splitext(image_path)[0] + '.txt'
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(result)

def handle_prompt(prompt_template, folder_name, image_name, caption_start_template):
    if isinstance(prompt_template, list):
        prompt_template = '\n'.join(prompt_template)
    caption_start = caption_start_template.replace('@folder_name', folder_name).replace('@image_name', image_name)
    prompt_text = prompt_template.replace('@folder_name', folder_name).replace('@image_name', image_name) + '\n' + caption_start
    return prompt_text

def run(image_path, folder_name, semaphore, folder_path):
    with semaphore:
        debug_print(f"Processing image: {image_path}")

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        prompt_text = handle_prompt(prompt_template, folder_name, image_name, caption_start_template)

        if api_model == 'qwen':
            debug_print(f"Sending to Qwen Model...")
            response = process_image_qwen(image_path, prompt_text)

            if response:
                result_text = response
                save_result_to_file(image_path, result_text)
                print("\n" + image_path + "\n" + result_text)
            else:
                print(f"No response for {image_path}")

        elif api_model == 'gemini-pro-vision':
            modified_payload = inject_prompt_into_payload(API_Payload, prompt_text)
            base64_image = image_to_base64(image_path)
            mimeType = 'image/jpeg'
            modified_payload['contents'][0]['parts'][1]['inlineData']['mimeType'] = mimeType
            modified_payload['contents'][0]['parts'][1]['inlineData']['data'] = base64_image

            headers = {'Content-Type': 'application/json'}
            timeout = httpx.Timeout(httpx_timeout_value)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(URI, headers=headers, json=modified_payload)

            if response.status_code == 200:
                result = response.json()
                result_text = result['candidates'][0]['content']['parts'][0]['text']
                save_result_to_file(image_path, result_text)
                print("\n" + image_path + "\n" + result_text)
            else:
                print(f"Error: {response.status_code} - {response.text}")

        elif api_model == 'ooba':
            modified_payload = inject_prompt_into_payload(API_Payload, prompt_text)
            base64_image = image_to_base64(image_path)
            photodescription = f'<img src="data:image/jpeg;base64,{base64_image}">'
            modified_payload['prompt'] = photodescription + "\n" + modified_payload['prompt']

            timeout = httpx.Timeout(httpx_timeout_value)
            with httpx.Client(timeout=timeout) as client:
                response = client.post(URI, json=modified_payload)

            if response.status_code == 200:
                result = response.json()['results'][0]['text'].strip()
                save_result_to_file(image_path, result)
                print("\n" + image_path + "\n" + result)
            else:
                print(f"Error: {response.status_code} - {response.text}")

        # Update progress_data
        progress_data['items_processed'] += 1
        progress_data['items_processed_current_folder'] += 1
        progress_data['last_processed_file'] = image_path
        if progress_data['current_folder'] != folder_path:
            progress_data['current_folder'] = folder_path
            progress_data['items_processed_current_folder'] = 1
        progressBar.update_progress_data(progress_data, folder_image_counts)
        
            
def inject_prompt_into_payload(payload, prompt_text):
    new_payload = payload.copy()
    if '@prompt' in new_payload:
        new_payload = json.dumps(new_payload).replace('@prompt', prompt_text)
    new_payload = json.loads(new_payload)
    return new_payload

def get_folder_from_gui():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected
    
def process_queue():
    progressBar.update_total_progress(progress_data['items_processed'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images in subfolders of a master folder.')
    parser.add_argument('--folder', type=str, help='Path to the master folder containing subfolders with images.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed logging.')

    args = parser.parse_args()
    debug_mode = args.debug

    if api_model == 'qwen':
        debug_print(f"Loading Qwen Model...")
        initialize_qwen_model()

    if args.folder:
        master_folder_path = args.folder
    else:
        master_folder_path = get_folder_from_gui()

    if not master_folder_path:
        print("No folder selected. Exiting.")
        exit()

    total_files, total_folders, folder_image_counts = count_files_and_folders(master_folder_path)
    print(f"Total image files: {total_files}, Total folders with images: {total_folders}")

    progressBar = ProgressBarApp(total_files, total_folders, folder_image_counts)

    all_tasks_done = threading.Event()
    shutdown_flag = threading.Event()

def start_processing():
    try:
        run_folders(folder_image_counts, semaphore)
    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Shutting down.")
        shutdown_flag.set()
    finally:
        all_tasks_done.set()
        progressBar.update_total_progress(total_folders)
        progressBar.close()

# Create and start the processing thread
processing_thread = threading.Thread(target=start_processing)
processing_thread.start()

def process_queue_thread():
    while not all_tasks_done.is_set():
        process_queue()
    progressBar.close()

queue_thread = threading.Thread(target=process_queue_thread)
queue_thread.start()
progressBar.run()