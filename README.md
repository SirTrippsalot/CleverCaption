# CleverCaption - LLM based batch Image Captioning Tool

CleverCaption is a Python tool that processes images in subfolders of a given directory, generates captions using a remote API, and saves the results in text files corresponding to each image.

## Features

- Processes multiple images in bulk from nested folder structures.
- Utilizes a remote API to generate captions based on image content.
- Presents a progress UI using Tkinter.
- Handles concurrent API requests and manages timeouts.
- Converts images to base64 for API submission.
- Saves caption results as `.txt` files alongside images.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed
- Required Python packages installed:
  - requests
  - Pillow
  - httpx
  - asyncio
  - tkinter

## Installation

To install CleverCaption, follow these steps:

1. Clone or download the repository to your local machine.
2. Use pip to install the necessary packages:
   ```sh
   pip install requests Pillow httpx asyncio tkinter
   ```
   
Alternatively if conda is installed simply run the install.bat to create a conda environment.

## Usage

To use CleverCaption, follow these steps:

1. Ensure your images are organized into subfolders within a master folder.
2. Run the main script with the path to the master folder:
   ```sh
   python CleverCaption.py --folder "path/to/your/master/folder"
   ```
   If you don't provide a folder path, a GUI will prompt you to select a folder.

3. The progress of the captioning process can be monitored through the GUI that pops up.

Alternatively if using the conda install method, simply double-click run.bat.

## oobabooga text-generation-webui

Follow these steps to configure and use CleverCaption with OOBA BOOGA WebUI and the LLAVA multimodal model:

### Step 1: OOBA BOOGA WebUI Configuration
- Set up the OOBA BOOGA WebUI from its [GitHub repository](https://github.com/oobabooga/text-generation-webui).
- Run OOBA BOOGA with the multimodal model using the following switches:
  ```
  --multimodal-pipeline llava-llama-2-13b --extensions multimodal
  ```

### Step 2: LLAVA Model Configuration
- Access the LLAVA model on [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.5-13b).
- Modify the `config.json` file within the LLAVA model directory:
  - Change `"model_type": "llava"` to `"model_type": "llama"`.

---

Ensure all configurations are set before running the tool. The instructions above should work alongside the provided CleverCaption documentation and OOBA BOOGA's guidelines.

## Contribution

Contributions to CleverCaption are welcome. If you have a suggestion that would make this better, please fork the repo and create a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.