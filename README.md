# SDXL-LoRA🎣
Using LoRA to fine tune SDXL v1.0 + Refiner Image Gen model. 

## SDXL-LoRA Fine-Tuning Notebook

This notebook is part of the **Dynamic Image Synthesis** project, where we automate the generation and recoloring of fishing lure assets. It enables:

* **Generate existing lures in photographic contexts**: Create realistic renders of lure designs in various settings (e.g., held in hand, against natural backgrounds).
* **Recolor existing lures**: Apply new color schemes to base models (for example, swapping a firetiger pattern for chartreuse pearl) using LoRA adapters.

Below is a brief overview of each cell and its purpose.

1. **Open in Colab Badge**

   * Adds a clickable badge to launch the notebook directly in Google Colab.

2. **Environment Setup**

   * Installs required Python packages and GitHub versions:

     * `diffusers[torch]`, `transformers`, `accelerate`, `peft`, `safetensors`, `huggingface_hub`
     * Clones the `diffusers` repository and installs extras for text-to-image examples
     * Upgrades `diffusers` and `peft` directly from their `main` branches
     * Cleans up any conflicting installations (e.g., `xformers`)

3. **HF Token Retrieval (Connection)**

   * Uses Colab's `userdata` API to fetch and display a stored Hugging Face token (`worktoken`).

4. **HF Token & Login**

   * Imports `login` from `huggingface_hub` and logs in with the retrieved token.

5. **Google Drive Mount**

   * Mounts the user's Google Drive to access datasets and save models under `/content/drive`.

6. **Load SDXL Base & Refiner**

   * Loads the StabilityAI SDXL base and refiner pipelines into GPU memory for subsequent training and inference.

7. **Attach LoRA Adapter**

   * Configures a LoRA adapter and attaches it to the base UNet for fine-tuning.

8. **Check for Source Images**

   * Scans a standardized folder on Drive for input images and reports the count.

9. **Generate Metadata File**

   * Constructs a `metadata.jsonl` file mapping each image filename to its training caption, written to the dataset directory.

10. **Free Refiner & Clear GPU Cache**

    * Deletes the refiner pipeline from memory and clears the CUDA cache to free resources for training.

11. **Launch Fine-Tuning**

    * Runs the LoRA fine-tuning script (`train_text_to_image_lora_sdxl.py`) via `accelerate launch`, specifying data paths, hyperparameters, and checkpoint options.

12. **Inference with Stacked LoRAs**

    * Loads two trained LoRA weight files onto the base model, combines them, and generates a demo image saved back to Drive.

---

*This README is intended as a high-level guide. For full details, refer to the code cells in the notebook.*
