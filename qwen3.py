import torch
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
local_model_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
image_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\debug_som_marked_screenshot.png"
prompt = "Find the bounding box number of the Three traingles logo."

# ============================================
# 1. VERIFY GPU AVAILABILITY
# ============================================
print("=" * 60)
print("SYSTEM CHECK")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
 print(f"CUDA Version: {torch.version.cuda}")
 print(f"GPU: {torch.cuda.get_device_name(0)}")
 print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
 print(" WARNING: CUDA not available! Model will run on CPU (very slow)")
print()

# ============================================
# 2. CHECK MODEL FILES
# ============================================
print("=" * 60)
print("CHECKING MODEL FILES")
print("=" * 60)
if not os.path.exists(local_model_path):
 raise FileNotFoundError(f"Model path not found: {local_model_path}")

required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json']
model_files = os.listdir(local_model_path)
print(f"Files in model directory: {len(model_files)} files")

missing_files = [f for f in required_files if not any(f in mf for mf in model_files)]
if missing_files:
 print(f" WARNING: Some files might be missing: {missing_files}")
print()

# ============================================
# 3. CONFIGURE QUANTIZATION (4-bit for RTX 4060)
# ============================================
print("=" * 60)
print("CONFIGURING MODEL")
print("=" * 60)

try:
 # For 8GB VRAM (RTX 4060), 4-bit quantization is essential
 bnb_config = BitsAndBytesConfig(
 load_in_4bit=True,
 bnb_4bit_quant_type="nf4",
 bnb_4bit_compute_dtype=torch.float16,
 bnb_4bit_use_double_quant=True, # Extra memory saving
 )
 print(" 4-bit quantization configured")
 use_quantization = True
except Exception as e:
 print(f" Quantization config failed: {e}")
 print("Proceeding without quantization (will use more VRAM)")
 bnb_config = None
 use_quantization = False

print()

# ============================================
# 4. LOAD MODEL
# ============================================
print("=" * 60)
print("LOADING MODEL (This may take 1-2 minutes...)")
print("=" * 60)

try:
 # Method 1: Use AutoModelForVision2Seq (recommended)
 model = AutoModelForVision2Seq.from_pretrained(
 local_model_path,
 quantization_config=bnb_config if use_quantization else None,
 device_map="auto",
 trust_remote_code=True,
 torch_dtype=torch.float16 if not use_quantization else None,
 low_cpu_mem_usage=True,
 )
 print(" Model loaded successfully using AutoModelForVision2Seq")
 
except Exception as e:
 print(f" AutoModelForVision2Seq failed: {str(e)[:200]}")
 print("\nTrying alternative loading method...")
 
 try:
 # Method 2: Use Qwen-specific class
 from transformers import Qwen2VLForConditionalGeneration
 model = Qwen2VLForConditionalGeneration.from_pretrained(
 local_model_path,
 quantization_config=bnb_config if use_quantization else None,
 device_map="auto",
 trust_remote_code=True,
 torch_dtype=torch.float16 if not use_quantization else None,
 )
 print(" Model loaded using Qwen2VLForConditionalGeneration")
 except Exception as e2:
 print(f" Alternative method also failed: {str(e2)[:200]}")
 raise RuntimeError("All model loading methods failed. See errors above.")

print()

# ============================================
# 5. LOAD PROCESSOR
# ============================================
print("=" * 60)
print("LOADING PROCESSOR")
print("=" * 60)

processor = AutoProcessor.from_pretrained(
 local_model_path,
 trust_remote_code=True
)
print(" Processor loaded successfully")
print()

# ============================================
# 6. PREPARE INPUT
# ============================================
print("=" * 60)
print("PREPARING INPUT")
print("=" * 60)

# Verify image exists
if not os.path.exists(image_path):
 raise FileNotFoundError(f"Image not found: {image_path}")

# Load and verify image
try:
 img = Image.open(image_path)
 print(f" Image loaded: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
except Exception as e:
 raise ValueError(f"Failed to load image: {e}")

# Format conversation
messages = [
 {
 "role": "user",
 "content": [
 {"type": "image", "image": image_path},
 {"type": "text", "text": prompt},
 ],
 }
]
print(f" Prompt: {prompt}")
print()

# ============================================
# 7. PROCESS & GENERATE
# ============================================
print("=" * 60)
print("GENERATING RESPONSE")
print("=" * 60)

try:
 # Try using qwen_vl_utils if available
 from qwen_vl_utils import process_vision_info
 
 text = processor.apply_chat_template(
 messages, 
 tokenize=False, 
 add_generation_prompt=True
 )
 image_inputs, video_inputs = process_vision_info(messages)
 
 inputs = processor(
 text=[text],
 images=image_inputs,
 videos=video_inputs,
 padding=True,
 return_tensors="pt",
 )
 print(" Using qwen_vl_utils for processing")
 
except ImportError:
 # Fallback method
 print(" qwen_vl_utils not found, using standard processing")
 
 text = processor.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True
 )
 
 inputs = processor(
 text=[text],
 images=[image_path],
 padding=True,
 return_tensors="pt",
 )

# Move to GPU
inputs = inputs.to(model.device)
print(f" Inputs moved to: {model.device}")

# Generate
print("Generating (this may take 10-30 seconds)...")
with torch.no_grad():
 generated_ids = model.generate(
 **inputs,
 max_new_tokens=128,
 do_sample=False, # Deterministic output
 temperature=None,
 top_p=None,
 )

# Decode
generated_ids_trimmed = [
 out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
 generated_ids_trimmed,
 skip_special_tokens=True,
 clean_up_tokenization_spaces=False
)

# ============================================
# 8. OUTPUT
# ============================================
print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print(f"\n{output_text[0]}\n")
print("=" * 60)

# Optional: Show memory usage
if torch.cuda.is_available():
 print(f"\nGPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
 print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")