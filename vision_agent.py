"""
Vision Agent - System 2: Hybrid Cloud + Local Vision
Primary: Gemini 2.0 Flash (Vision)
Fallback: Qwen3-VL-4B (Local)
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import re
from typing import List, Dict, Optional
import os

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai library not installed. Run: pip install google-generativeai")


class VisionAgent:
    """
    Hybrid vision-based element finder:
    - Primary: Gemini 2.0 Flash (Fast & Powerful Vision)
    - Fallback: Qwen3-VL-4B (Local)
    """
    
    def __init__(
        self, 
        local_model_path: str,
        gemini_api_key: str = None,
        prefer_local: bool = False
    ):
        """
        Initialize hybrid vision agent
        
        Args:
            local_model_path: Path to local Qwen3-VL-4B model directory
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            prefer_local: If True, use local model first (for testing)
        """
        self.local_model_path = local_model_path
        self.prefer_local = prefer_local
        
        # Gemini setup
        self.gemini_available = False
        self.gemini_model = None
        self.gemini_model_name = "gemini-2.5-flash"
        
        # Local model setup
        self.local_model = None
        self.processor = None
        self.local_model_loaded = False
        
        # Statistics
        self.stats = {
            'gemini_success': 0,
            'gemini_failures': 0,
            'local_success': 0,
            'local_failures': 0
        }
        
        print(f"🤖 Initializing Hybrid VisionAgent")
        print(f"   Primary: Gemini ({self.gemini_model_name})")
        print(f"   Fallback: Qwen3-VL-4B (Local)")
        
        # Setup Gemini
        if not prefer_local:
            self._setup_gemini(gemini_api_key)
        
        print(f"✅ Hybrid vision agent ready!")
        if self.gemini_available:
            print(f"   ⚡ Gemini: Available")
        else:
            print(f"   ⚠️  Gemini: Not available (will use local only)")
        print(f"   🖥️  Local: Ready (lazy load)")
    
    def _setup_gemini(self, api_key: str = None):
        """Setup Gemini API"""
        if not GEMINI_AVAILABLE:
            print("   ⚠️  google-generativeai library not installed")
            return
        
        try:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                print("   ⚠️  No Gemini API key provided")
                print("       Set GEMINI_API_KEY env var or pass to constructor")
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
            
            self.gemini_available = True
            print(f"   ✅ Gemini configured")
            
        except Exception as e:
            print(f"   ⚠️  Gemini setup failed: {e}")
            self.gemini_available = False
    
    def _load_local_model(self):
        """
        Load Qwen3-VL-4B model with 4-bit quantization (lazy loading)
        """
        if self.local_model_loaded:
            print("✅ Local model already loaded")
            return
        
        print("\n" + "=" * 70)
        print("📦 Loading Qwen3-VL-4B model with 4-bit quantization...")
        print("   This may take 30-60 seconds on first load...")
        print("=" * 70)
        
        try:
            # 4-bit quantization config for 8GB VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            print("\n📦 Loading model...")
            # Load model with quantization
            self.local_model = AutoModelForVision2Seq.from_pretrained(
                self.local_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            print("📦 Loading processor...")
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.local_model_path,
                trust_remote_code=True
            )
            
            self.local_model_loaded = True
            
            # Print device info
            device = next(self.local_model.parameters()).device
            print(f"\n✅ Local model loaded successfully!")
            print(f"   Device: {device}")
            print(f"   Quantization: 4-bit")
            
            # Check VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB")
            
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Failed to load local model: {e}")
            print("\nTroubleshooting:")
            print("1. Check model path is correct")
            print("2. Ensure model files are downloaded")
            print("3. Install: pip install transformers torch bitsandbytes accelerate")
            raise
    
    def find_element_by_vision(
        self,
        annotated_image_path: str,
        elements: List[Dict],
        task: str
    ) -> int:
        """
        Find element using hybrid vision: Try Gemini → Fallback to Local
        
        Args:
            annotated_image_path: Path to SoM-annotated screenshot
            elements: List of DOM elements with IDs
            task: Task description (e.g., "click the blue button")
            
        Returns:
            Element index (0-N) or -1 if not found
        """
        if not elements:
            print("❌ No elements provided")
            return -1
        
        # Create prompt once
        prompt = self._create_vision_prompt(elements, task)
        
        # Try primary model first
        if not self.prefer_local and self.gemini_available:
            element_idx = self._try_gemini(annotated_image_path, prompt, elements)
            if element_idx is not None:
                return element_idx
            # If Gemini failed, fallback to local
            print("   🔄 Falling back to local model...")
        
        # Use local model
        element_idx = self._try_local(annotated_image_path, prompt, elements)
        return element_idx if element_idx is not None else -1
    
    def _try_gemini(
        self, 
        image_path: str, 
        prompt: str, 
        elements: List[Dict]
    ) -> Optional[int]:
        """
        Try Gemini API for vision analysis
        
        Returns:
            Element index or None on failure (triggers fallback)
        """
        try:
            print(f"🤖 Analyzing with Gemini ({self.gemini_model_name})...")
            
            # Upload image
            image_file = genai.upload_file(image_path)
            
            # Generate response
            response = self.gemini_model.generate_content(
                [image_file, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                )
            )
            
            llm_response = response.text.strip()
            print(f"🤖 Gemini Response: {llm_response}")
            
            # Parse response
            element_idx = self._parse_response(llm_response, elements)
            
            if element_idx >= 0:
                self.stats['gemini_success'] += 1
                return element_idx
            else:
                self.stats['gemini_failures'] += 1
                return element_idx
            
        except Exception as e:
            # Handle specific errors
            error_msg = str(e).lower()
            
            if 'quota' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
                print(f"   ⚠️  Gemini rate limited")
            elif 'resource_exhausted' in error_msg:
                print(f"   ⚠️  Gemini quota exceeded")
            else:
                print(f"   ⚠️  Gemini error: {e}")
            
            self.stats['gemini_failures'] += 1
            return None  # Trigger fallback
    
    def _try_local(
        self, 
        image_path: str, 
        prompt: str, 
        elements: List[Dict]
    ) -> Optional[int]:
        """
        Try local Qwen3-VL model
        
        Returns:
            Element index or None on failure
        """
        try:
            # Ensure model is loaded
            if not self.local_model_loaded:
                self._load_local_model()
            
            print(f"🤖 Analyzing with Qwen3-VL-4B (local)...")
            
            # Generate response using existing logic
            response = self._generate_local_response(image_path, prompt)
            
            print(f"🤖 Local Response: {response}")
            
            # Parse response
            element_idx = self._parse_response(response, elements)
            
            if element_idx >= 0:
                self.stats['local_success'] += 1
            else:
                self.stats['local_failures'] += 1
            
            return element_idx
            
        except Exception as e:
            print(f"❌ Local model error: {e}")
            self.stats['local_failures'] += 1
            return None
    
    def _create_vision_prompt(self, elements: List[Dict], task: str) -> str:
        """
        Create prompt for vision models (works for both Gemini and Qwen3-VL)
        
        Args:
            elements: List of elements
            task: User task
            
        Returns:
            Formatted prompt string
        """
        # Build element list (keep it concise)
        element_lines = []
        for el in elements[:20]:  # Limit to first 20 for prompt clarity
            x, y = el['bbox']['x'], el['bbox']['y']
            text = el['text'][:30] if el['text'] else '(no text)'
            element_lines.append(
                f"[{el['id']}] {el['tag']} \"{text}\" at ({x}, {y})"
            )
        
        prompt = f"""You are an intelligent web automation agent. 
You are analyzing a screenshot where interactive elements are marked with red tags (0, 1, 2...) very close to the element.

TASK: {task}

VISIBLE ELEMENTS:
{chr(10).join(element_lines)}

INSTRUCTIONS:
1. Analyze the visual appearance of the candidate elements (color, shape, icon).
2. Compare each element against the task description.
3. Eliminate elements that do not match the visual description.
4. Select the single best match.

FORMAT:
Observation: (Briefly describe the visual style of relevant boxes)
Thought: (Why does one box match better than the others?)
Answer: (The box number only)

Example Response:
Observation: Box 0 is a white input field. Box 2 is a blue rectangular button.
Thought: The task asks for a "blue box". Box 2 is blue. Box 0 is white.
Answer: 2

Your Analysis:"""
        
        return prompt
    
    def _generate_local_response(self, image_path: str, prompt: str) -> str:
        """
        Generate response using local Qwen3-VL model
        
        Args:
            image_path: Path to annotated screenshot
            prompt: Formatted prompt
            
        Returns:
            Model response text
        """
        try:
            # Format messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image_path],
                padding=True,
                return_tensors="pt",
            )
            
            # Move to model's device
            inputs = inputs.to(self.local_model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.local_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            print(f"❌ Local generation failed: {e}")
            raise
    
    def _parse_response(self, response: str, elements: List[Dict]) -> int:
        """
        Parse vision model response (works for both Gemini and Qwen3-VL)
        Prioritizes explicit 'Answer: X' and avoids false positive 'NONE's in reasoning text.
        """
        try:
            # 🟢 1. First, look for a strict "Answer: X" pattern
            match = re.search(r'Answer:\s*(\d+)', response, re.IGNORECASE)
            
            if match:
                element_id = int(match.group(1))
                # Validate range immediately
                if 0 <= element_id < len(elements):
                    return element_id
            
            # 🟡 2. If no strict answer, CHECK FOR "NONE" NOW
            if re.search(r'Answer:\s*NONE', response, re.IGNORECASE) or response.strip().upper() == 'NONE':
                print("   Model returned NONE")
                return -1
            
            # 🟠 3. Fallback: Last number in text
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                element_id = int(numbers[-1])
                if 0 <= element_id < len(elements):
                    return element_id

            print(f"⚠️  No valid ID found in response: {response}")
            return -1
                
        except Exception as e:
            print(f"⚠️  Failed to parse response: {e}")
            return -1
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total_requests = sum(self.stats.values())
        
        return {
            'total_requests': total_requests,
            'gemini': {
                'success': self.stats['gemini_success'],
                'failures': self.stats['gemini_failures'],
                'rate': self.stats['gemini_success'] / max(1, self.stats['gemini_success'] + self.stats['gemini_failures']) * 100
            },
            'local': {
                'success': self.stats['local_success'],
                'failures': self.stats['local_failures'],
                'rate': self.stats['local_success'] / max(1, self.stats['local_success'] + self.stats['local_failures']) * 100
            }
        }
    
    def print_statistics(self):
        """Print usage statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 HYBRID VISION AGENT STATISTICS")
        print("=" * 60)
        print(f"Total requests: {stats['total_requests']}")
        print(f"\n⚡ Gemini API:")
        print(f"   Success: {stats['gemini']['success']}")
        print(f"   Failures: {stats['gemini']['failures']}")
        print(f"   Success rate: {stats['gemini']['rate']:.1f}%")
        print(f"\n🖥️  Local Qwen3-VL:")
        print(f"   Success: {stats['local']['success']}")
        print(f"   Failures: {stats['local']['failures']}")
        print(f"   Success rate: {stats['local']['rate']:.1f}%")
        print("=" * 60)
    
    def unload_model(self):
        """
        Unload local model from memory to free VRAM
        """
        if not self.local_model_loaded:
            print("⚠️  Local model not loaded")
            return
        
        try:
            print("\n🔄 Unloading local model...")
            
            # Delete model and processor
            del self.local_model
            del self.processor
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.local_model = None
            self.processor = None
            self.local_model_loaded = False
            
            print("✅ Local model unloaded, VRAM freed")
            
        except Exception as e:
            print(f"⚠️  Error unloading model: {e}")


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    from som_annotator import SoMAnnotator
    import os
    
    print("=" * 70)
    print("HYBRID VISION AGENT TEST - System 2")
    print("Primary: Gemini 2.0 Flash | Fallback: Qwen3-VL-4B")
    print("=" * 70)
    
    # Configuration
    print("\n📋 Configuration:")
    
    # Setup
    LOCAL_MODEL_PATH = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    GEMINI_API_KEY = ""  # Replace or use env var
    
    if GEMINI_API_KEY == "":
        GEMINI_API_KEY = None  # Will check env var
    
    # Initialize components
    extractor = DOMExtractor()
    annotator = SoMAnnotator()
    vision_agent = VisionAgent(
        local_model_path=LOCAL_MODEL_PATH,
        gemini_api_key=GEMINI_API_KEY
    )
    
    try:
        # User input
        print("\n📋 Test Setup:")
        url = input("Enter URL (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        task = input("Enter VISUAL task (e.g., 'click the blue button', 'click search icon'): ").strip()
        
        if not task:
            print("❌ No task provided")
            exit(1)
        
        print("\n" + "=" * 70)
        
        # Start browser
        print(f"\n🌐 Starting browser and navigating to {url}...")
        if not extractor.start_browser(headless=False):
            exit(1)
        
        if not extractor.navigate(url):
            exit(1)
        
        # Extract elements
        print("\n🔍 Extracting DOM elements...")
        elements = extractor.extract_elements()
        
        if not elements:
            print("❌ No elements found")
            exit(1)
        
        print(f"✅ Found {len(elements)} interactive elements")
        
        # Take screenshot
        print("\n📸 Taking screenshot...")
        screenshot = extractor.take_screenshot()
        
        if not screenshot:
            print("❌ Screenshot failed")
            exit(1)
        
        # Create annotated image
        print("\n🎨 Creating SoM-annotated screenshot...")
        annotated_img, filtered_elements = annotator.filter_and_annotate(
            screenshot,
            elements,
            max_elements=15
        )
        
        # Save annotated image
        annotated_path = "test_hybrid_som_annotated.png"
        with open(annotated_path, 'wb') as f:
            f.write(annotated_img)
        
        print(f"✅ Saved annotated screenshot: {annotated_path}")
        
        # Run Hybrid System 2 analysis
        print("\n" + "=" * 70)
        print("🤖 HYBRID SYSTEM 2: Vision Analysis")
        print("=" * 70)
        print(f"Task: '{task}'")
        
        element_idx = vision_agent.find_element_by_vision(
            annotated_path,
            filtered_elements,
            task
        )
        
        print("\n" + "=" * 70)
        
        if element_idx >= 0:
            element = filtered_elements[element_idx]
            real_id = element['id']
            
            print("✅ SYSTEM 2 SUCCESS!")
            print("=" * 70)
            print(f"\n🎯 Vision Model Selected Element:")
            print(f"   Box Index: {element_idx}")
            print(f"   Real ID:   {real_id}")
            print(f"   Tag:       {element['tag']}")
            print(f"   Type:      {element['type']}")
            print(f"   Text:      {element['text'][:60] if element['text'] else '(no text)'}")
            print(f"   Position:  ({element['bbox']['x']}, {element['bbox']['y']})")
            print(f"   Size:      {element['bbox']['w']}x{element['bbox']['h']}")
            
            if element['html_id']:
                print(f"   HTML ID:   {element['html_id']}")
            
            # Optional click
            print("\n" + "=" * 70)
            click_choice = input("🖱️  Click this element? (y/n): ").strip().lower()
            
            if click_choice == 'y':
                print("\n🖱️  Clicking element...")
                if extractor.click_by_id(real_id, elements):
                    print("✅ Click executed successfully!")
                    print("   Waiting 2 seconds to see result...")
                    import time
                    time.sleep(2)
                else:
                    print("❌ Click failed")
        
        else:
            print("❌ SYSTEM 2 FAILED")
            print("=" * 70)
            print("\n📊 Analysis:")
            print(f"   - Vision models could not identify element for: '{task}'")
            print(f"   - Annotated screenshot: {annotated_path}")
            print("\n💡 Suggestions:")
            print("   - Check if element is visible in screenshot")
            print("   - Try more descriptive visual task")
            print("   - Verify SoM annotations are correct")
        
        # Show statistics
        vision_agent.print_statistics()
        
        print("\n" + "=" * 70)
        input("⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Unload model
        print("\n🔄 Cleaning up...")
        vision_agent.unload_model()
        extractor.close()
        print("\n✅ Test complete!")