"""
DOM Text Agent - System 1: Hybrid Cloud + Local
Primary: Groq (Llama 3.3 70B Versatile)
Fallback: Llama 3.2:3B (Local Ollama)
"""

import ollama
import re
from typing import List, Dict, Optional
import os

# Groq imports
try:
    from groq import Groq, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  groq library not installed. Run: pip install groq")


class DOMTextAgent:
    """
    Hybrid text-based DOM automation:
    - Primary: Groq API (Llama 3.3 70B - Fast & Accurate)
    - Fallback: Local Ollama (when rate limited)
    """
    
    def __init__(
        self, 
        local_model: str = "llama3.2:3b",
        groq_api_key: str = None,
        prefer_local: bool = False
    ):
        """
        Initialize hybrid agent with cloud + local models
        
        Args:
            local_model: Ollama model name for fallback
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            prefer_local: If True, use local model first (for testing)
        """
        self.local_model = local_model
        self.prefer_local = prefer_local
        self.groq_available = False
        self.groq_client = None
        # Selected Groq model: Best balance of speed/accuracy
        self.groq_model_name = "llama-3.3-70b-versatile"
        
        # Statistics
        self.stats = {
            'groq_success': 0,
            'groq_failures': 0,
            'local_success': 0,
            'local_failures': 0
        }
        
        print(f"🤖 Initializing Hybrid DOMTextAgent")
        print(f"   Primary: Groq ({self.groq_model_name})")
        print(f"   Fallback: {local_model} (Local)")
        
        # Setup Groq
        if not prefer_local:
            self._setup_groq(groq_api_key)
        
        # Setup Ollama
        self._setup_ollama()
        
        print(f"✅ Hybrid agent ready!")
        if self.groq_available:
            print(f"   ⚡ Groq: Available")
        else:
            print(f"   ⚠️  Groq: Not available (will use local only)")
        print(f"   🖥️  Local: Ready")
    
    def _setup_groq(self, api_key: str = None):
        """Setup Groq API"""
        if not GROQ_AVAILABLE:
            print("   ⚠️  Groq library not installed")
            return
        
        try:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('GROQ_API_KEY')
            
            if not api_key:
                print("   ⚠️  No Groq API key provided")
                print("       Set GROQ_API_KEY env var or pass to constructor")
                return
            
            # Initialize Client
            self.groq_client = Groq(api_key=api_key)
            
            self.groq_available = True
            print(f"   ✅ Groq configured")
            
        except Exception as e:
            print(f"   ⚠️  Groq setup failed: {e}")
            self.groq_available = False
    
    def _setup_ollama(self):
        """Setup local Ollama"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if not any(self.local_model in name for name in model_names):
                print(f"   ⚠️  Local model '{self.local_model}' not found")
                print(f"       Run: ollama pull {self.local_model}")
            else:
                print(f"   ✅ Local model ready")
                
        except Exception as e:
            print(f"   ⚠️  Ollama check failed: {e}")
    
    def find_element_for_task(
        self, 
        elements: List[Dict], 
        task: str, 
        page_info: Dict
    ) -> int:
        """
        Find element using hybrid approach: Try Groq → Fallback to Local
        
        Args:
            elements: List of DOM elements
            task: User task description
            page_info: Page information
            
        Returns:
            Element ID or -1 if not found
        """
        if not elements:
            print("❌ No elements to analyze")
            return -1
        
        # Create prompt once
        prompt = self._create_prompt(elements, task, page_info)
        
        # Try primary model first
        if not self.prefer_local and self.groq_available:
            element_id = self._try_groq(prompt, elements)
            if element_id is not None:
                return element_id
            # If Groq failed, fallback to local
            print("   🔄 Falling back to local model...")
        
        # Use local model
        element_id = self._try_local(prompt, elements)
        return element_id if element_id is not None else -1
    
    def _try_groq(self, prompt: str, elements: List[Dict]) -> Optional[int]:
        """
        Try Groq API
        
        Returns:
            Element ID or None on failure (triggers fallback)
        """
        try:
            print(f"🤖 Analyzing with Groq ({self.groq_model_name})...")
            
            # Generate with Groq
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.groq_model_name,
                temperature=0.1,
                max_tokens=150,
            )
            
            llm_response = chat_completion.choices[0].message.content.strip()
            print(f"🤖 Groq Response: {llm_response}")
            
            # Parse response
            element_id = self._parse_response(llm_response, elements)
            
            if element_id >= 0:
                self.stats['groq_success'] += 1
                return element_id
            else:
                self.stats['groq_failures'] += 1
                return element_id
            
        except Exception as e:
            # Handle specific Groq errors
            error_msg = str(e).lower()
            
            if 'rate limit' in error_msg or '429' in error_msg:
                print(f"   ⚠️  Groq rate limited (TPM/RPM exceeded)")
            elif 'context_length_exceeded' in error_msg:
                print(f"   ⚠️  Groq context length exceeded")
            else:
                print(f"   ⚠️  Groq error: {e}")
            
            self.stats['groq_failures'] += 1
            return None  # Trigger fallback
    
    def _try_local(self, prompt: str, elements: List[Dict]) -> Optional[int]:
        """
        Try local Ollama model
        
        Returns:
            Element ID or None on failure
        """
        try:
            print(f"🤖 Analyzing with {self.local_model}...")
            
            # Query Ollama
            response = ollama.generate(
                model=self.local_model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.1,
                    'num_predict': 150,
                }
            )
            
            llm_response = response['response'].strip()
            print(f"🤖 Local Response: {llm_response}")
            
            # Parse response
            element_id = self._parse_response(llm_response, elements)
            
            if element_id >= 0:
                self.stats['local_success'] += 1
            else:
                self.stats['local_failures'] += 1
            
            return element_id
            
        except Exception as e:
            print(f"❌ Local model error: {e}")
            self.stats['local_failures'] += 1
            return None
    
    def _create_prompt(
        self, 
        elements: List[Dict], 
        task: str, 
        page_info: Dict
    ) -> str:
        """
        Create optimized prompt with visual guardrails
        """
        # Limit to first 50 elements (Groq has limits, keep context efficient)
        limited_elements = elements[:50]
        
        element_lines = []
        for el in limited_elements:
            x, y = el['bbox']['x'], el['bbox']['y']
            text = el['text'][:40] if el['text'] else '(no text)'
            html_info = f"id='{el['html_id']}'"
            
            line = f"[{el['id']}] {el['tag']} \"{text}\" {html_info} type={el['type']} at ({x}, {y})"
            element_lines.append(line)
        
        prompt = f"""PAGE: {page_info.get('title', 'Unknown')}
TASK: {task}

INTERACTIVE ELEMENTS:
{chr(10).join(element_lines)}

INSTRUCTIONS:
1. Identify the TARGET element that matches the TASK "{task}".
2. SEARCH strategies:
   - Text Match: Does the element text match the task?
   - Attribute Match: Does the ID or Class contain the task words?
   
3. 🔴 VISUAL GUARDRAIL (CRITICAL): 
   - You are a TEXT-ONLY system. You cannot see colors, icons, or shapes.
   - If the task requires a visual match (e.g., "click the red button"), but the element's ID/Class does not contain "red", return "ID: NONE".
   - DO NOT GUESS based on common sense.
   - Returning "NONE" allows our Vision System to take over.

4. If unsure, return "ID: NONE".

FORMAT:
REASONING: (Step-by-step thoughts. If task is visual/color and no text match found, state "Task requires vision" and return NONE)
ID: (Number or NONE)

Your response:"""
        
        return prompt
    
    def _parse_response(self, response: str, elements: List[Dict]) -> int:
        """
        Parse LLM response to extract element ID
        Works for both Groq and Ollama responses
        """
        try:
            # Use regex to find ID
            match = re.search(r'ID:\s*(\d+|NONE)', response, re.IGNORECASE)
            
            if not match:
                # Fallback: look for any number
                numbers = re.findall(r'\b\d+\b', response)
                if not numbers:
                    return -1
                element_id = int(numbers[-1])
            else:
                id_str = match.group(1).upper()
                if id_str == 'NONE':
                    return -1
                element_id = int(id_str)
            
            # Validate range
            if 0 <= element_id < len(elements):
                return element_id
            return -1
                
        except Exception as e:
            print(f"⚠️ Parsing error: {e}")
            return -1
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total_requests = sum(self.stats.values())
        
        return {
            'total_requests': total_requests,
            'groq': {
                'success': self.stats['groq_success'],
                'failures': self.stats['groq_failures'],
                'rate': self.stats['groq_success'] / max(1, self.stats['groq_success'] + self.stats['groq_failures']) * 100
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
        print("📊 HYBRID AGENT STATISTICS")
        print("=" * 60)
        print(f"Total requests: {stats['total_requests']}")
        print(f"\n⚡ Groq API:")
        print(f"   Success: {stats['groq']['success']}")
        print(f"   Failures: {stats['groq']['failures']}")
        print(f"   Success rate: {stats['groq']['rate']:.1f}%")
        print(f"\n🖥️  Local Ollama:")
        print(f"   Success: {stats['local']['success']}")
        print(f"   Failures: {stats['local']['failures']}")
        print(f"   Success rate: {stats['local']['rate']:.1f}%")
        print("=" * 60)
    
    def explain_selection(self, element: Dict) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        explanation_parts.append(f"Selected {element['tag']} element")
        
        if element['text']:
            explanation_parts.append(f"with text '{element['text'][:50]}'")
        
        if element['type'] != 'button':
            explanation_parts.append(f"(type: {element['type']})")
        
        x, y = element['bbox']['x'], element['bbox']['y']
        position_desc = []
        if y < 100:
            position_desc.append("top")
        elif y > 800:
            position_desc.append("bottom")
        if x < 200:
            position_desc.append("left")
        elif x > 1500:
            position_desc.append("right")
        
        if position_desc:
            explanation_parts.append(f"at {'-'.join(position_desc)} of page")
        
        if element['html_id']:
            explanation_parts.append(f"(id='{element['html_id']}')")
        
        return " ".join(explanation_parts)


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    
    print("=" * 70)
    print("HYBRID DOM TEXT AGENT - System 1 Test")
    print("Primary: Groq (Llama 3.3 70B) | Fallback: Local Ollama")
    print("=" * 70)
    
    # Setup - REPLACE THIS WITH YOUR ACTUAL GROQ KEY
    GROQ_API_KEY = "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"  # Put your key here or use env var
    
    # Initialize
    extractor = DOMExtractor()
    if GROQ_API_KEY and GROQ_API_KEY != "":
        agent = DOMTextAgent(groq_api_key=GROQ_API_KEY)
    else:
        # Check env var
        agent = DOMTextAgent()
    
    try:
        # User input
        print("\n📋 Setup:")
        url = input("Enter URL (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        task = input("Enter task (e.g., 'click search button'): ").strip()
        
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
        
        # Extract DOM elements
        print("\n🔍 Extracting DOM elements...")
        elements = extractor.extract_elements()
        page_info = extractor.get_page_info()
        
        if not elements:
            print("❌ No elements found on page")
            exit(1)
        
        print(f"✅ Found {len(elements)} interactive elements")
        print(f"📄 Page: {page_info['title']}")
        
        # Run System 1 analysis
        print("\n" + "=" * 70)
        print(f"🤖 HYBRID SYSTEM 1 ANALYSIS")
        print(f"Task: '{task}'")
        print("=" * 70)
        
        element_id = agent.find_element_for_task(elements, task, page_info)
        
        print("\n" + "=" * 70)
        
        if element_id >= 0:
            element = elements[element_id]
            
            print("✅ SYSTEM 1 SUCCESS!")
            print("=" * 70)
            print(f"\n🎯 Selected Element [{element_id}]:")
            print(f"   Tag:      {element['tag']}")
            print(f"   Type:     {element['type']}")
            print(f"   Text:     {element['text'][:60] if element['text'] else '(no text)'}")
            print(f"   Position: ({element['bbox']['x']}, {element['bbox']['y']})")
            
            if element['html_id']:
                print(f"   HTML ID:  {element['html_id']}")
            
            # Optional: Click
            print("\n" + "=" * 70)
            click_choice = input("🖱️  Click this element? (y/n): ").strip().lower()
            
            if click_choice == 'y':
                print("\n🖱️  Clicking element...")
                if extractor.click_by_id(element_id, elements):
                    print("✅ Click executed!")
                    import time
                    time.sleep(2)
            
        else:
            print("❌ SYSTEM 1 FAILED")
            print("=" * 70)
            print(f"\n   No matching element found for: '{task}'")
            print("\n🔄 Next Step:")
            print("   → Would trigger SYSTEM 2 (Vision)")
        
        # Show statistics
        agent.print_statistics()
        
        print("\n" + "=" * 70)
        input("⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()
        print("\n✅ Test complete!")