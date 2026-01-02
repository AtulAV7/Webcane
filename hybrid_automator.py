"""
Hybrid Automator - Main Orchestrator
Sequential fallback: System 1 (DOM) → System 2 (Vision)
"""

from dom_extractor import DOMExtractor
from dom_text_agent import DOMTextAgent
from som_annotator import SoMAnnotator
from vision_agent import VisionAgent
from typing import List, Dict
import time
import os

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class HybridAutomator:
    """
    Hybrid web automation with two-system approach:
    - System 1: Fast DOM + Text LLM analysis
    - System 2: Vision-based fallback with SoM annotations
    """
    
    def __init__(
        self, 
        vision_model_path: str = None,
        groq_api_key: str = None,
        gemini_api_key: str = None
    ):
        """
        Initialize all automation components
        
        Args:
            vision_model_path: Path to Qwen3-VL model (optional, lazy load)
            groq_api_key: Groq API Key (overrides env var)
            gemini_api_key: Gemini API Key (overrides env var)
        """
        print("=" * 70)
        print("HYBRID AUTOMATOR - Initializing Two-System Architecture")
        print("=" * 70)
        
        # Save config for lazy loading
        self.vision_model_path = vision_model_path
        self.gemini_api_key = gemini_api_key
        
        # Initialize components
        print("\n📦 Initializing components...")
        self.extractor = DOMExtractor()
        
        # ✅ PASS GROQ KEY HERE
        # If groq_api_key is None, the agent will look for GROQ_API_KEY in env (loaded from .env)
        self.text_agent = DOMTextAgent(groq_api_key=groq_api_key)
        self.annotator = SoMAnnotator()
        
        # Vision agent (lazy load)
        self.vision_agent = None
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'system1_success': 0,
            'system2_success': 0,
            'failures': 0
        }
        
        print("✅ Hybrid Automator initialized")
        print("\n🎯 Strategy:")
        print("   1. Always try System 1 (DOM + Text) first")
        print("   2. Fall back to System 2 (Vision) if needed")
        print("=" * 70 + "\n")
    
    def _ensure_vision_agent(self):
        """Lazy load vision agent when needed"""
        if self.vision_agent is None:
            if not self.vision_model_path:
                # Ask for model path
                print("\n⚠️  Vision model path not set.")
                model_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
                if not model_path:
                    raise ValueError("Vision model path required for System 2")
                self.vision_model_path = model_path
            
            print(f"\n🤖 Loading Vision Agent (first use)...")
            self.vision_agent = VisionAgent(self.vision_model_path)
    
    def start(self, url: str, headless: bool = False) -> bool:
        """
        Start browser session and navigate to URL
        
        Args:
            url: Target URL
            headless: Run browser in headless mode
            
        Returns:
            True on success, False on failure
        """
        print(f"\n🌐 Starting browser session...")
        print(f"   URL: {url}")
        print(f"   Headless: {headless}")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Start browser
        if not self.extractor.start_browser(headless=headless):
            print("❌ Failed to start browser")
            return False
        
        # Navigate
        if not self.extractor.navigate(url):
            print("❌ Failed to navigate")
            return False
        
        page_info = self.extractor.get_page_info()
        print(f"\n✅ Ready!")
        print(f"   Title: {page_info['title']}")
        print(f"   URL: {page_info['url']}")
        
        return True
    
    def execute_task(self, task: str) -> Dict:
        """
        Execute a task using sequential fallback strategy
        
        Args:
            task: Task description (e.g., "click search button")
            
        Returns:
            Result dict with success status and method used
        """
        print("\n" + "=" * 70)
        print(f"🎯 TASK: {task}")
        print("=" * 70)
        
        self.stats['total_attempts'] += 1
        start_time = time.time()
        
        # STEP 1: Extract DOM elements
        print("\n📊 STEP 1: Extracting DOM elements...")
        elements = self.extractor.extract_elements()
        page_info = self.extractor.get_page_info()
        
        if not elements:
            print("❌ No interactive elements found on page")
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'failed',
                'element_id': -1,
                'element': None,
                'reason': 'No elements found'
            }
        
        print(f"✅ Found {len(elements)} interactive elements")
        
        # STEP 2: Try System 1 (DOM + Text Agent)
        print("\n" + "-" * 70)
        print("🚀 STEP 2: SYSTEM 1 - DOM Text Analysis")
        print("-" * 70)
        print("   Strategy: Fast text-based element matching")
        print(f"   Analyzing: {min(len(elements), 50)} elements")
        
        system1_start = time.time()
        element_id = self.text_agent.find_element_for_task(elements, task, page_info)
        system1_time = time.time() - system1_start
        
        if element_id >= 0:
            element = elements[element_id]
            print(f"\n✅ SYSTEM 1 SUCCESS! (⚡ {system1_time:.2f}s)")
            print(f"   Found: [{element_id}] {element['tag']} \"{element['text'][:40]}\"")
            
            # Click the element
            print(f"\n🖱️  Clicking element...")
            self.extractor.click_by_id(element_id, elements)
            
            self.stats['system1_success'] += 1
            elapsed = time.time() - start_time
            
            print(f"✅ Task completed in {elapsed:.2f}s")
            
            return {
                'success': True,
                'method': 'dom',
                'element_id': element_id,
                'element': element,
                'time': elapsed
            }
        
        # STEP 3: System 1 failed - Activate System 2
        print(f"\n⚠️  SYSTEM 1 FAILED (took {system1_time:.2f}s)")
        print("   Reason: No matching element found via text analysis")
        
        print("\n" + "-" * 70)
        print("🔄 STEP 3: SYSTEM 2 - Vision Fallback")
        print("-" * 70)
        print("   Strategy: Visual analysis with SoM annotations")
        
        try:
            # Ensure vision agent is loaded
            self._ensure_vision_agent()
            
            # Take screenshot
            print("\n📸 Taking screenshot...")
            screenshot = self.extractor.take_screenshot()
            
            if not screenshot:
                print("❌ Screenshot failed")
                self.stats['failures'] += 1
                return {
                    'success': False,
                    'method': 'failed',
                    'element_id': -1,
                    'element': None,
                    'reason': 'Screenshot failed'
                }
            
            # Annotate with SoM
            print("🎨 Creating SoM annotations...")
            annotated_img, filtered_elements = self.annotator.filter_and_annotate(
                screenshot,
                elements,
                max_elements=60  # Limit for vision clarity
            )
            
            # Save annotated image temporarily
            annotated_path = "temp_som_annotated.png"
            with open(annotated_path, 'wb') as f:
                f.write(annotated_img)
            
            print(f"✅ Annotated {len(filtered_elements)} elements")
            
            # Analyze with vision
            print("\n🤖 Analyzing with Vision LLM...")
            system2_start = time.time()
            
            # NEW WAY: Vision returns the BOX NUMBER (Index 0, 1, 2...)
            vision_index = self.vision_agent.find_element_by_vision(
                annotated_path,
                filtered_elements,
                task
            )
            system2_time = time.time() - system2_start
            
            # Check if the index is valid (within the range of our filtered list)
            if vision_index >= 0 and vision_index < len(filtered_elements):
                
                # 1. Get the actual element object from the filtered list using the index
                target_element = filtered_elements[vision_index]
                
                # 2. Extract the REAL unique ID (e.g., 42, 105) from that element
                real_id = target_element['id']
                
                print(f"\n✅ SYSTEM 2 SUCCESS! (🐌 {system2_time:.2f}s)")
                print(f"   Visual Selection: Box #{vision_index}")
                print(f"   Mapped to Real ID: {real_id}")
                print(f"   Tag: {target_element['tag']} Text: \"{target_element['text'][:40]}\"")
                
                # 3. Click using the REAL ID
                print(f"\n🖱️  Clicking element...")
                click_success = self.extractor.click_by_id(real_id, elements)
                
                if click_success:
                    self.stats['system2_success'] += 1
                    elapsed = time.time() - start_time
                    
                    print(f"✅ Task completed in {elapsed:.2f}s (S1: {system1_time:.2f}s + S2: {system2_time:.2f}s)")
                    
                    return {
                        'success': True,
                        'method': 'vision',
                        'element_id': real_id,
                        'element': target_element,
                        'time': elapsed
                    }
                else:
                    print(f"❌ Click failed for ID {real_id}")
            
            # If we get here, Vision failed or returned an invalid index
            print(f"\n❌ SYSTEM 2 FAILED (took {system2_time:.2f}s)")
            
        except Exception as e:
            print(f"\n❌ SYSTEM 2 ERROR: {e}")
        
        # STEP 4: Both systems failed
        print("\n" + "=" * 70)
        print("❌ TASK FAILED - Both systems unable to complete")
        print("=" * 70)
        
        self.stats['failures'] += 1
        elapsed = time.time() - start_time
        
        return {
            'success': False,
            'method': 'failed',
            'element_id': -1,
            'element': None,
            'reason': 'Both systems failed',
            'time': elapsed
        }
    
    def execute_sequence(self, tasks: List[str], stop_on_failure: bool = False) -> List[Dict]:
        """
        Execute multiple tasks in sequence
        
        Args:
            tasks: List of task descriptions
            stop_on_failure: Stop execution if a task fails
            
        Returns:
            List of result dictionaries
        """
        print("\n" + "=" * 70)
        print(f"📋 EXECUTING TASK SEQUENCE ({len(tasks)} tasks)")
        print("=" * 70)
        
        results = []
        
        for idx, task in enumerate(tasks, 1):
            print(f"\n\n{'#' * 70}")
            print(f"TASK {idx}/{len(tasks)}")
            print(f"{'#' * 70}")
            
            result = self.execute_task(task)
            results.append(result)
            
            if not result['success'] and stop_on_failure:
                print(f"\n⚠️  Stopping sequence due to failure")
                break
            
            # Small delay between tasks
            if idx < len(tasks):
                time.sleep(1)
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get automation performance statistics
        
        Returns:
            Statistics dictionary
        """
        total = self.stats['total_attempts']
        
        if total == 0:
            return {
                'total_attempts': 0,
                'system1_success': 0,
                'system1_rate': 0.0,
                'system2_success': 0,
                'system2_rate': 0.0,
                'failures': 0,
                'failure_rate': 0.0
            }
        
        return {
            'total_attempts': total,
            'system1_success': self.stats['system1_success'],
            'system1_rate': (self.stats['system1_success'] / total) * 100,
            'system2_success': self.stats['system2_success'],
            'system2_rate': (self.stats['system2_success'] / total) * 100,
            'failures': self.stats['failures'],
            'failure_rate': (self.stats['failures'] / total) * 100
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("📊 AUTOMATION STATISTICS")
        print("=" * 70)
        
        if stats['total_attempts'] == 0:
            print("\nNo tasks executed yet")
            print("=" * 70)
            return
        
        print(f"\nTotal tasks attempted: {stats['total_attempts']}")
        print("\nPerformance breakdown:")
        print(f"  System 1 (DOM):     {stats['system1_success']:2d} / {stats['total_attempts']:2d} ({stats['system1_rate']:5.1f}%)")
        print(f"  System 2 (Vision):  {stats['system2_success']:2d} / {stats['total_attempts']:2d} ({stats['system2_rate']:5.1f}%)")
        print(f"  Failed:             {stats['failures']:2d} / {stats['total_attempts']:2d} ({stats['failure_rate']:5.1f}%)")
        
        # Success rate
        success_rate = stats['system1_rate'] + stats['system2_rate']
        print(f"\n✅ Overall success rate: {success_rate:.1f}%")
        
        # System efficiency
        if stats['system1_success'] + stats['system2_success'] > 0:
            s1_ratio = stats['system1_success'] / (stats['system1_success'] + stats['system2_success']) * 100
            print(f"⚡ System 1 efficiency: {s1_ratio:.1f}% (of successful tasks)")
        
        print("=" * 70)
    
    def get_current_state(self) -> Dict:
        """
        Get current page state for verification
        
        Returns:
            Current state dictionary
        """
        page_info = self.extractor.get_page_info()
        elements = self.extractor.extract_elements()
        
        return {
            'url': page_info['url'],
            'title': page_info['title'],
            'elements': elements,
            'element_count': len(elements)
        }
    
    def close(self):
        """Close browser and cleanup"""
        print("\n" + "=" * 70)
        print("🔄 CLOSING HYBRID AUTOMATOR")
        print("=" * 70)
        
        # Print final statistics
        self.print_statistics()
        
        # Cleanup vision agent if loaded
        if self.vision_agent:
            print("\n🔄 Unloading vision model...")
            self.vision_agent.unload_model()
        
        # Close browser
        self.extractor.close()
        
        print("\n✅ Hybrid Automator closed")


# Demo function
def demo_hybrid_system():
    """
    Demonstration of hybrid system on Wikipedia
    """
    print("=" * 70)
    print("DEMO: Hybrid Automation System")
    print("=" * 70)
    
    # Ask for vision model path
    model_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    
    automator = HybridAutomator(vision_model_path=model_path if model_path else None)
    
    try:
        # Start session
        automator.start("https://www.wikipedia.org", headless=False)
        
        # Define tasks
        tasks = [
            "click the search box",
            "click English Wikipedia",
        ]
        
        print(f"\n📋 Demo will execute {len(tasks)} tasks")
        input("Press Enter to start...")
        
        # Execute
        results = automator.execute_sequence(tasks)
        
        # Summary
        print("\n" + "=" * 70)
        print("DEMO RESULTS")
        print("=" * 70)
        for idx, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            method = result.get('method', 'N/A').upper()
            print(f"Task {idx}: {status} via {method}")
        
        automator.print_statistics()
        
    finally:
        automator.close()


# Interactive test
if __name__ == "__main__":
    import sys
    
    # Check if demo mode
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_hybrid_system()
        exit(0)
    
    print("=" * 70)
    print("HYBRID WEB AUTOMATOR - Interactive Mode")
    print("=" * 70)
    
    # Initialize
    model_path = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    automator = HybridAutomator(vision_model_path=model_path if model_path else None)
    
    # Get URL
    url = input("\nEnter URL to automate: ").strip()
    
    if not url:
        print("❌ No URL provided")
        exit(1)
    
    if not automator.start(url, headless=False):
        print("❌ Failed to start session")
        exit(1)
    
    print("\n" + "=" * 70)
    print("READY FOR AUTOMATION")
    print("=" * 70)
    print("\n💡 System Strategy:")
    print("   • System 1: Fast DOM text analysis (tries first)")
    print("   • System 2: Vision fallback (only if S1 fails)")
    print("\n💡 Commands:")
    print("   • Enter a task to execute it")
    print("   • Type 'stats' to see current statistics")
    print("   • Type 'state' to see current page state")
    print("   • Type 'quit' to exit")
    print("=" * 70)
    
    try:
        # Interactive loop
        while True:
            print("\n" + "-" * 70)
            command = input("\nCommand or task: ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if command.lower() == 'stats':
                automator.print_statistics()
                continue
            
            if command.lower() == 'state':
                state = automator.get_current_state()
                print(f"\n📄 Current Page State:")
                print(f"   URL: {state['url']}")
                print(f"   Title: {state['title']}")
                print(f"   Interactive elements: {state['element_count']}")
                continue
            
            # Execute task
            result = automator.execute_task(command)
            
            # Show result summary
            print("\n" + "=" * 70)
            if result['success']:
                print(f"🎉 TASK COMPLETED via {result['method'].upper()}")
                if result.get('time'):
                    print(f"   Time: {result['time']:.2f}s")
            else:
                print("❌ TASK FAILED - Both systems unable to complete")
                if result.get('reason'):
                    print(f"   Reason: {result['reason']}")
            print("=" * 70)
            
            # Quick stats
            stats = automator.get_statistics()
            print(f"\n📊 Session: S1={stats['system1_success']}, S2={stats['system2_success']}, Failed={stats['failures']}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        automator.close()
        print("\n✅ Session ended")