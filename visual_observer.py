"""
Visual Observer - Page Description using Vision AI
Uses Gemini Vision to describe the current page before planning.
This helps the planner avoid hallucinating steps that aren't possible.
"""

import os
from typing import Optional

# Gemini imports
try:
    import google.generativeai as genai
    from PIL import Image
    import io
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai or PIL not installed")


class VisualObserver:
    """
    Analyzes screenshots to describe what's visible on the page.
    Used to provide context to the planning agent.
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the Visual Observer
        
        Args:
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.gemini_available = False
        self.gemini_model = None
        self.gemini_model_name = "gemini-2.5-flash"
        
        print("👁️  Initializing Visual Observer")
        
        if not GEMINI_AVAILABLE:
            print("   ⚠️  Required libraries not installed")
            return
        
        try:
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                print("   ⚠️  No Gemini API key provided")
                return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
            self.gemini_available = True
            print("   ✅ Visual Observer ready")
            
        except Exception as e:
            print(f"   ⚠️  Setup failed: {e}")
    
    def describe_page(self, screenshot_bytes: bytes) -> Optional[str]:
        """
        Analyze a screenshot and return a description of what's visible.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            
        Returns:
            2-3 sentence description of the page, or None on failure
        """
        if not self.gemini_available:
            return None
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            prompt = """Describe this webpage screenshot in 2-3 concise sentences.
Focus on:
1. What website/page is this (if identifiable)?
2. What interactive elements are visible (buttons, search boxes, forms)?
3. Are there any popups, overlays, or blockers visible?

Keep it brief and factual. Example:
"This is YouTube's homepage. There is a search bar at the top center. The page shows trending video thumbnails in a grid layout."

Your description:"""
            
            response = self.gemini_model.generate_content(
                [image, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=150,
                )
            )
            
            description = response.text.strip()
            print(f"👁️  Page Description: {description}")
            return description
            
        except Exception as e:
            print(f"   ⚠️  Vision analysis failed: {e}")
            return None
    
    def describe_failure_context(
        self, 
        screenshot_bytes: bytes, 
        failed_action: str,
        failure_reason: str
    ) -> Optional[str]:
        """
        Analyze the page after a failure to provide context for replanning.
        
        Args:
            screenshot_bytes: Current screenshot
            failed_action: What action failed
            failure_reason: Why it failed
            
        Returns:
            Analysis of current state and suggestions
        """
        if not self.gemini_available:
            return None
        
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            prompt = f"""A web automation action just failed. Analyze this screenshot.

FAILED ACTION: {failed_action}
FAILURE REASON: {failure_reason}

Answer these questions concisely:
1. What is currently visible on the page?
2. Why might the action have failed based on what you see?
3. What should be tried instead?

Keep response under 4 sentences."""
            
            response = self.gemini_model.generate_content(
                [image, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                )
            )
            
            analysis = response.text.strip()
            print(f"👁️  Failure Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            print(f"   ⚠️  Failure analysis failed: {e}")
            return None


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("VISUAL OBSERVER TEST")
    print("=" * 70)
    
    observer = VisualObserver()
    
    if not observer.gemini_available:
        print("❌ Cannot test - Gemini not available")
        exit(1)
    
    # Test with a sample screenshot if available
    test_image_path = "test_screenshot.png"
    
    if os.path.exists(test_image_path):
        with open(test_image_path, 'rb') as f:
            screenshot = f.read()
        
        description = observer.describe_page(screenshot)
        print(f"\n✅ Description: {description}")
    else:
        print(f"\n⚠️  No test image found at {test_image_path}")
        print("   Run this from a project with a screenshot to test")
