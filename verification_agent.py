"""
Verification Agent - Action Success Verification
Primary: Gemini 1.5 Flash
Fallback: Llama 3.2:3B (Local Ollama)
Checks if automation actions succeeded by comparing page states
"""

import ollama
import json
import re
from typing import Dict, Optional, Any
import os
import time

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai library not installed. Run: pip install google-generativeai")


class VerificationAgent:
    """
    Hybrid verification agent:
    - Primary: Gemini 1.5 Flash (Fast & Accurate)
    - Fallback: Llama 3.2:3B (Local)
    Verifies if automation actions succeeded by analyzing state changes
    Uses heuristics for simple cases, LLM for ambiguous verification
    """
    
    def __init__(
        self, 
        local_model: str = "llama3.2:3b",
        gemini_api_key: str = None,
        prefer_local: bool = False
    ):
        """
        Initialize hybrid verification agent
        """
        self.local_model = local_model
        self.prefer_local = prefer_local
        
        # Gemini setup
        self.gemini_available = False
        self.gemini_model = None
        self.gemini_model_name = "gemini-2.5-flash-lite"
        
        # Statistics
        self.stats = {
            'gemini_success': 0,
            'gemini_failures': 0,
            'local_success': 0,
            'local_failures': 0,
            'heuristic_verifications': 0
        }
        
        print(f"✓ Initializing Hybrid Verification Agent")
        print(f"   Primary: Gemini ({self.gemini_model_name})")
        print(f"   Fallback: {local_model} (Local)")
        
        # Setup Gemini
        if not prefer_local:
            self._setup_gemini(gemini_api_key)
        
        # Setup Ollama
        self._setup_ollama()
        
        print(f"✅ Hybrid verification agent ready!")
        if self.gemini_available:
            print(f"   ⚡ Gemini: Available")
        else:
            print(f"   ⚠️  Gemini: Not available (will use local only)")
        print(f"   🖥️  Local: Ready")
    
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
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model with optimized settings for verification
            self.gemini_model = genai.GenerativeModel(
                self.gemini_model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent verification
                    max_output_tokens=300,
                    response_mime_type="application/json"
                )
            )
            
            self.gemini_available = True
            print(f"   ✅ Gemini configured")
            
        except Exception as e:
            print(f"   ⚠️  Gemini setup failed: {e}")
            self.gemini_available = False
    
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

    def verify_action(
        self, 
        action: Dict, 
        before_state: Dict, 
        after_state: Dict
    ) -> Dict:
        """
        Verify if an action succeeded by comparing states
        """
        try:
            action_type = action.get('action', '').lower()
            
            print(f"\n🔍 Verifying: [{action_type}] {action.get('description', '')}")
            
            # Detect changes
            changes = self._detect_changes(before_state, after_state)
            
            # Try heuristic verification first (fast, no API calls)
            heuristic_result = self._heuristic_verification(action, changes, before_state, after_state)
            
            if heuristic_result is not None:
                # Heuristic gave us a clear answer
                self.stats['heuristic_verifications'] += 1
                print(f"   ⚡ Fast verification: {'✅ SUCCESS' if heuristic_result['success'] else '❌ FAILED'}")
                return heuristic_result
            
            # Need LLM for ambiguous case - use hybrid approach
            print(f"   🤖 Using LLM for verification...")
            
            # Try primary model first
            if not self.prefer_local and self.gemini_available:
                llm_result = self._verify_with_gemini(action, changes)
                if llm_result is not None:
                    return llm_result
                # If Gemini failed, fallback to local
                print("   🔄 Falling back to local model...")
            
            # Use local model
            llm_result = self._verify_with_local(action, changes)
            return llm_result if llm_result is not None else self._get_fallback_result(False)
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'reason': f'Verification error: {str(e)}',
                'retry_recommended': True
            }

    def _detect_changes(self, before: Dict, after: Dict) -> Dict:
        """Detect what changed between states"""
        changes = {
            'url_changed': False,
            'title_changed': False,
            'new_url': after.get('url', ''),
            'old_url': before.get('url', ''),
            'new_title': after.get('title', ''),
            'old_title': before.get('title', ''),
            'element_count_change': 0
        }
        
        # URL change
        if before.get('url', '') != after.get('url', ''):
            changes['url_changed'] = True
        
        # Title change
        if before.get('title', '') != after.get('title', ''):
            changes['title_changed'] = True
        
        # Element count change
        before_count = before.get('element_count', 0)
        after_count = after.get('element_count', 0)
        changes['element_count_change'] = after_count - before_count
        
        return changes

    def _heuristic_verification(self, action: Dict, changes: Dict, before: Dict, after: Dict) -> Optional[Dict]:
        """
        Fast heuristic verification for common cases
        """
        action_type = action.get('action', '').lower()
        target = action.get('target', '').lower()
        
        # NAVIGATE: Check URL contains target
        if action_type == 'navigate':
            return self.verify_navigation(target, changes['new_url'])
        
        # WAIT: Always succeeds
        if action_type == 'wait':
            return {
                'success': True,
                'confidence': 1.0,
                'reason': 'Wait action completed',
                'retry_recommended': False
            }
        
        # TYPE: Typing text doesn't cause page changes - assume success
        if action_type == 'type':
            return {
                'success': True,
                'confidence': 0.9,
                'reason': 'Text typed into focused element (no page changes expected)',
                'retry_recommended': False
            }
        
        # PRESS_KEY: Depends on the key
        if action_type == 'press_key':
            key = target.lower()
            # Enter/Submit keys should cause page changes
            if key in ['enter', 'return']:
                if changes['url_changed'] or changes['title_changed'] or changes['element_count_change'] > 5:
                    return {
                        'success': True,
                        'confidence': 0.9,
                        'reason': f'Page changed after pressing {target}',
                        'retry_recommended': False
                    }
                else:
                    # Enter pressed but nothing changed - might be loading or failed
                    # Return None to let LLM decide based on context
                    return None 
            else:
                # Other keys (Tab, Escape) - assume success
                return {
                    'success': True,
                    'confidence': 0.8,
                    'reason': f'Key {target} pressed',
                    'retry_recommended': False
                }
        
        # FIND_AND_CLICK: Check what we're clicking
        if action_type == 'find_and_click':
            # Check if we're clicking an INPUT/SEARCH element (won't cause page changes)
            is_input_element = any(word in target for word in [
                'search', 'input', 'field', 'box', 'text', 'email', 'password', 'username'
            ])
            
            if is_input_element:
                return {
                    'success': True,
                    'confidence': 0.9,
                    'reason': 'Input element clicked (no page changes expected)',
                    'retry_recommended': False
                }
            
            # For other clicks, expect changes
            return self.verify_element_interaction(action, changes)
        
        # SCROLL: Assume success
        if action_type == 'scroll':
            return {
                'success': True,
                'confidence': 0.8,
                'reason': 'Scroll action completed',
                'retry_recommended': False
            }
        
        return None

    def verify_navigation(self, target: str, actual_url: str) -> Dict:
        """Verify navigation action"""
        target_clean = target.replace('https://', '').replace('http://', '').strip('/')
        actual_clean = actual_url.replace('https://', '').replace('http://', '')
        
        success = target_clean.lower() in actual_clean.lower()
        
        if success:
            return {
                'success': True,
                'confidence': 0.95,
                'reason': f"URL matches target",
                'retry_recommended': False
            }
        # If URL didn't match, it might be a redirect or error. 
        # Returning None allows LLM to double check, or we can fail hard.
        # Heuristic implies we expect a match.
        return {
             'success': False,
             'confidence': 0.9,
             'reason': f"URL {actual_url} does not contain {target}",
             'retry_recommended': True
        }

    def verify_element_interaction(self, action: Dict, changes: Dict) -> Optional[Dict]:
        """Verify element interaction (click, etc.)"""
        # Strong indicators of success
        if changes['url_changed']:
            return {
                'success': True,
                'confidence': 0.9,
                'reason': f"URL changed to {changes['new_url']}",
                'retry_recommended': False
            }
        
        if changes['title_changed']:
            return {
                'success': True,
                'confidence': 0.85,
                'reason': f"Page title changed to '{changes['new_title']}'",
                'retry_recommended': False
            }
        
        # New elements appeared
        if changes['element_count_change'] > 5:
            return {
                'success': True,
                'confidence': 0.8,
                'reason': f"{changes['element_count_change']} new elements appeared",
                'retry_recommended': False
            }
        
        # Nothing changed - likely failed, but returning None lets LLM confirm
        if changes['element_count_change'] == 0 and not changes['url_changed']:
             return None # Let LLM decide if "No Change" is a failure for this specific target
        
        return None

    def _verify_with_gemini(self, action: Dict, changes: Dict) -> Optional[Dict]:
        """Try Gemini API for verification"""
        try:
            prompt = self._create_verification_prompt(action, changes)
            
            # Generate verification
            response = self.gemini_model.generate_content(prompt)
            llm_response = response.text.strip()
            
            result = self._parse_verification_response(llm_response)
            
            if result:
                self.stats['gemini_success'] += 1
                print(f"   ✅ Gemini verification: {'SUCCESS' if result['success'] else 'FAILED'}")
                return result
            else:
                self.stats['gemini_failures'] += 1
                return None
            
        except Exception as e:
            self.stats['gemini_failures'] += 1
            print(f"   ⚠️  Gemini verification failed: {e}")
            return None

    def _verify_with_local(self, action: Dict, changes: Dict) -> Optional[Dict]:
        """Try local Ollama model for verification"""
        try:
            prompt = self._create_verification_prompt(action, changes)
            print(f"   🤖 Verifying with {self.local_model}...")
            
            response = ollama.generate(
                model=self.local_model,
                prompt=prompt + "\nRespond in JSON only.",
                stream=False,
                options={'temperature': 0.1, 'num_predict': 200}
            )
            
            llm_response = response['response'].strip()
            result = self._parse_verification_response(llm_response)
            
            if result:
                self.stats['local_success'] += 1
                print(f"   ✅ Local verification: {'SUCCESS' if result['success'] else 'FAILED'}")
                return result
            else:
                self.stats['local_failures'] += 1
                return None
                
        except Exception as e:
            print(f"   ❌ Local model error: {e}")
            self.stats['local_failures'] += 1
            return None

    def _create_verification_prompt(self, action: Dict, changes: Dict) -> str:
        """Create prompt for LLM verification"""
        return f"""You are a Web Automation Verifier. 
Verify if this action succeeded based on the state changes.

ACTION: {action.get('action')}
TARGET: {action.get('target')}
DESC: {action.get('description')}

CHANGES OBSERVED:
- URL Changed: {changes['url_changed']} (Now: {changes['new_url']})
- Title Changed: {changes['title_changed']} (Now: {changes['new_title']})
- Elements Changed: {changes['element_count_change']}

RULES:
1. Navigations must show URL changes.
2. Form submissions/Link clicks usually change URL or Title.
3. If no changes occurred but action was 'click', it might have failed OR be a dynamic overlay. Use best judgment.

Respond with valid JSON only:
{{
  "success": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "short explanation",
  "retry_recommended": true/false
}}"""

    def _parse_verification_response(self, response: str) -> Optional[Dict]:
        """Parse LLM verification response"""
        try:
            # Clean up markdown
            clean_response = response.replace('```json', '').replace('```', '').strip()
            # Extract JSON part if mixed with text
            match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if match:
                clean_response = match.group(0)
            
            result = json.loads(clean_response)
            
            return {
                'success': bool(result.get('success', False)),
                'confidence': float(result.get('confidence', 0.5)),
                'reason': str(result.get('reason', 'Parsed from LLM')),
                'retry_recommended': bool(result.get('retry_recommended', True))
            }
        except Exception as e:
            print(f"   ⚠️ Parse error: {e}")
            return None

    def _get_fallback_result(self, success: bool) -> Dict:
        """Fallback result"""
        return {
            'success': success,
            'confidence': 0.3,
            'reason': 'Verification models failed - assuming result',
            'retry_recommended': not success
        }

    def print_statistics(self):
        """Print usage statistics"""
        stats = self.stats
        print("\n" + "=" * 60)
        print("📊 HYBRID VERIFICATION AGENT STATISTICS")
        print("=" * 60)
        print(f"⚡ Heuristic (fast): {stats['heuristic_verifications']}")
        print(f"🌐 Gemini Success:   {stats['gemini_success']}")
        print(f"🌐 Gemini Failures:  {stats['gemini_failures']}")
        print(f"🖥️  Local Success:    {stats['local_success']}")
        print(f"🖥️  Local Failures:   {stats['local_failures']}")
        print("=" * 60)


# ==========================================
# TEST RUNNER (FULL RESTORATION)
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID VERIFICATION AGENT TEST")
    print("Primary: Gemini 1.5 Flash | Fallback: Llama 3.2:3B")
    print("=" * 70)
    
    # Setup - REPLACE THIS or use env var
    GEMINI_API_KEY = "" 
    
    # If using Env Var, allow it to pass through
    if GEMINI_API_KEY == "":
        GEMINI_API_KEY = None 
    
    agent = VerificationAgent(gemini_api_key=GEMINI_API_KEY)
    
    # Test Case 1: Successful navigation (heuristic)
    print("\n" + "=" * 70)
    print("TEST 1: Successful Navigation (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'navigate',
        'target': 'youtube.com',
        'description': 'Navigate to YouTube'
    }
    
    before = {
        'url': 'https://google.com',
        'title': 'Google',
        'element_count': 20
    }
    
    after = {
        'url': 'https://www.youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")

    # Test Case 2: Failed click (heuristic)
    print("\n" + "=" * 70)
    print("TEST 2: Failed Click (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'login button',
        'description': 'Click login button'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50 # No change expected for failed click
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")

    # Test Case 3: Successful click (heuristic)
    print("\n" + "=" * 70)
    print("TEST 3: Successful Click - URL change (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'search button',
        'description': 'Click search button'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com/results?search_query=cats',
        'title': 'cats - YouTube',
        'element_count': 60
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")

    # Test Case 4: Wait action (heuristic)
    print("\n" + "=" * 70)
    print("TEST 4: Wait Action (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'wait',
        'target': '2',
        'description': 'Wait for page load'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")

    # Test Case 5: Verify action (requires LLM - ambiguous)
    print("\n" + "=" * 70)
    print("TEST 5: Verify Action (LLM - Ambiguous)")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'submit',
        'description': 'Submit form data'
    }
    
    before = {
        'url': 'https://site.com/form',
        'title': 'Form',
        'element_count': 20
    }
    
    after = {
        'url': 'https://site.com/form', # URL didn't change
        'title': 'Form', 
        'element_count': 22 # Slight count change
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")

    # Test Case 6: Type action (heuristic)
    print("\n" + "=" * 70)
    print("TEST 6: Type Action (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'type',
        'target': 'cats',
        'description': 'Type search query'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 7: Press Enter (heuristic)
    print("\n" + "=" * 70)
    print("TEST 7: Press Enter (Heuristic)")
    print("=" * 70)
    
    action = {
        'action': 'press_key',
        'target': 'Enter',
        'description': 'Submit search'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com/results?search_query=test',
        'title': 'test - YouTube',
        'element_count': 55
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    # Test Case 8: Click search box (heuristic - input element)
    print("\n" + "=" * 70)
    print("TEST 8: Click Search Box (Heuristic - Input Element)")
    print("=" * 70)
    
    action = {
        'action': 'find_and_click',
        'target': 'search box',
        'description': 'Click search box'
    }
    
    before = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50
    }
    
    after = {
        'url': 'https://youtube.com',
        'title': 'YouTube',
        'element_count': 50  # No change expected for input click
    }
    
    result = agent.verify_action(action, before, after)
    print(f"\n📊 Result:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")
    print(f"   Retry: {result['retry_recommended']}")
    
    print("\n" + "=" * 70)
    print("✅ All verification tests complete!")
    print("=" * 70)
    
    # Show statistics
    agent.print_statistics()
    
    # Interactive testing
    print("\n💡 Interactive verification testing")
    print("   Type 'quit' to exit\n")
    
    while True:
        try:
            print("\n" + "-" * 70)
            print("Enter action details for verification:")
            
            action_type = input("Action type (navigate/find_and_click/type/wait/press_key/scroll/verify) or 'quit': ").strip().lower()
            
            if action_type in ['quit', 'exit', 'q']:
                break
            
            if not action_type:
                continue
            
            target = input("Target: ").strip()
            description = input("Description (optional): ").strip()
            
            action = {
                'action': action_type,
                'target': target,
                'description': description or f"{action_type} {target}"
            }
            
            print("\nBefore state:")
            before_url = input("  URL before (default: https://google.com): ").strip() or "https://google.com"
            before_title = input("  Title before (default: Google): ").strip() or "Google"
            before_count = input("  Element count before (default: 50): ").strip()
            before_count = int(before_count) if before_count.isdigit() else 50
            
            before = {
                'url': before_url,
                'title': before_title,
                'element_count': before_count
            }
            
            print("\nAfter state:")
            after_url = input("  URL after (default: same as before): ").strip() or before_url
            after_title = input("  Title after (default: same as before): ").strip() or before_title
            after_count = input("  Element count after (default: same as before): ").strip()
            after_count = int(after_count) if after_count.isdigit() else before_count
            
            after = {
                'url': after_url,
                'title': after_title,
                'element_count': after_count
            }
            
            # Verify
            result = agent.verify_action(action, before, after)
            
            print(f"\n📊 Verification Result:")
            print(f"   {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Reason: {result['reason']}")
            print(f"   Retry Recommended: {result['retry_recommended']}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    agent.print_statistics()
    
    print("\n✅ Verification Agent test complete!")