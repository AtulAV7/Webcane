"""
Planning Agent - Task Decomposition
Primary: Gemini 2.0 Flash
Fallback: Llama 3.2:3B (Local Ollama)
Breaks complex multi-step goals into atomic actions
"""

import ollama
import json
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


class PlanningAgent:
    """
    Hybrid planning agent:
    - Primary: Gemini 2.0 Flash (Fast & Smart)
    - Fallback: Llama 3.2:3B (Local)
    Decomposes high-level goals into step-by-step action plans
    """
    
    # Valid action types
    VALID_ACTIONS = [
        'navigate',       # Go to a URL
        'find_and_click', # Find and click an element
        'type',           # Type text
        'wait',           # Wait N seconds
        'scroll',         # Scroll page
        'verify',         # Verify something exists
        'press_key'       # Press a keyboard key (Enter, Tab, etc.)
    ]
    
    def __init__(
        self, 
        local_model: str = "llama3.2:3b",
        gemini_api_key: str = None,
        prefer_local: bool = False
    ):
        """
        Initialize hybrid planning agent
        
        Args:
            local_model: Ollama model name for fallback
            gemini_api_key: Gemini API key (or set GEMINI_API_KEY env var)
            prefer_local: If True, use local model first (for testing)
        """
        self.local_model = local_model
        self.prefer_local = prefer_local
        
        # Gemini setup
        self.gemini_available = False
        self.gemini_model = None
        self.gemini_model_name = "gemini-2.5-flash"
        
        # Statistics
        self.stats = {
            'gemini_success': 0,
            'gemini_failures': 0,
            'local_success': 0,
            'local_failures': 0
        }
        
        print(f"🧠 Initializing Hybrid Planning Agent")
        print(f"   Primary: Gemini ({self.gemini_model_name})")
        print(f"   Fallback: {local_model} (Local)")
        
        # Setup Gemini
        if not prefer_local:
            self._setup_gemini(gemini_api_key)
        
        # Setup Ollama
        self._setup_ollama()
        
        print(f"✅ Hybrid planning agent ready!")
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
                print("       Set GEMINI_API_KEY env var or pass to constructor")
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize model
            self.gemini_model = genai.GenerativeModel(
                self.gemini_model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
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
    
    def decompose_task(
        self, 
        goal: str, 
        current_url: str = "about:blank",
        page_description: str = None,
        failure_context: Dict = None
    ) -> List[Dict]:
        """
        Decompose high-level goal into atomic action steps using hybrid approach
        
        Args:
            goal: High-level task description
            current_url: Current page URL for context
            page_description: Visual description of current page (from Visual Observer)
            failure_context: Context from a previous failed attempt (for replanning)
            
        Returns:
            List of action dictionaries, or empty list on failure
        """
        if not goal or not goal.strip():
            print("❌ Empty goal provided")
            return []
        
        try:
            # Simplify and clean the goal
            simplified_goal = self.simplify_task(goal)
            
            print(f"\n🧠 Planning: {simplified_goal}")
            print(f"   Current URL: {current_url}")
            if page_description:
                print(f"   👁️ Page Context: {page_description[:100]}...")
            if failure_context:
                print(f"   🔄 Replanning after failure: {failure_context.get('reason', 'Unknown')}")
            
            # Create planning prompt with context
            prompt = self._create_planning_prompt(
                simplified_goal, 
                current_url, 
                page_description,
                failure_context
            )
            
            # Try primary model first
            if not self.prefer_local and self.gemini_available:
                plan = self._try_gemini(prompt)
                if plan is not None:
                    return self._finalize_plan(plan, simplified_goal)
                # If Gemini failed, fallback to local
                print("   🔄 Falling back to local model...")
            
            # Use local model
            plan = self._try_local(prompt)
            return self._finalize_plan(plan, simplified_goal) if plan is not None else []
            
        except Exception as e:
            print(f"❌ Task decomposition failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _try_gemini(self, prompt: str) -> Optional[List[Dict]]:
        """
        Try Gemini API for planning
        
        Returns:
            Plan list or None on failure (triggers fallback)
        """
        try:
            print(f"🤖 Generating plan with Gemini ({self.gemini_model_name})...")
            
            # Generate plan
            response = self.gemini_model.generate_content(prompt)
            
            llm_response = response.text.strip()
            print(f"🤖 Gemini Response length: {len(llm_response)} chars")
            
            # Parse the plan
            plan = self._parse_plan(llm_response)
            
            if plan:
                self.stats['gemini_success'] += 1
                return plan
            else:
                self.stats['gemini_failures'] += 1
                return None
            
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
    
    def _try_local(self, prompt: str) -> Optional[List[Dict]]:
        """
        Try local Ollama model with a SIMPLIFIED prompt
        
        Returns:
            Plan list or None on failure
        """
        try:
            print(f"🤖 Generating plan with {self.local_model}...")
            
            # Extract key info from the original prompt for a simpler version
            # The local model struggles with complex prompts, so we simplify
            simplified_prompt = self._create_simplified_prompt_for_local(prompt)
            
            # Query Ollama with simplified prompt
            response = ollama.generate(
                model=self.local_model,
                prompt=simplified_prompt,
                stream=False,
                options={
                    'temperature': 0.1,  # Lower temperature for more predictable output
                    'num_predict': 400,
                }
            )
            
            llm_response = response['response'].strip()
            print(f"🤖 Local Response length: {len(llm_response)} chars")
            
            # Parse the plan
            plan = self._parse_plan(llm_response)
            
            if plan:
                self.stats['local_success'] += 1
            else:
                self.stats['local_failures'] += 1
            
            return plan
            
        except Exception as e:
            print(f"❌ Local model error: {e}")
            self.stats['local_failures'] += 1
            return None
    
    def _create_simplified_prompt_for_local(self, original_prompt: str) -> str:
        """
        Create a much simpler prompt for the local model (Llama 3.2 3B).
        Local models struggle with long, complex prompts.
        """
        # Extract URL and goal from original prompt
        import re
        
        url_match = re.search(r'URL:\s*(\S+)', original_prompt)
        current_url = url_match.group(1) if url_match else "unknown"
        
        goal_match = re.search(r'USER GOAL:\s*(.+?)(?:\n|$)', original_prompt)
        goal = goal_match.group(1).strip() if goal_match else "unknown goal"
        
        # Create a MUCH simpler prompt with STRICT rules about preserving target text
        simplified = f"""Convert this task to JSON steps. Output ONLY valid JSON array.

URL: {current_url}
TASK: {goal}

STRICT RULES:
1. You are ALREADY on this URL. Do NOT add "navigate" unless going to a DIFFERENT website.
2. ONLY do exactly what the task says. NO extra steps.
3. NEVER simplify or shorten the user's words. Copy them EXACTLY.
4. Valid actions: navigate, find_and_click, type, scroll, press_key, wait
5. Valid keys for press_key: Enter, Tab, Escape, ArrowUp, ArrowDown. NOT "Play button".

TASK PATTERNS:
- "click video titled X" = [{{"step":1,"action":"find_and_click","target":"video titled X","description":"Click video"}}]
- "search for X" = [{{"step":1,"action":"find_and_click","target":"search box","description":"Click search"}},{{"step":2,"action":"type","target":"X","description":"Type"}},{{"step":3,"action":"press_key","target":"Enter","description":"Submit"}}]

EXAMPLES:

Task: "click video titled kitten falls off a bike"
Output: [{{"step":1,"action":"find_and_click","target":"video titled kitten falls off a bike","description":"Click video"}}]

Task: "search for samsung review"
Output: [{{"step":1,"action":"find_and_click","target":"search box","description":"Click search"}},{{"step":2,"action":"type","target":"samsung review","description":"Type query"}},{{"step":3,"action":"press_key","target":"Enter","description":"Submit"}}]

Now output JSON for: {goal}
JSON:"""
        
        return simplified
        
        return simplified
    
    def _finalize_plan(self, plan: List[Dict], goal: str) -> List[Dict]:
        """
        Validate and finalize the plan
        
        Args:
            plan: Raw parsed plan
            goal: Original goal
            
        Returns:
            Validated plan or empty list
        """
        if not plan:
            print("❌ Failed to generate valid plan")
            return []
        
        # Validate plan
        if not self.validate_plan(plan):
            print("⚠️  Plan validation failed, but returning anyway")
        
        print(f"✅ Generated plan with {len(plan)} steps")
        return plan
    
    def _create_planning_prompt(
        self, 
        goal: str, 
        current_url: str,
        page_description: str = None,
        failure_context: Dict = None
    ) -> str:
        """
        Create prompt for LLM to generate action plan
        
        Args:
            goal: Simplified goal
            current_url: Current page URL
            page_description: Visual description of current page
            failure_context: Context from previous failure (for replanning)
            
        Returns:
            Formatted prompt string
        """
        # Build context section
        context_section = f"URL: {current_url}"
        
        if page_description:
            context_section += f"\nPAGE DESCRIPTION (from visual analysis): {page_description}"
        
        # Build failure context if replanning
        failure_section = ""
        if failure_context:
            failure_section = f"""

⚠️ REPLANNING AFTER FAILURE:
Previous failed action: {failure_context.get('failed_action', 'Unknown')}
Failure reason: {failure_context.get('reason', 'Unknown')}
Current page state: {failure_context.get('current_state', 'Unknown')}

IMPORTANT: Generate a RECOVERY plan that addresses the failure. Do NOT repeat the same action that failed.
"""
        
        prompt = f"""You are a web automation planning agent. Break down high-level goals into step-by-step actions.

CURRENT STATE:
{context_section}
{failure_section}
USER GOAL: {goal}

Break this goal into atomic actions. Each action must be ONE of these types:
- navigate: Go to a URL (target = full URL or domain)
- find_and_click: Find and click an element (target = element description like "search button", "login link", "video titled X")
- type: Type text into the currently focused element (target = text to type)
- wait: Wait for page to load (target = number of seconds, usually 1-3)
- scroll: Scroll the page (target = "down", "up", or pixel amount like "500")
- press_key: Press a keyboard key (target = "Enter", "Tab", "Escape", etc.)
- verify: Check if something exists (target = what to verify)

🔴 CONTEXT AWARENESS (CRITICAL):
1. CHECK THE CURRENT URL FIRST. If already on the target website, do NOT add a "navigate" step.
   - Example: If URL is "youtube.com/results?search_query=cats" and goal is "click a video", do NOT navigate to youtube.com again.
2. ONLY do what the user explicitly asks. Do NOT add extra steps beyond the goal.
   - If user says "search for X", STOP after the search is submitted. Do NOT click on results.
   - If user says "click video titled X", just find and click that video. Do NOT search again.

IMPORTANT RULES:
3. Be SPECIFIC in element descriptions (e.g., "search button" not "button")
4. Include "wait" steps after actions that load new content
5. Keep each step atomic (ONE action per step)
6. Number steps sequentially starting from 1
7. For searches: click search box → type query → press Enter. STOP THERE unless user asks to click something.
8. If PAGE DESCRIPTION is provided, use it to understand what's actually visible on screen

🔴 SCROLL-TO-FIND RULE:
9. If the goal is to click a SPECIFIC item (like a video title) that might not be immediately visible:
   - Add a "scroll" step before the "find_and_click" step
10. For "click video titled X" tasks, include one scroll step

Return ONLY a valid JSON array. Examples:

For "search for cats" when already on youtube.com:
[
  {{"step": 1, "action": "find_and_click", "target": "search box", "description": "Click search box"}},
  {{"step": 2, "action": "type", "target": "cats", "description": "Type search query"}},
  {{"step": 3, "action": "press_key", "target": "Enter", "description": "Submit search"}}
]

For "click video titled funny cats" when on youtube.com/results:
[
  {{"step": 1, "action": "scroll", "target": "down", "description": "Scroll to see more videos"}},
  {{"step": 2, "action": "find_and_click", "target": "video titled funny cats", "description": "Click the target video"}}
]

Now plan for: {goal}

Return ONLY the JSON array, nothing else:"""
        
        return prompt
    
    def _parse_plan(self, response: str) -> List[Dict]:
        """
        Parse LLM response to extract action plan
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of action dicts, or empty list on failure
        """
        try:
            # Try to find JSON array in response
            
            # First, try direct JSON parse
            try:
                plan = json.loads(response)
                if isinstance(plan, list):
                    return self._validate_and_clean_plan(plan)
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from markdown code blocks
            # Look for ```json ... ``` or ``` ... ```
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
                try:
                    plan = json.loads(json_str)
                    if isinstance(plan, list):
                        return self._validate_and_clean_plan(plan)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON from text
            # Look for [...] pattern
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    plan = json.loads(json_str)
                    if isinstance(plan, list):
                        return self._validate_and_clean_plan(plan)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract multiple {...} objects
            object_pattern = r'\{[^}]+\}'
            objects = re.findall(object_pattern, response, re.DOTALL)
            if objects:
                plan = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        plan.append(obj)
                    except json.JSONDecodeError:
                        continue
                
                if plan:
                    return self._validate_and_clean_plan(plan)
            
            print(f"⚠️  Could not parse JSON from response")
            print(f"Response preview: {response[:200]}...")
            return []
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            return []
    
    def _validate_and_clean_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Validate and clean parsed plan
        
        Args:
            plan: Raw parsed plan
            
        Returns:
            Cleaned and validated plan
        """
        cleaned = []
        
        for idx, step in enumerate(plan, 1):
            # Ensure required fields
            if not isinstance(step, dict):
                continue
            
            action = step.get('action', '').lower()
            if action not in self.VALID_ACTIONS:
                print(f"⚠️  Invalid action '{action}' in step {idx}, skipping")
                continue
            
            # Build cleaned step
            cleaned_step = {
                'step': step.get('step', idx),
                'action': action,
                'target': str(step.get('target', '')),
                'description': step.get('description', f"{action} action")
            }
            
            cleaned.append(cleaned_step)
        
        # Renumber steps
        for idx, step in enumerate(cleaned, 1):
            step['step'] = idx
        
        return cleaned
    
    def validate_plan(self, plan: List[Dict]) -> bool:
        """
        Validate that plan is well-formed
        
        Args:
            plan: Action plan to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not plan:
            return False
        
        for idx, step in enumerate(plan, 1):
            # Check required fields
            if 'step' not in step or 'action' not in step or 'target' not in step:
                print(f"⚠️  Step {idx} missing required fields")
                return False
            
            # Check action type
            if step['action'] not in self.VALID_ACTIONS:
                print(f"⚠️  Step {idx} has invalid action: {step['action']}")
                return False
            
            # Check step numbering
            if step['step'] != idx:
                print(f"⚠️  Step numbering issue at {idx} (has {step['step']})")
                # Don't fail on this, just warn
        
        return True
    
    def simplify_task(self, complex_task: str) -> str:
        """
        Simplify and clean task description
        
        Args:
            complex_task: Raw user input
            
        Returns:
            Cleaned and simplified task
        """
        # Remove extra whitespace
        task = ' '.join(complex_task.split())
        
        # Convert to lowercase for consistency
        task = task.lower()
        
        # Remove common filler words but keep meaning
        task = task.replace(' please ', ' ')
        task = task.replace(' could you ', '')
        task = task.replace(' can you ', '')
        task = task.replace(' i want to ', '')
        task = task.replace(' i need to ', '')
        
        return task.strip()
    
    def replan_after_failure(
        self,
        original_goal: str,
        failed_step: Dict,
        failure_reason: str,
        current_url: str,
        page_description: str = None
    ) -> List[Dict]:
        """
        Generate a recovery plan after a step failure.
        
        Args:
            original_goal: The original user goal
            failed_step: The step that failed
            failure_reason: Why it failed
            current_url: Current page URL
            page_description: Visual description of current page state
            
        Returns:
            New recovery plan, or empty list on failure
        """
        print("\n" + "🔄 " * 23)
        print("REFLECTION: Generating recovery plan...")
        print("🔄 " * 23)
        
        failure_context = {
            'failed_action': f"{failed_step.get('action', 'unknown')}: {failed_step.get('target', 'unknown')}",
            'reason': failure_reason,
            'current_state': page_description or 'Unknown'
        }
        
        # Use decompose_task with failure context
        recovery_plan = self.decompose_task(
            goal=original_goal,
            current_url=current_url,
            page_description=page_description,
            failure_context=failure_context
        )
        
        if recovery_plan:
            print(f"✅ Generated recovery plan with {len(recovery_plan)} steps")
        else:
            print("❌ Failed to generate recovery plan")
        
        return recovery_plan
    
    def print_plan(self, plan: List[Dict]):
        """
        Print plan in readable format
        
        Args:
            plan: Action plan to display
        """
        if not plan:
            print("❌ No plan to display")
            return
        
        print("\n" + "=" * 70)
        print(f"📋 ACTION PLAN ({len(plan)} steps)")
        print("=" * 70)
        
        for step in plan:
            action_icon = {
                'navigate': '🌐',
                'find_and_click': '🖱️',
                'type': '⌨️',
                'wait': '⏳',
                'scroll': '📜',
                'verify': '✔',
                'press_key': '⌨️'
            }.get(step['action'], '▶️')
            
            print(f"\n{step['step']:2d}. {action_icon} [{step['action'].upper()}]")
            print(f"    Target: {step['target']}")
            print(f"    {step['description']}")
        
        print("\n" + "=" * 70)
    
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
        print("📊 HYBRID PLANNING AGENT STATISTICS")
        print("=" * 60)
        print(f"Total requests: {stats['total_requests']}")
        print(f"\n⚡ Gemini API:")
        print(f"   Success: {stats['gemini']['success']}")
        print(f"   Failures: {stats['gemini']['failures']}")
        print(f"   Success rate: {stats['gemini']['rate']:.1f}%")
        print(f"\n🖥️  Local Ollama:")
        print(f"   Success: {stats['local']['success']}")
        print(f"   Failures: {stats['local']['failures']}")
        print(f"   Success rate: {stats['local']['rate']:.1f}%")
        print("=" * 60)


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID PLANNING AGENT TEST")
    print("Primary: Gemini 2.0 Flash | Fallback: Llama 3.2:3B")
    print("=" * 70)
    
    # Setup
    GEMINI_API_KEY = "AIzaSyDreyXEO5EYGgy9oJYvPXHI0GSD3KwV8ik"  # Replace or use env var
    
    if GEMINI_API_KEY == "":
        GEMINI_API_KEY = None  # Will check env var
    
    agent = PlanningAgent(gemini_api_key=GEMINI_API_KEY)
    
    # Test cases
    test_cases = [
        {
            'goal': "search for Mr Beast on youtube",
            'url': "https://google.com"
        },
        {
            'goal': "go to github and find the login button",
            'url': "about:blank"
        },
        {
            'goal': "search wikipedia for Taj Mahal",
            'url': "https://google.com"
        },
        {
            'goal': "go to youtube, search for cats, and play the first video",
            'url': "about:blank"
        }
    ]
    
    print("\n🧪 Running test cases...\n")
    
    for idx, test in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {idx}")
        print("=" * 70)
        print(f"📋 GOAL: {test['goal']}")
        print(f"🌐 Current URL: {test['url']}")
        print("-" * 70)
        
        plan = agent.decompose_task(test['goal'], test['url'])
        
        if plan:
            agent.print_plan(plan)
            
            # Show validation result
            is_valid = agent.validate_plan(plan)
            print(f"\n{'✅' if is_valid else '⚠️'} Plan validation: {'PASSED' if is_valid else 'ISSUES FOUND'}")
        else:
            print("\n❌ Failed to generate plan")
        
        if idx < len(test_cases):
            input("\nPress Enter for next test case...")
    
    print("\n" + "=" * 70)
    print("✅ All tests complete!")
    print("=" * 70)
    
    # Show statistics
    agent.print_statistics()
    
    # Interactive mode
    print("\n💡 Interactive mode - Enter your own goals")
    print("   Type 'quit' to exit\n")
    
    while True:
        try:
            goal = input("\nEnter goal (or 'quit'): ").strip()
            
            if goal.lower() in ['quit', 'exit', 'q']:
                break
            
            if not goal:
                continue
            
            url = input("Current URL (or press Enter for 'about:blank'): ").strip()
            if not url:
                url = "about:blank"
            
            plan = agent.decompose_task(goal, url)
            
            if plan:
                agent.print_plan(plan)
            else:
                print("\n❌ Failed to generate plan")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            break
    
    # Final statistics
    agent.print_statistics()
    
    print("\n✅ Planning Agent test complete!")