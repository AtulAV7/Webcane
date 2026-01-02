"""
DOM Extractor for Web Automation
Extracts interactive elements with bounding boxes and properties
"""

from playwright.sync_api import sync_playwright, Page, Browser
from typing import List, Dict, Optional
import json


class DOMExtractor:
    """Extract interactive DOM elements with positioning data for automation"""
    
    def __init__(self):
        """Initialize browser management - no browser started yet"""
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
    def start_browser(self, headless: bool = False) -> bool:
        """
        Launch Chromium browser with Playwright.
        REUSES existing browser if already running to prevent asyncio errors.
        
        Args:
            headless: Run browser in headless mode
            
        Returns:
            True on success, False on failure
        """
        # 🟢 FIX: Check if Playwright is already active to avoid asyncio conflict
        if self.playwright and self.browser:
            try:
                if self.browser.is_connected():
                    # Check if page exists and is open
                    if self.page and not self.page.is_closed():
                        print("   ♻️  Browser already active. Reusing session.")
                        return True
                    else:
                        print("   ♻️  Browser active, opening new page.")
                        self.page = self.browser.new_page(viewport={'width': 1920, 'height': 1080})
                        return True
            except Exception as e:
                print(f"   ⚠️  Existing browser session invalid: {e}. Restarting...")
                self.close()

        # Start new session if none exists
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=headless)
            
            # Create page with consistent viewport
            self.page = self.browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            print("✅ Browser started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start browser: {e}")
            if "asyncio loop" in str(e):
                print("   💡 Tip: This error happens when restarting. The fix above prevents this.")
            return False
    
    def navigate(self, url: str) -> bool:
        """
        Navigate to URL and wait for page load
        
        Args:
            url: Target URL
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            print("❌ Browser not started. Call start_browser() first.")
            return False
            
        try:
            # Navigate with proper waits
            self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for network to be idle (with fallback)
            try:
                self.page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass # Ignore timeout if network is busy (ads, etc.)
            
            print(f"✅ Navigated to: {url}")
            return True
            
        except Exception as e:
            print(f"❌ Navigation failed: {e}")
            return False
    
    def extract_elements(self) -> List[Dict]:
        """
        Extract interactive elements visible in the CURRENT VIEWPORT only.
        """
        if not self.page:
            return []
        
        try:
            js_code = """
            () => {
                const elements = [];
                let id = 0;
                const selectors = [
                    'button', 'a', 'input', 'textarea', 'select',
                    '[role="button"]', '[role="link"]', '[onclick]', '[tabindex]'
                ];
                
                const allElements = new Set();
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => allElements.add(el));
                });
                
                // Get viewport dimensions
                const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

                allElements.forEach(el => {
                    try {
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        
                        // 1. Basic Visibility Check
                        const isVisible = (
                            style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            style.opacity !== '0' &&
                            el.offsetWidth > 0 &&
                            el.offsetHeight > 0
                        );
                        if (!isVisible) return;
                        
                        // 2. 🟢 VIEWPORT CHECK: Element must be within the current window
                        // We add a small buffer (-10/+10) to catch partially visible edge elements
                        const inViewport = (
                            rect.top < vh && 
                            rect.bottom > 0 &&
                            rect.left < vw && 
                            rect.right > 0
                        );
                        if (!inViewport) return;

                        // 3. Extract Text (with accessibility fix)
                        let text = "";
                        const ariaLabel = el.getAttribute('aria-label');
                        const title = el.getAttribute('title');
                        const placeholder = el.getAttribute('placeholder');
                        const value = el.value;
                        const innerText = el.innerText || el.textContent;
                        
                        if (ariaLabel) text = ariaLabel;
                        else if (title) text = title;
                        else if (placeholder) text = placeholder;
                        else if (value && el.tagName === 'INPUT') text = value;
                        else if (innerText) text = innerText;
                        
                        text = (text || "").replace(/\\s+/g, ' ').trim().substring(0, 100);

                        elements.push({
                            id: id++,
                            tag: el.tagName.toLowerCase(),
                            text: text,
                            type: el.tagName === 'INPUT' ? (el.type || 'text') : 'button',
                            bbox: {
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                w: Math.round(rect.width),
                                h: Math.round(rect.height)
                            },
                            html_id: el.id || '',
                            html_classes: el.className || ''
                        });
                        
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                return elements.sort((a, b) => a.id - b.id);
            }
            """
            
            elements = self.page.evaluate(js_code)

            try:
                with open('dom_elements.json', 'w', encoding='utf-8') as f:
                    json.dump(elements, f, indent=2, ensure_ascii=False)
                # print(f"💾 Debug: Saved to dom_elements.json") 
            except Exception as save_err:
                print(f"⚠️ Failed to save debug JSON: {save_err}")
                
            print(f"✅ Extracted {len(elements)} elements (Viewport only)")
            return elements
            
        except Exception as e:
            print(f"❌ Failed to extract elements: {e}")
            return []
    
    def click_by_id(self, element_id: int, elements: List[Dict]) -> bool:
        """
        Click an element by its ID using mouse coordinates
        
        Args:
            element_id: Element ID from extract_elements()
            elements: List of elements from extract_elements()
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            print("❌ Browser not started.")
            return False
        
        try:
            # Find element with matching ID
            element = next((el for el in elements if el['id'] == element_id), None)
            
            if not element:
                print(f"❌ Element with ID {element_id} not found")
                return False
            
            # Calculate center point
            bbox = element['bbox']
            center_x = bbox['x'] + bbox['w'] / 2
            center_y = bbox['y'] + bbox['h'] / 2
            
            # Click at center
            self.page.mouse.click(center_x, center_y)
            
            # Wait after click
            self.page.wait_for_timeout(500)
            
            print(f"✅ Clicked element {element_id} ({element['tag']}) at ({center_x:.0f}, {center_y:.0f})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to click element: {e}")
            return False
    
    def take_screenshot(self) -> Optional[bytes]:
        """
        Take screenshot of the CURRENT VIEWPORT only (not full page).
        Prevents OOM errors on long pages.
        """
        if not self.page:
            print("❌ Browser not started.")
            return None
        
        try:
            # 🟢 FIX: full_page=False captures only the visible window (1920x1080)
            screenshot_bytes = self.page.screenshot(full_page=False)
            print("✅ Screenshot captured (Viewport only)")
            return screenshot_bytes
            
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")
            return None
    
    def get_page_info(self) -> Dict:
        """
        Get current page information
        
        Returns:
            Dictionary with title, url, and viewport dimensions
        """
        if not self.page:
            return {
                'title': '',
                'url': '',
                'viewport': {'width': 0, 'height': 0}
            }
        
        try:
            return {
                'title': self.page.title(),
                'url': self.page.url,
                'viewport': {
                    'width': self.page.viewport_size['width'],
                    'height': self.page.viewport_size['height']
                }
            }
        except Exception as e:
            print(f"❌ Failed to get page info: {e}")
            return {
                'title': '',
                'url': '',
                'viewport': {'width': 0, 'height': 0}
            }
    
    def type_text(self, text: str) -> bool:
        '''Type text into focused element'''
        if not self.page:
            return False
        try:
            self.page.keyboard.type(text, delay=50)
            print(f"✅ Typed: {text}")
            return True
        except Exception as e:
            print(f"❌ Type failed: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        '''Press keyboard key (Enter, Tab, Escape, etc.)'''
        if not self.page:
            return False
        try:
            self.page.keyboard.press(key)
            print(f"✅ Pressed: {key}")
            return True
        except Exception as e:
            print(f"❌ Key press failed: {e}")
            return False
    
    def scroll_page(self, direction: str = "down", pixels: int = 500) -> bool:
        '''Scroll the page'''
        if not self.page:
            return False
        try:
            if direction.lower() == "down":
                self.page.evaluate(f"window.scrollBy(0, {pixels})")
            elif direction.lower() == "up":
                self.page.evaluate(f"window.scrollBy(0, -{pixels})")
            print(f"✅ Scrolled {direction}")
            return True
        except Exception as e:
            print(f"❌ Scroll failed: {e}")
            return False
            
    def close(self):
        """Close browser and cleanup resources"""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            
            print("✅ Browser closed")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
        
        finally:
            self.page = None
            self.browser = None
            self.playwright = None


# Interactive test function
if __name__ == "__main__":
    print("=" * 60)
    print("DOM EXTRACTOR - Interactive Test")
    print("=" * 60)
    
    extractor = DOMExtractor()
    
    try:
        # User input
        url = input("\nEnter URL to analyze (e.g., https://example.com): ").strip()
        
        if not url:
            print("❌ No URL provided. Exiting.")
            exit(1)
        
        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print(f"\n🌐 Target URL: {url}")
        
        # Start browser
        print("\n📦 Starting browser...")
        if not extractor.start_browser(headless=False):
            exit(1)
        
        # Navigate
        print(f"\n🔄 Navigating to {url}...")
        if not extractor.navigate(url):
            exit(1)
        
        # Get page info
        page_info = extractor.get_page_info()
        print(f"\n📄 Page Title: {page_info['title']}")
        print(f"📐 Viewport: {page_info['viewport']['width']}x{page_info['viewport']['height']}")
        
        # Extract elements
        print("\n🔍 Extracting interactive elements...")
        elements = extractor.extract_elements()
        
        print(f"\n✅ Found {len(elements)} interactive elements\n")
        
        if elements:
            # Show first 5
            print("=" * 60)
            print("First 5 elements:")
            print("=" * 60)
            for el in elements[:5]:
                bbox_str = f"({el['bbox']['x']}, {el['bbox']['y']}) {el['bbox']['w']}x{el['bbox']['h']}"
                text_preview = el['text'][:30] if el['text'] else '(no text)'
                print(f"  [{el['id']:3d}] {el['tag']:8s} {el['type']:10s} '{text_preview}' at {bbox_str}")
            
            if len(elements) > 5:
                print(f"  ... and {len(elements) - 5} more elements")
            
            # Save all to JSON
            output_file = 'dom_elements.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(elements, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Saved all elements to: {output_file}")
            
            # Optional: Take screenshot
            print("\n📸 Taking screenshot...")
            screenshot = extractor.take_screenshot()
            if screenshot:
                screenshot_file = 'dom_screenshot.png'
                with open(screenshot_file, 'wb') as f:
                    f.write(screenshot)
                print(f"💾 Saved screenshot to: {screenshot_file}")
        
        else:
            print("⚠️ No interactive elements found on this page.")
        
        # Wait before closing
        input("\n⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()
        print("\n✅ Done!")