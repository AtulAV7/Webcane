"""
SoM (Set-of-Mark) Annotator
Draws numbered boxes on interactive elements for vision LLM analysis
"""

from PIL import Image, ImageDraw, ImageFont
import io
from typing import List, Dict, Tuple, Optional

class SoMAnnotator:
    """Annotate screenshots with numbered boxes for vision-based automation"""
    
    def __init__(self):
        """Initialize drawing parameters and settings"""
        # Visual style parameters
        self.box_color = '#FF0000'  # Red
        self.box_thickness = 2      # Thinner to see context
        self.text_color = '#FFFFFF' # White text
        self.text_bg_color = '#FF0000' # Red background
        self.font_size = 18         # Readable size
        
        # Load font using your original robust logic
        self.font = self._load_font()
        
        print("✅ SoMAnnotator initialized (Outside Labels Mode)")
    
    def _load_font(self):
        """Load font for text annotations with fallback paths"""
        try:
            # Common font paths (Linux, macOS, Windows)
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\arial.ttf",
                "arial.ttf",
                "DejaVuSans.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    return ImageFont.truetype(font_path, self.font_size)
                except:
                    continue
            
            # Fallback
            print("⚠️  Using default font (no TrueType font found)")
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"⚠️  Font loading error: {e}")
            return ImageFont.load_default()

    def create_annotated_screenshot(
        self, 
        screenshot_bytes: bytes, 
        elements: List[Dict]
    ) -> Tuple[bytes, List[Dict]]:
        """
        Draws boxes labeled with INDEX (0, 1, 2...) instead of ID.
        """
        if not screenshot_bytes:
            return b'', []
        if not elements:
            return screenshot_bytes, []
        
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            draw = ImageDraw.Draw(image)
            
            # Draw each element using its list INDEX (0, 1, 2...)
            for idx, element in enumerate(elements):
                # Pass 'str(idx)' as the label so the image shows 0, 1, 2...
                self._draw_box_outside(draw, element, str(idx), self.box_color)
            
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue(), elements
            
        except Exception as e:
            print(f"❌ Failed to annotate screenshot: {e}")
            return screenshot_bytes, []

    def filter_and_annotate(
        self,
        screenshot_bytes: bytes,
        elements: List[Dict],
        max_elements: int = 30
    ) -> Tuple[bytes, List[Dict]]:
        """
        Filter elements (viewport/size) and annotate with sequential numbers.
        """
        if not elements:
            return screenshot_bytes, []
            
        # 1. Filter out tiny elements
        valid_elements = [
            el for el in elements 
            if el['bbox']['w'] >= 10 and el['bbox']['h'] >= 10
        ]
        
        # 2. Sort (Top -> Bottom, Left -> Right)
        sorted_elements = sorted(
            valid_elements, 
            key=lambda el: (el['bbox']['y'], el['bbox']['x'])
        )
        
        # 3. Limit count
        filtered_elements = sorted_elements[:max_elements]
        
        # 4. Annotate (This calls create_annotated_screenshot internally)
        return self.create_annotated_screenshot(screenshot_bytes, filtered_elements)

    def _draw_box_outside(
        self, 
        draw: ImageDraw.ImageDraw, 
        element: Dict, 
        label: str,
        color: str
    ):
        """
        Draws the box with the label 'popped out' on top-left.
        If element is at the very top of screen, flips label inside.
        """
        try:
            bbox = element['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # 1. Draw the Outline (Bounding Box)
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=self.box_thickness)
            
            # 2. Calculate Label Dimensions
            padding = 4
            try:
                # Newer Pillow
                left, top, right, bottom = draw.textbbox((0, 0), label, font=self.font)
                text_w = right - left
                text_h = bottom - top
            except:
                # Older Pillow
                text_w, text_h = draw.textsize(label, font=self.font)
            
            # 3. Calculate Tag Position (Outside - Top Left)
            tag_x = x
            tag_y = y - (text_h + padding * 2)
            
            # 4. Safety Flip: If tag goes off-screen top, push it inside
            if tag_y < 0:
                tag_y = y
                
            # 5. Draw Label Background
            draw.rectangle(
                [
                    (tag_x, tag_y), 
                    (tag_x + text_w + padding * 2, tag_y + text_h + padding * 2)
                ], 
                fill=self.text_bg_color,
                outline=color
            )
            
            # 6. Draw Number
            draw.text(
                (tag_x + padding, tag_y + padding), 
                label, 
                fill=self.text_color, 
                font=self.font
            )
        except Exception as e:
            print(f"⚠️ Failed to draw element {label}: {e}")

    def create_multi_color_annotation(
        self,
        screenshot_bytes: bytes,
        elements: List[Dict],
        color_scheme: str = 'rainbow'
    ) -> Tuple[bytes, List[Dict]]:
        """
        (Preserved) Create annotation with multiple colors
        """
        if not screenshot_bytes or not elements:
            return screenshot_bytes, elements
        
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            draw = ImageDraw.Draw(image)
            
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF']
            
            for idx, element in enumerate(elements):
                color = colors[idx % len(colors)]
                # Use the new drawing method here too
                self._draw_box_outside(draw, element, str(idx), color)
            
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue(), elements
            
        except Exception as e:
            print(f"❌ Multi-color annotation failed: {e}")
            return screenshot_bytes, elements


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    
    print("=" * 70)
    print("SoM ANNOTATOR TEST")
    print("Set-of-Mark: Draw numbered boxes on interactive elements")
    print("=" * 70)
    
    extractor = DOMExtractor()
    annotator = SoMAnnotator()
    
    try:
        # User input
        url = input("\nEnter URL to annotate (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
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
        
        print("✅ Screenshot captured")
        
        # Test 1: Full annotation
        print("\n" + "=" * 70)
        print("TEST 1: Full Annotation (all elements)")
        print("=" * 70)
        
        annotated_img, mapping = annotator.create_annotated_screenshot(
            screenshot,
            elements
        )
        
        with open('som_annotated.png', 'wb') as f:
            f.write(annotated_img)
        
        print(f"\n✅ Saved to: som_annotated.png")
        print(f"   📊 Annotated {len(mapping)} elements")
        print(f"   🔍 Open this file to verify box positions")
        
        # Test 2: Filtered annotation
        print("\n" + "=" * 70)
        print("TEST 2: Filtered Annotation (first 10 elements)")
        print("=" * 70)
        
        filtered_img, filtered_elements = annotator.filter_and_annotate(
            screenshot,
            elements,
            max_elements=10
        )
        
        with open('som_filtered.png', 'wb') as f:
            f.write(filtered_img)
        
        print(f"\n✅ Saved to: som_filtered.png")
        print(f"   📊 Annotated {len(filtered_elements)} elements (filtered)")
        
        # Show filtered element details
        print("\n📋 Filtered Elements:")
        for el in filtered_elements[:5]:
            text = el['text'][:30] if el['text'] else '(no text)'
            print(f"   [{el['id']}] {el['tag']:8s} '{text}' at ({el['bbox']['x']}, {el['bbox']['y']})")
        
        if len(filtered_elements) > 5:
            print(f"   ... and {len(filtered_elements) - 5} more")
        
        # Test 3: Multi-color annotation
        print("\n" + "=" * 70)
        print("TEST 3: Multi-Color Annotation (first 10 with rainbow colors)")
        print("=" * 70)
        
        rainbow_img, rainbow_elements = annotator.create_multi_color_annotation(
            screenshot,
            elements[:10],
            color_scheme='rainbow'
        )
        
        with open('som_rainbow.png', 'wb') as f:
            f.write(rainbow_img)
        
        print(f"\n✅ Saved to: som_rainbow.png")
        print(f"   🌈 Multi-color annotation with {len(rainbow_elements)} elements")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"\n✅ Generated 3 annotated screenshots:")
        print(f"   1. som_annotated.png  - All {len(elements)} elements")
        print(f"   2. som_filtered.png   - Top 10 elements (filtered)")
        print(f"   3. som_rainbow.png    - Top 10 with colors")
        print(f"\n💡 Open these files to verify:")
        print(f"   - Boxes are correctly positioned")
        print(f"   - Numbers are clearly visible")
        print(f"   - No overlapping issues")
        
        # Wait for user
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