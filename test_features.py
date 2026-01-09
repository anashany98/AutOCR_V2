import os
import cv2
import numpy as np
import logging
from modules.image_utils import preprocess_image_for_ocr, deskew_image

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_image_preprocessing():
    print("\n" + "="*50)
    print("🧪 TESTING IMAGE PRE-PROCESSING (Deskew & Denoise)")
    print("="*50)

    # 1. Create a dummy black image with a white rotated rectangle (simulating text block)
    # Using a large white bar on black background is easy for contour detection
    img = np.zeros((600, 600), dtype=np.uint8)
    
    # Draw a rotated rectangle: Center(300,300), Size(100, 400), Angle=30 degrees
    rect = ((300, 300), (100, 400), 30)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255), -1)
    
    # Add some "noise" (random white dots)
    noise = np.random.randint(0, 2, (600, 600)) * 255
    noise_mask = np.random.rand(600, 600) > 0.99  # 1% noise
    img[noise_mask] = 255

    # Save dummy skewed image
    test_path = "test_skew_input.png"
    cv2.imwrite(test_path, img)
    print(f"✅ Generated synthetic test image: {test_path} (Skewed ~30 deg + Noise)")

    # 2. Check Deskew Logic explicitly
    print("\n🔍 Checking Rotation Detection...")
    _, detected_angle = deskew_image(img)
    print(f"   -> Algorithm detected angle: {detected_angle:.2f} degrees")
    
    if abs(detected_angle) > 5:
        print("   ✅ Angle detection works (result is significant).")
    else:
        print("   ⚠️ Angle detection low (might be ambiguous shape).")

    # 3. Run Full Pipeline
    print("\n⚙️ Running 'preprocess_image_for_ocr' pipeline...")
    try:
        processed_path = preprocess_image_for_ocr(test_path, deskew=True, denoise=True)
        print(f"   -> Processed file saved to: {processed_path}")
        
        if processed_path != test_path and os.path.exists(processed_path):
            result_img = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
            print(f"   -> Result image shape: {result_img.shape}")
            print("   ✅ Pipeline executed successfully (File created).")
            
            # Clean up temp file
            os.remove(processed_path)
            print("   -> Temporary output cleaned up.")
        else:
            print("   ❌ Pipeline failed (Returned original path or file missing).")

    except Exception as e:
        print(f"   ❌ Execution crashed: {e}")

    # Cleanup input
    if os.path.exists(test_path):
        os.remove(test_path)
        print("\n✅ Verification Complete: Input test file cleaned up.")

if __name__ == "__main__":
    test_image_preprocessing()
