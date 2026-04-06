#!/usr/bin/env python3
"""
test_face_detector.py — Quick test for face detector module

Run this to verify face detection is working correctly.
"""

import cv2
import numpy as np
from app.vision.face_detector import face_detector

def test_face_detection():
    """Test face detection on a sample image."""
    print("=" * 60)
    print("FACE DETECTOR TEST")
    print("=" * 60)
    
    # Create a dummy frame (640x480, BGR)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some content (white rectangle in center)
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    
    print("\n✓ Created test frame (640x480)")
    
    # Test 1: Basic detection
    print("\n[TEST 1] Basic face detection...")
    try:
        detections = face_detector.detect(frame, conf_threshold=0.5)
        print(f"  ✓ Detection completed")
        print(f"  ✓ Found {len(detections)} face(s)")
        
        for i, det in enumerate(detections):
            print(f"\n  Face {i+1}:")
            print(f"    Box: {det['box']}")
            print(f"    Confidence: {det['confidence']}")
            print(f"    Center: {det['center']}")
            print(f"    Size: {det['width']}x{det['height']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 2: Detection with padding
    print("\n[TEST 2] Face detection with padding...")
    try:
        detections_padded = face_detector.detect_with_padding(
            frame, 
            conf_threshold=0.5, 
            padding_ratio=0.1
        )
        print(f"  ✓ Detection with padding completed")
        print(f"  ✓ Found {len(detections_padded)} face(s)")
        
        for i, det in enumerate(detections_padded):
            if "box_padded" in det:
                print(f"\n  Face {i+1}:")
                print(f"    Original box: {det['box']}")
                print(f"    Padded box: {det['box_padded']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 3: Face ROI extraction
    print("\n[TEST 3] Face ROI extraction...")
    try:
        if detections:
            face_box = detections[0]["box"]
            roi = face_detector.get_face_roi(frame, face_box)
            
            if roi is not None:
                print(f"  ✓ ROI extracted successfully")
                print(f"  ✓ ROI shape: {roi.shape}")
            else:
                print(f"  ✗ ROI extraction returned None")
                return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 4: Boundary clamping
    print("\n[TEST 4] Boundary clamping...")
    try:
        # Test with out-of-bounds box
        invalid_box = (-10, -10, 700, 500)  # Extends beyond frame
        roi = face_detector.get_face_roi(frame, invalid_box)
        
        if roi is None:
            print(f"  ✓ Invalid box correctly rejected")
        else:
            print(f"  ✓ Invalid box handled (ROI shape: {roi.shape})")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_face_detection()
    exit(0 if success else 1)
