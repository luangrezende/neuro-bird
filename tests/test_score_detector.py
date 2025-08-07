import cv2
import numpy as np
import sys
import os
import mss
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from vision import ScoreDetector
from utils.config import config

def main():
    detector = ScoreDetector()
    
    sct = mss.mss()
    
    capture_config = config.get_section('vision')['screen_capture']['region']
    capture_region = {
        "top": capture_config['top'],
        "left": capture_config['left'], 
        "width": capture_config['width'],
        "height": capture_config['height']
    }
    
    display_config = config.get_section('vision')['display']
    invalid_score = display_config['defaults']['invalid_score']
    initial_confidence = display_config['defaults']['initial_confidence']
    fps_target_fallback = display_config['defaults']['fps_target_fallback']
    test_settings = display_config['test_settings']
    
    print(f"Using capture region: {capture_region}")
    print(f"App: {config.get('app.name')} v{config.get('app.version')}")
    print(f"GPU enabled: {config.get('vision.ocr.gpu')}")
    print("Press 'q' to quit")
    
    frame_count = initial_confidence
    start_time = time.time()
    last_score = invalid_score
    last_bbox = None
    ocr_times = []
    ocr_count = initial_confidence
    fps_target = config.get('vision.screen_capture.fps_target', fps_target_fallback)
    
    try:
        while True:
            screenshot = sct.grab(capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            annotated_frame = frame.copy()
            
            frame_count += test_settings['frame_increment']
            elapsed_time = time.time() - start_time
            zero_threshold = initial_confidence
            if elapsed_time > zero_threshold:
                fps = frame_count / elapsed_time
                
                colors = display_config['colors']
                font = getattr(cv2, display_config['font_face'])
                font_size = display_config['font_sizes']['medium']
                font_thickness = display_config['font_thickness']
                fps_pos = display_config['text_positions']
                
                cv2.putText(annotated_frame, "FPS: ", fps_pos['fps_label'], 
                           font, font_size, colors['green'], font_thickness)
                cv2.putText(annotated_frame, f"{fps:.1f}", fps_pos['fps_value'], 
                           font, font_size, colors['red'], font_thickness)
            
            if frame_count % fps_target == 0:
                ocr_start = time.time()
                score, bbox = detector.extract_score_with_ocr(frame)
                ocr_end = time.time()
                
                ocr_time = (ocr_end - ocr_start) * test_settings['ms_conversion_factor']
                ocr_times.append(ocr_time)
                ocr_count += test_settings['ocr_count_increment']
                
                if bbox is not None:
                    last_score = score
                    last_bbox = bbox
            
            if last_bbox is not None:
                x, y, w, h = last_bbox
                colors = display_config['colors']
                thickness = display_config['rectangle_thickness']
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), colors['green'], thickness)
                
                label_text = "Score region"
                color = colors['green']
                font = getattr(cv2, display_config['font_face'])
                font_size = display_config['font_sizes']['small']
                font_thickness = display_config['font_thickness']
                label_offset = display_config['score_label_offset']
                
                cv2.putText(annotated_frame, label_text, (x, y + h + label_offset[1]), 
                           font, font_size, color, font_thickness)
            
            if last_score >= -invalid_score:
                colors = display_config['colors']
                font = getattr(cv2, display_config['font_face'])
                font_size = display_config['font_sizes']['medium']
                font_thickness = display_config['font_thickness']
                score_pos = display_config['text_positions']
                
                cv2.putText(annotated_frame, "Score: ", score_pos['score_label'], 
                           font, font_size, colors['green'], font_thickness)
                cv2.putText(annotated_frame, f"{last_score}", score_pos['score_value'], 
                           font, font_size, colors['red'], font_thickness)
            
            if ocr_times:
                recent_times = test_settings['recent_times_average']
                avg_ocr_time = sum(ocr_times[-recent_times:]) / min(len(ocr_times), recent_times)
                
                colors = display_config['colors']
                font = getattr(cv2, display_config['font_face'])
                font_size = display_config['font_sizes']['medium']
                font_thickness = display_config['font_thickness']
                ocr_pos = display_config['text_positions']
                
                cv2.putText(annotated_frame, "OCR: ", ocr_pos['ocr_time_label'], 
                           font, font_size, colors['green'], font_thickness)
                cv2.putText(annotated_frame, f"{avg_ocr_time:.1f}ms", ocr_pos['ocr_time_value'], 
                           font, font_size, colors['red'], font_thickness)
            
            window_title = display_config['window_title']
            cv2.imshow(window_title, annotated_frame)
            
            key_wait = test_settings['key_wait_ms']
            key_mask = test_settings['key_mask']
            quit_key = test_settings['quit_key']
            if cv2.waitKey(key_wait) & key_mask == ord(quit_key):
                break
    
    except Exception as e:
        pass
    
    finally:
        detector.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
