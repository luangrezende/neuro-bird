"""
Score detection test with real-time visual feedback.
"""
import time
import traceback
from typing import Optional

import cv2
import mss
import numpy as np

from modules.utils.config import config
from modules.utils.visual_renderer import VisualRenderer
from modules.vision.score_detector import ScoreDetector


class ScoreDetectorTest:
    def __init__(self):
        self.detector = ScoreDetector()
        self.renderer = VisualRenderer()
        self.sct = mss.mss()
        
        capture_config = config.get_section('vision')['screen_capture']['region']
        self.capture_region = {
            "top": capture_config['top'],
            "left": capture_config['left'], 
            "width": capture_config['width'],
            "height": capture_config['height']
        }
        
        self.display_config = config.get_section('vision')['display']
        self.test_settings = self.display_config['test_settings']
        
        self.frame_count = self.display_config['defaults']['initial_confidence']
        self.start_time: Optional[float] = None
        self.ocr_times = []
        
    def run(self) -> None:
        self.start_time = time.time()
        
        try:
            self._main_loop()
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            
    def _main_loop(self) -> None:
        while True:
            screenshot = self.sct.grab(self.capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            annotated_frame = frame.copy()
            
            self._update_fps()
            self._process_ocr(frame)
            self._update_display_metrics()
            self._render_frame(annotated_frame)
            
            if self._should_quit():
                break
                
    def _update_fps(self) -> None:
        self.frame_count += self.test_settings['frame_increment']
        
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            zero_threshold = self.display_config['defaults']['initial_confidence']
            
            if elapsed_time > zero_threshold:
                fps = self.frame_count / elapsed_time
                self.renderer.update_fps(fps)
            
    def _process_ocr(self, frame: np.ndarray) -> None:
        ocr_interval = self.test_settings['ocr_interval']
        
        if self.frame_count % ocr_interval == 0:
            ocr_start = time.time()
            self.detector.detect_and_update(frame)
            ocr_end = time.time()
            
            ocr_time = (ocr_end - ocr_start) * self.test_settings['ms_conversion_factor']
            self.ocr_times.append(ocr_time)
            
    def _update_display_metrics(self) -> None:
        score = self.detector.game_state.get_score()
        self.renderer.update_score(score)
        
        if self.ocr_times:
            recent_times = self.test_settings['recent_times_average']
            avg_ocr_time = sum(self.ocr_times[-recent_times:]) / min(len(self.ocr_times), recent_times)
            self.renderer.update_ocr_time(avg_ocr_time)
            
    def _render_frame(self, frame: np.ndarray) -> None:
        self.renderer.draw_all_info(frame)
        window_title = self.renderer.get_window_title()
        cv2.imshow(window_title, frame)
        
    def _should_quit(self) -> bool:
        key_wait = self.test_settings['key_wait_ms']
        key_mask = self.test_settings['key_mask']
        quit_key = self.test_settings['quit_key']
        
        return cv2.waitKey(key_wait) & key_mask == ord(quit_key)


def main() -> None:
    try:
        test = ScoreDetectorTest()
        test.run()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()