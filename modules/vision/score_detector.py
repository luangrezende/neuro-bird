import cv2
import numpy as np
import easyocr
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import config

class ScoreDetector:
    def __init__(self):
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available but required. Install PyTorch with CUDA.")
        
        ocr_config = config.get_section('vision')['ocr']
        self.reader = easyocr.Reader(
            ocr_config['languages'], 
            gpu=ocr_config['gpu'], 
            verbose=ocr_config['verbose']
        )
        
        defaults = config.get_section('vision')['display']['defaults']
        self.last_valid_score = defaults['invalid_score']
        self.last_valid_bbox = None
        
        self.detection_config = config.get_section('vision')['score_detection']
        self.invalid_score = defaults['invalid_score']
        self.initial_confidence = defaults['initial_confidence']
        self.detail_level = defaults['detail_level']
        
    def extract_score_with_ocr(self, frame):
        h, w = frame.shape[:2]
        
        score_region_height = self.detection_config['region_height']
        score_region_width = self.detection_config['region_width']
        start_y_offset = self.detection_config['start_y_offset']
        start_x_offset = self.detection_config['start_x_offset']
        scale_factor = self.detection_config['scale_factor']
        
        start_y = max(0, start_y_offset)
        end_y = min(h, start_y + score_region_height)
        
        if start_x_offset == -1:
            start_x = max(0, (w - score_region_width) // 2)
        else:
            start_x = max(0, start_x_offset)
        
        roi = frame[start_y:end_y, start_x:start_x + score_region_width]
        
        interpolation_map = {
            'LANCZOS4': cv2.INTER_LANCZOS4,
            'CUBIC': cv2.INTER_CUBIC,
            'LINEAR': cv2.INTER_LINEAR
        }
        interp_method = interpolation_map.get(self.detection_config['interpolation'], cv2.INTER_LANCZOS4)
        roi_resized = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=interp_method)
        
        ocr_config = config.get_section('vision')['ocr']
        results = self.reader.readtext(
            roi_resized,
            text_threshold=ocr_config['text_threshold'],
            width_ths=ocr_config['width_threshold'],
            height_ths=ocr_config['height_threshold'],
            min_size=ocr_config['min_size'],
            detail=self.detail_level
        )
        
        best_score_result = None
        best_confidence = self.initial_confidence
        
        for (bbox, text, confidence) in results:
            conf_value = float(confidence)
            
            score_match = re.search(self.detection_config['pattern'], text, re.IGNORECASE)
            if score_match:
                score_value = int(score_match.group(1))
                score_range = self.detection_config['score_range']
                if score_range['min'] <= score_value <= score_range['max']:
                    if conf_value > ocr_config['confidence_threshold'] and (best_score_result is None or conf_value > best_confidence):
                        points = np.array(bbox).astype(int)
                        points = points // scale_factor
                        
                        x = min(points[:, 0]) + start_x
                        y = min(points[:, 1]) + start_y
                        w = max(points[:, 0]) - min(points[:, 0])
                        h = max(points[:, 1]) - min(points[:, 1])
                        
                        best_score_result = (score_value, (x, y, w, h), conf_value)
                        best_confidence = conf_value
        
        if best_score_result:
            score, bbox, _ = best_score_result
            self.last_valid_score = score
            self.last_valid_bbox = bbox
            return score, bbox
        
        if self.last_valid_score >= -self.invalid_score:
            return self.last_valid_score, self.last_valid_bbox
        
        return self.invalid_score, None
    
    def cleanup(self):
        self.last_valid_score = self.invalid_score
        self.last_valid_bbox = None
