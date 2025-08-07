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
        
        display_config = config.get_section('vision')['display']
        self.last_valid_score = display_config['defaults']['invalid_score']
        self.last_valid_bbox = None
        
        self.detection_config = config.get_section('vision')['score_detection']
        self.display_config = display_config
        
    def extract_score_with_ocr(self, frame):
        h, w = frame.shape[:2]
        
        score_region_height = self.detection_config['region_height']
        score_region_width = self.detection_config['region_width']
        start_y_offset = self.detection_config['start_y_offset']
        scale_factor = self.detection_config['scale_factor']
        
        start_y = max(0, start_y_offset)
        end_y = min(h, start_y + score_region_height)
        start_x = max(0, (w - score_region_width) // 2)
        end_x = min(w, start_x + score_region_width)
        
        roi = frame[start_y:end_y, start_x:end_x]
        
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
            detail=self.display_config['defaults']['detail_level']
        )
        
        best_score_result = None
        best_confidence = self.display_config['defaults']['initial_confidence']
        
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
            score, bbox, conf = best_score_result
            self.last_valid_score = score
            self.last_valid_bbox = bbox
            return score, bbox
        
        invalid_score = self.display_config['defaults']['invalid_score']
        if self.last_valid_score >= -invalid_score:
            return self.last_valid_score, self.last_valid_bbox
        
        return invalid_score, None
    
    def detect_and_draw_score(self, frame):
        annotated_frame = frame.copy()
        
        score, bbox = self.extract_score_with_ocr(frame)
        
        if bbox is not None:
            x, y, w, h = bbox
            colors = self.display_config['colors']
            thickness = self.display_config['rectangle_thickness']
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), colors['green'], thickness)
            
            invalid_score = self.display_config['defaults']['invalid_score']
            if score >= -invalid_score:
                label = f"Score: {score}"
                color = colors['green']
            else:
                label = "Score region"
                color = colors['yellow']
            
            font = getattr(cv2, self.display_config['font_face'])
            font_size = self.display_config['font_sizes']['medium']
            font_thickness = self.display_config['font_thickness']
            label_offset = self.display_config['score_label_offset']
                
            cv2.putText(annotated_frame, label, (x + label_offset[0], y - label_offset[1]), 
                       font, font_size, color, font_thickness)
        
        return annotated_frame, score
    
    def cleanup(self):
        invalid_score = self.display_config['defaults']['invalid_score']
        self.last_valid_score = invalid_score
        self.last_valid_bbox = None
