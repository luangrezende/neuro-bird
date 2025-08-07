import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from utils.config import config

class VisualRenderer:
    def __init__(self):
        self.display_config = config.get_section('vision')['display']
        self.current_fps = 0.0
        self.current_score = self.display_config['defaults']['invalid_score']
        self.current_ocr_time = 0.0
        
    def update_fps(self, fps):
        self.current_fps = fps
        
    def update_score(self, score):
        self.current_score = score
        
    def update_ocr_time(self, ocr_time):
        self.current_ocr_time = ocr_time
        
    def draw_all_info(self, frame):
        self.draw_fps(frame, self.current_fps)
        self.draw_score_info(frame, self.current_score)
        self.draw_ocr_time(frame, self.current_ocr_time)
        
    def draw_fps(self, frame, fps):
        colors = self.display_config['colors']
        font = getattr(cv2, self.display_config['font_face'])
        font_size = self.display_config['font_sizes']['medium']
        font_thickness = self.display_config['font_thickness']
        fps_pos = self.display_config['text_positions']
        
        cv2.putText(frame, "FPS: ", fps_pos['fps_label'], 
                   font, font_size, colors['green'], font_thickness)
        cv2.putText(frame, f"{fps:.1f}", fps_pos['fps_value'], 
                   font, font_size, colors['red'], font_thickness)
    
    def draw_score_info(self, frame, score):
        colors = self.display_config['colors']
        font = getattr(cv2, self.display_config['font_face'])
        font_size = self.display_config['font_sizes']['medium']
        font_thickness = self.display_config['font_thickness']
        score_pos = self.display_config['text_positions']
        
        cv2.putText(frame, "Score: ", score_pos['score_label'], 
                   font, font_size, colors['green'], font_thickness)
        
        invalid_score = self.display_config['defaults']['invalid_score']
        score_text = f"{score}"
        score_color = colors['red']
            
        cv2.putText(frame, score_text, score_pos['score_value'], 
                   font, font_size, score_color, font_thickness)
    
    def draw_ocr_time(self, frame, avg_ocr_time):
        colors = self.display_config['colors']
        font = getattr(cv2, self.display_config['font_face'])
        font_size = self.display_config['font_sizes']['medium']
        font_thickness = self.display_config['font_thickness']
        ocr_pos = self.display_config['text_positions']
        
        cv2.putText(frame, "OCR Time: ", ocr_pos['ocr_time_label'], 
                   font, font_size, colors['green'], font_thickness)
        cv2.putText(frame, f"{avg_ocr_time:.1f}ms", ocr_pos['ocr_time_value'], 
                   font, font_size, colors['red'], font_thickness)
    
    def get_window_title(self):
        return self.display_config['window_title']
