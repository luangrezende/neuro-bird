"""
Visual rendering utilities for score detection feedback and debugging.
"""
from typing import Any, Dict, List, Optional, Tuple

import cv2

class VisualRenderer:
    def __init__(self, config_section: str = 'vision'):
        from .config import config
        self.display_config = config.get_section(config_section)['display']
        self.module_config = config.get_section(config_section)
        self.metrics = {}
        
    def update_metric(self, key: str, value: Any):
        self.metrics[key] = value
        
    def draw_all_info(self, frame):
        self.draw_regions(frame)
        self.draw_metrics(frame)
        
    def draw_regions(self, frame):
        for section_name, section_config in self.module_config.items():
            if isinstance(section_config, dict) and self._is_region_config(section_config):
                self._draw_region_from_config(frame, section_name, section_config)
                
    def _is_region_config(self, config: Dict) -> bool:
        region_keys = self.display_config['region_detection']['keys']
        return any(key in config for key in region_keys)
        
    def _draw_region_from_config(self, frame, region_name: str, region_config: Dict):
        colors = self.display_config['colors']
        h, w = frame.shape[:2]
        
        center_indicator = self.display_config['region_detection']['center_indicator']
        
        region_height = region_config['region_height']
        region_width = region_config['region_width']
        start_y_offset = region_config['start_y_offset']
        start_x_offset = region_config['start_x_offset']
        
        start_y = max(0, start_y_offset)
        end_y = min(h, start_y + region_height)
        
        if start_x_offset == center_indicator:
            start_x = max(0, (w - region_width) // 2)
        else:
            start_x = max(0, start_x_offset)
            
        end_x = min(w, start_x + region_width)
        
        rect_props = self.display_config['rectangle_properties']
        thickness = rect_props['region_thickness']
        color_name = rect_props['region_color']
        color = colors[color_name]
        
        label = f"{region_name.replace('_', ' ').title()} Region"
        self.draw_rectangle(frame, (start_x, start_y), (end_x, end_y), color, thickness, label)
        
    def draw_metrics(self, frame):
        for key, value in self.metrics.items():
            self.draw_metric(frame, key, value)
            
    def draw_metric(self, frame, metric_name: str, value: Any):
        if not self.display_config[f'show_{metric_name}']:
            return
            
        colors = self.display_config['colors']
        font = getattr(cv2, self.display_config['font_face'])
        font_size = self.display_config['font_sizes']['medium']
        font_thickness = self.display_config['font_thickness']
        
        positions = self.display_config['text_positions']
        label_pos = positions.get(f'{metric_name}_label')
        value_pos = positions.get(f'{metric_name}_value')
        
        if label_pos and value_pos:
            cv2.putText(frame, f"{metric_name.replace('_', ' ').title()}: ", 
                       tuple(label_pos), font, font_size, colors['green'], font_thickness)
            
            formatted_value = self.format_metric_value(metric_name, value)
            
            cv2.putText(frame, formatted_value, tuple(value_pos), 
                       font, font_size, colors['red'], font_thickness)
                       
    def format_metric_value(self, metric_name: str, value: Any) -> str:
        formatting = self.display_config['formatting']
        if metric_name == 'fps':
            return f"{value:.{formatting['decimal_places_fps']}f}"
        elif 'time' in metric_name.lower():
            return f"{value:.{formatting['decimal_places_time']}f}ms"
        else:
            return str(value)
        
    def draw_rectangle(self, frame, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                      color: Optional[List[int]] = None, thickness: Optional[int] = None, 
                      label: Optional[str] = None):
        rect_props = self.display_config['rectangle_properties']
        colors = self.display_config['colors']
        
        final_color: List[int]
        if color is None:
            color_name = rect_props['default_color']
            final_color = colors[color_name]
        else:
            final_color = color
        
        final_thickness: int
        if thickness is None:
            final_thickness = rect_props['default_thickness']
        else:
            final_thickness = thickness
            
        cv2.rectangle(frame, start_pos, end_pos, final_color, final_thickness)
        
        if label:
            font = getattr(cv2, self.display_config['font_face'])
            font_size = self.display_config['font_sizes']['small']
            font_thickness = self.display_config['font_thickness']
            
            rect_props = self.display_config['rectangle_properties']
            label_offset = rect_props['label_offset']
            label_margin = rect_props['label_margin']
            label_pos = (start_pos[0], max(label_offset + label_margin, start_pos[1] - label_offset))
            cv2.putText(frame, label, label_pos, font, font_size, final_color, font_thickness)
            
    def get_window_title(self) -> str:
        return self.display_config['window_title']
        
    def update_fps(self, fps):
        self.update_metric('fps', fps)
        
    def update_score(self, score):
        self.update_metric('score', score)
        
    def update_ocr_time(self, ocr_time):
        self.update_metric('ocr_time', ocr_time)
