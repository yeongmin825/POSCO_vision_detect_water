import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class BoundingBoxAdjuster:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        self.image = self.original_image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Bounding box 초기값 (이미지 중앙에 위치)
        self.x1 = int(self.width * 0.25)
        self.y1 = int(self.height * 0.25)
        self.x2 = int(self.width * 0.75)
        self.y2 = int(self.height * 0.75)
        
        # 마우스 상태
        self.dragging = False
        self.drag_corner = None
        
        # GUI 설정
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Bounding Box 조절 도구")
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 이미지 프레임
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 컨트롤 프레임
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 이미지 캔버스
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 마우스 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # 컨트롤 위젯들
        ttk.Label(control_frame, text="Bounding Box 조절", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # X1 슬라이더
        ttk.Label(control_frame, text="X1 (왼쪽)").pack()
        self.x1_var = tk.IntVar(value=self.x1)
        self.x1_slider = ttk.Scale(control_frame, from_=0, to=self.width, 
                                  variable=self.x1_var, orient=tk.HORIZONTAL,
                                  command=self.update_from_sliders)
        self.x1_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Y1 슬라이더
        ttk.Label(control_frame, text="Y1 (위쪽)").pack()
        self.y1_var = tk.IntVar(value=self.y1)
        self.y1_slider = ttk.Scale(control_frame, from_=0, to=self.height, 
                                  variable=self.y1_var, orient=tk.HORIZONTAL,
                                  command=self.update_from_sliders)
        self.y1_slider.pack(fill=tk.X, pady=(0, 10))
        
        # X2 슬라이더
        ttk.Label(control_frame, text="X2 (오른쪽)").pack()
        self.x2_var = tk.IntVar(value=self.x2)
        self.x2_slider = ttk.Scale(control_frame, from_=0, to=self.width, 
                                  variable=self.x2_var, orient=tk.HORIZONTAL,
                                  command=self.update_from_sliders)
        self.x2_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Y2 슬라이더
        ttk.Label(control_frame, text="Y2 (아래쪽)").pack()
        self.y2_var = tk.IntVar(value=self.y2)
        self.y2_slider = ttk.Scale(control_frame, from_=0, to=self.height, 
                                  variable=self.y2_var, orient=tk.HORIZONTAL,
                                  command=self.update_from_sliders)
        self.y2_slider.pack(fill=tk.X, pady=(0, 10))
        
        # 값 표시 라벨
        self.value_label = ttk.Label(control_frame, text="", font=("Arial", 10))
        self.value_label.pack(pady=10)
        
        # 버튼들
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="저장", command=self.save_coordinates).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="리셋", command=self.reset_coordinates).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="이미지 저장", command=self.save_image).pack(fill=tk.X, pady=2)
        
        # 이미지 표시
        self.display_image()
        
    def display_image(self):
        # 이미지 크기 조정 (캔버스에 맞게)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # 이미지 비율 유지하면서 크기 조정
            scale = min(canvas_width / self.width, canvas_height / self.height)
            new_width = int(self.width * scale)
            new_height = int(self.height * scale)
            
            resized_image = cv2.resize(self.image, (new_width, new_height))
            
            # OpenCV BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            
            # 캔버스에 이미지 표시
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Bounding box 그리기
            self.draw_bounding_box(scale)
            
            # 값 업데이트
            self.update_value_label()
    
    def draw_bounding_box(self, scale=1.0):
        # 스케일된 좌표 계산
        x1_scaled = int(self.x1 * scale)
        y1_scaled = int(self.y1 * scale)
        x2_scaled = int(self.x2 * scale)
        y2_scaled = int(self.y2 * scale)
        
        # 기존 박스 삭제
        self.canvas.delete("bbox")
        
        # 새 박스 그리기
        self.canvas.create_rectangle(x1_scaled, y1_scaled, x2_scaled, y2_scaled,
                                   outline="red", width=2, tags="bbox")
        
        # 코너에 작은 사각형 그리기
        corner_size = 8
        self.canvas.create_rectangle(x1_scaled-corner_size, y1_scaled-corner_size,
                                   x1_scaled+corner_size, y1_scaled+corner_size,
                                   fill="red", tags="bbox")
        self.canvas.create_rectangle(x2_scaled-corner_size, y1_scaled-corner_size,
                                   x2_scaled+corner_size, y1_scaled+corner_size,
                                   fill="red", tags="bbox")
        self.canvas.create_rectangle(x1_scaled-corner_size, y2_scaled-corner_size,
                                   x1_scaled+corner_size, y2_scaled+corner_size,
                                   fill="red", tags="bbox")
        self.canvas.create_rectangle(x2_scaled-corner_size, y2_scaled-corner_size,
                                   x2_scaled+corner_size, y2_scaled+corner_size,
                                   fill="red", tags="bbox")
    
    def update_from_sliders(self, event=None):
        self.x1 = self.x1_var.get()
        self.y1 = self.y1_var.get()
        self.x2 = self.x2_var.get()
        self.y2 = self.y2_var.get()
        
        # 좌표 정렬 (x1 < x2, y1 < y2)
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
            self.x1_var.set(self.x1)
            self.x2_var.set(self.x2)
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
            self.y1_var.set(self.y1)
            self.y2_var.set(self.y2)
        
        self.display_image()
    
    def update_value_label(self):
        self.value_label.config(text=f"좌표: ({self.x1}, {self.y1}) - ({self.x2}, {self.y2})\n"
                                    f"크기: {self.x2 - self.x1} x {self.y2 - self.y1}")
    
    def on_mouse_down(self, event):
        # 마우스 위치를 원본 이미지 좌표로 변환
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = min(canvas_width / self.width, canvas_height / self.height)
        
        mouse_x = int(event.x / scale)
        mouse_y = int(event.y / scale)
        
        # 코너 근처인지 확인
        corner_threshold = 20
        if abs(mouse_x - self.x1) < corner_threshold and abs(mouse_y - self.y1) < corner_threshold:
            self.drag_corner = "top_left"
            self.dragging = True
        elif abs(mouse_x - self.x2) < corner_threshold and abs(mouse_y - self.y1) < corner_threshold:
            self.drag_corner = "top_right"
            self.dragging = True
        elif abs(mouse_x - self.x1) < corner_threshold and abs(mouse_y - self.y2) < corner_threshold:
            self.drag_corner = "bottom_left"
            self.dragging = True
        elif abs(mouse_x - self.x2) < corner_threshold and abs(mouse_y - self.y2) < corner_threshold:
            self.drag_corner = "bottom_right"
            self.dragging = True
    
    def on_mouse_drag(self, event):
        if not self.dragging:
            return
        
        # 마우스 위치를 원본 이미지 좌표로 변환
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = min(canvas_width / self.width, canvas_height / self.height)
        
        mouse_x = int(event.x / scale)
        mouse_y = int(event.y / scale)
        
        # 좌표 범위 제한
        mouse_x = max(0, min(mouse_x, self.width))
        mouse_y = max(0, min(mouse_y, self.height))
        
        # 드래그하는 코너에 따라 좌표 업데이트
        if self.drag_corner == "top_left":
            self.x1 = mouse_x
            self.y1 = mouse_y
        elif self.drag_corner == "top_right":
            self.x2 = mouse_x
            self.y1 = mouse_y
        elif self.drag_corner == "bottom_left":
            self.x1 = mouse_x
            self.y2 = mouse_y
        elif self.drag_corner == "bottom_right":
            self.x2 = mouse_x
            self.y2 = mouse_y
        
        # 슬라이더 값 업데이트
        self.x1_var.set(self.x1)
        self.y1_var.set(self.y1)
        self.x2_var.set(self.x2)
        self.y2_var.set(self.y2)
        
        self.display_image()
    
    def on_mouse_up(self, event):
        self.dragging = False
        self.drag_corner = None
    
    def save_coordinates(self):
        coordinates = {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.x2 - self.x1,
            'height': self.y2 - self.y1
        }
        
        import json
        output_path = os.path.join(os.path.dirname(self.image_path), 'bounding_box_coordinates.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coordinates, f, indent=2, ensure_ascii=False)
        
        print(f"좌표가 저장되었습니다: {output_path}")
        print(f"좌표: {coordinates}")
    
    def reset_coordinates(self):
        # 이미지 중앙에 위치하도록 리셋
        self.x1 = int(self.width * 0.25)
        self.y1 = int(self.height * 0.25)
        self.x2 = int(self.width * 0.75)
        self.y2 = int(self.height * 0.75)
        
        # 슬라이더 값 업데이트
        self.x1_var.set(self.x1)
        self.y1_var.set(self.y1)
        self.x2_var.set(self.x2)
        self.y2_var.set(self.y2)
        
        self.display_image()
    
    def save_image(self):
        # Bounding box가 그려진 이미지 생성
        result_image = self.original_image.copy()
        cv2.rectangle(result_image, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
        
        # 파일명 생성
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(os.path.dirname(self.image_path), f"{base_name}_with_bbox.png")
        
        cv2.imwrite(output_path, result_image)
        print(f"이미지가 저장되었습니다: {output_path}")
    
    def run(self):
        # 창 크기 조정 이벤트 바인딩
        self.canvas.bind("<Configure>", lambda e: self.display_image())
        self.root.mainloop()

if __name__ == "__main__":
    image_path = "/home/kym/posco_vision/water_data_vertical/indy_5_ref.png"
    
    try:
        app = BoundingBoxAdjuster(image_path)
        app.run()
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
