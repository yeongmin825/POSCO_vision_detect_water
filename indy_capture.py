import cv2
import numpy as np
from neuromeka import IndyEye
from PIL import Image
from neuromeka import IndyDCP3
import time
import math
import os

def measure_average_angle(eye: IndyEye, duration_sec=5, area_min=15, area_max=170, radius=140):
    """
    IndyEye 카메라에서 이미지를 받아 threshold와 contour area를 UI로 조정하면서 각도 측정 후 평균 각도 리턴.
    실시간 화면에 검출 각도와 평균 각도를 표시.
    """

    water_data_dir = "/home/kym/posco_vision/water_data_vertical"
    os.makedirs(water_data_dir, exist_ok=True)

    save_counter = 1
    start_time = time.time()

    print("실시간 이미지 캡처를 시작합니다.")
    print("'s' 키를 누르면 이미지가 저장됩니다.")
    print("'q' 키를 누르면 종료됩니다.")

    while True:
        if time.time() - start_time > duration_sec:
            break

        # IndyEye에서 이미지 받기
        pil_image = eye.image()
        frame_rgb = np.array(pil_image)
        
        # BGR로 변환 (OpenCV는 BGR 사용)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 화면에 표시
        cv2.imshow('IndyEye Camera', frame_bgr)
        
        # 키 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF
        
        # 's' 키를 누르면 이미지 저장
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{water_data_dir}/water_{timestamp}_{save_counter:04d}.png"
            cv2.imwrite(filename, frame_bgr)
            print(f"이미지 저장됨: {filename}")
            save_counter += 1
        
        # 'q' 키를 누르면 종료
        elif key == ord('q'):
            print("사용자가 종료를 요청했습니다.")
            break

    cv2.destroyAllWindows()
    print(f"총 {save_counter-1}개의 이미지가 저장되었습니다.")

# --- 설정 ---
# 로봇 및 카메라 IP
# 파란 로봇
STEP_IP = "192.168.0.6"
EYE_IP = "192.168.0.2"

#빨간 로봇
# STEP_IP = "192.168.0.5"
# EYE_IP = "192.168.0.4"

# ⭐ [변경] 잘라낼 프레임의 크기를 여기에 정의합니다.
CROP_WIDTH = 800
CROP_HEIGHT = 600


# --- 로봇 및 카메라 연결 ---
try:
    indy = IndyDCP3(STEP_IP)
    print("✅ Indy 로봇에 성공적으로 연결되었습니다.")
except Exception as e:
    print(f"❌ 로봇 연결에 실패했습니다: {e}")
    exit()

try:
    eye = IndyEye(eye_ip=EYE_IP)
    print("✅ IndyEye 카메라에 성공적으로 연결되었습니다.")
except Exception as e:
    print(f"❌ 카메라 연결에 실패했습니다: {e}")
    
    exit()

# offset = -130.
# task_pos = indy.get_control_state()['p']
# task_pos[2] += offset

# indy.movel(task_pos, vel_ratio=10, acc_ratio=100)
# indy.wait_for_motion_state('is_target_reached')





measure_average_angle(eye, duration_sec=5000, area_min=15, area_max=170, radius=140)

