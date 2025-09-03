import cv2
import numpy as np
from neuromeka import IndyEye
from PIL import Image
from neuromeka import IndyDCP3
import time
import os
from skimage.metrics import structural_similarity as ssim
import json

class WaterDetectionRealtime:
    def __init__(self, ref_image_path, bbox, ssim_threshold=0.6):
        """
        실시간 물 검출기 초기화
        Args:
            ref_image_path: 참조 이미지 경로
            bbox: (x1, y1, x2, y2) 형태의 바운딩 박스
            ssim_threshold: SSIM 임계값 (이 값보다 높으면 유사, 낮으면 다름)
        """
        self.ref_image_path = ref_image_path
        self.bbox = bbox
        self.ssim_threshold = ssim_threshold
        # 소정렬 검색 패딩 (px)
        self.search_pad = 15
        
        # 참조 이미지 로드 및 ROI 추출
        self.ref_img = cv2.imread(ref_image_path)
        if self.ref_img is None:
            raise FileNotFoundError(f"참조 이미지를 로드할 수 없습니다: {ref_image_path}")
        
        x1, y1, x2, y2 = bbox
        self.ref_roi = self.ref_img[y1:y2, x1:x2].copy()
        
        print(f"참조 이미지 로드 완료: {ref_image_path}")
        print(f"ROI 크기: {self.ref_roi.shape}")
        print(f"바운딩 박스: x({x1}~{x2}), y({y1}~{y2})")
        print(f"SSIM 임계값: {ssim_threshold}")
    
    def compute_ssim(self, img1, img2):
        """두 이미지 간 SSIM 계산 (그레이스케일)"""
        # BGR을 그레이스케일로 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        try:
            ssim_val, _ = ssim(gray1, gray2, full=True, data_range=255)
        except TypeError:
            # 구버전 skimage 호환
            ssim_val, _ = ssim(gray1, gray2, full=True, data_range=255)
        return float(ssim_val)
    
    def align_roi_by_template(self, current_frame):
        """현재 프레임에서 ROI 주변 검색창을 만들어 템플릿 매칭으로 정렬된 ROI를 반환"""
        x1, y1, x2, y2 = self.bbox
        h = y2 - y1
        w = x2 - x1
        pad = self.search_pad
        H, W = current_frame.shape[:2]
        
        # 검색 창 좌표 (패딩 적용)
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(W, x2 + pad)
        sy2 = min(H, y2 + pad)
        
        # 그레이스케일 템플릿과 검색 창
        gray_ref = cv2.cvtColor(self.ref_roi, cv2.COLOR_BGR2GRAY)
        search_bgr = current_frame[sy1:sy2, sx1:sx2]
        gray_search = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
        
        # 검색 창이 템플릿보다 작으면 패딩 줄이기
        if gray_search.shape[0] < h or gray_search.shape[1] < w:
            return current_frame[y1:y2, x1:x2].copy(), (x1, y1)
        
        res = cv2.matchTemplate(gray_search, gray_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        mx, my = max_loc
        abs_x = sx1 + mx
        abs_y = sy1 + my
        
        # 정렬된 ROI 추출 (경계 체크)
        abs_x2 = min(abs_x + w, W)
        abs_y2 = min(abs_y + h, H)
        aligned_roi = current_frame[abs_y:abs_y2, abs_x:abs_x2].copy()
        
        # 만약 경계로 인해 크기가 달라졌다면 원래 ROI로 대체
        if aligned_roi.shape[:2] != (h, w):
            return current_frame[y1:y2, x1:x2].copy(), (x1, y1)
        return aligned_roi, (abs_x, abs_y)
    
    def analyze_current_frame(self, current_frame):
        """현재 프레임 분석"""
        # 템플릿 매칭으로 정렬된 ROI 사용
        aligned_roi, match_loc = self.align_roi_by_template(current_frame)
        
        # SSIM 계산
        ssim_score = self.compute_ssim(self.ref_roi, aligned_roi)
        
        # 유사도 판단
        is_similar = ssim_score >= self.ssim_threshold
        status = "No Water" if is_similar else "Yes Water"
        
        return {
            'ssim_score': ssim_score,
            'is_similar': is_similar,
            'status': status,
            'current_roi': aligned_roi,
            'match_location': match_loc
        }
    
    def draw_info_on_frame(self, frame, analysis_result):
        """프레임에 정보 표시"""
        x1, y1, x2, y2 = self.bbox
        
        # 바운딩 박스 그리기
        color = (0, 255, 0) if analysis_result['is_similar'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 상태 텍스트 표시
        ssim_score = analysis_result['ssim_score']
        status = analysis_result['status']
        
        # 배경 박스
        text_bg_y = y1 - 80
        if text_bg_y < 0:
            text_bg_y = y2 + 10
        
        cv2.rectangle(frame, (x1, text_bg_y), (x1 + 300, text_bg_y + 70), (0, 0, 0), -1)
        
        # 텍스트 표시
        cv2.putText(frame, f"SSIM: {ssim_score:.4f}", (x1 + 5, text_bg_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (x1 + 5, text_bg_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Threshold: {self.ssim_threshold}", (x1 + 5, text_bg_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def is_water(eye: IndyEye, ref_image_path, bbox, duration_sec=300, ssim_threshold=0.6):
    """
    실시간 물 검출 수행
    Args:
        eye: IndyEye 카메라 객체
        ref_image_path: 참조 이미지 경로
        bbox: 바운딩 박스 (x1, y1, x2, y2)
        duration_sec: 실행 시간 (초)
        ssim_threshold: SSIM 임계값
    """
    
    # 물 검출기 초기화
    detector = WaterDetectionRealtime(ref_image_path, bbox, ssim_threshold)
    
    # 저장 디렉토리 설정
    save_dir = "/home/kym/posco_vision/water_data_vertical"
    os.makedirs(save_dir, exist_ok=True)
    
    save_counter = 1
    start_time = time.time()
    
    print("\n=== 실시간 물 검출 시작 ===")
    # print("'s' 키: 현재 프레임과 분석 결과 저장")
    # print("'r' 키: 참조 이미지 다시 로드")
    # print("'+' 키: SSIM 임계값 증가 (+0.05)")
    # print("'-' 키: SSIM 임계값 감소 (-0.05)")
    # print("'q' 키: 종료")
    # print("=" * 50)
    
    while True:
        
        if time.time() - start_time > duration_sec:
            print(f"{duration_sec}초가 경과하여 자동 종료됩니다.")
            break
        
        # IndyEye에서 이미지 받기
        pil_image = eye.image()
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 현재 프레임 분석
        analysis_result = detector.analyze_current_frame(frame_bgr)

        # break
        
        # 프레임에 정보 표시
        display_frame = detector.draw_info_on_frame(frame_bgr.copy(), analysis_result)
        
        # 화면에 표시
        cv2.imshow('Water Detection - IndyEye Camera', display_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # 현재 프레임과 분석 결과 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 원본 프레임 저장
            frame_filename = f"{save_dir}/capture_{timestamp}_{save_counter:04d}.png"
            cv2.imwrite(frame_filename, frame_bgr)
            
            # 분석 결과가 표시된 프레임 저장
            result_filename = f"{save_dir}/analysis_{timestamp}_{save_counter:04d}.png"
            cv2.imwrite(result_filename, display_frame)
            
            # ROI 이미지들 저장
            roi_ref_filename = f"{save_dir}/roi_ref_{timestamp}_{save_counter:04d}.png"
            roi_cur_filename = f"{save_dir}/roi_cur_{timestamp}_{save_counter:04d}.png"
            cv2.imwrite(roi_ref_filename, detector.ref_roi)
            cv2.imwrite(roi_cur_filename, analysis_result['current_roi'])
            
            # 분석 결과를 JSON으로 저장
            report = {
                'timestamp': timestamp,
                'counter': save_counter,
                'ssim_score': analysis_result['ssim_score'],
                'ssim_threshold': detector.ssim_threshold,
                'is_similar': analysis_result['is_similar'],
                'status': analysis_result['status'],
                'bbox': bbox,
                'files': {
                    'original_frame': frame_filename,
                    'analysis_frame': result_filename,
                    'roi_reference': roi_ref_filename,
                    'roi_current': roi_cur_filename
                }
            }
            
            json_filename = f"{save_dir}/report_{timestamp}_{save_counter:04d}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"저장 완료 #{save_counter}: SSIM={analysis_result['ssim_score']:.4f}, 상태={analysis_result['status']}")
            save_counter += 1
        
        elif key == ord('r'):
            # 참조 이미지 다시 로드
            try:
                detector = WaterDetectionRealtime(ref_image_path, bbox, detector.ssim_threshold)
                print("참조 이미지를 다시 로드했습니다.")
            except Exception as e:
                print(f"참조 이미지 로드 실패: {e}")
        
        elif key == ord('+') or key == ord('='):
            # SSIM 임계값 증가
            detector.ssim_threshold = min(1.0, detector.ssim_threshold + 0.05)
            print(f"SSIM 임계값 증가: {detector.ssim_threshold:.2f}")
        
        elif key == ord('-'):
            # SSIM 임계값 감소
            detector.ssim_threshold = max(0.0, detector.ssim_threshold - 0.05)
            print(f"SSIM 임계값 감소: {detector.ssim_threshold:.2f}")
        
        elif key == ord('q'):
            print("사용자가 종료를 요청했습니다.")
            break
    
    cv2.destroyAllWindows()



    # print(f"총 {save_counter-1}개의 분석 결과가 저장되었습니다.")
    is_water = not analysis_result['is_similar']
    ssim_score = analysis_result['ssim_score']
    print(f"Analysis Result: {is_water}, SSIM: {ssim_score}")

    
        
    return is_water

def main():

    robot = 6
    if robot == 5:
        STEP_IP = "192.168.0.5"
        EYE_IP = "192.168.0.4"
    elif robot == 6:
        STEP_IP = "192.168.0.6"
        EYE_IP = "192.168.0.2"
    # 설정
    # 참조 이미지 경로
    if robot == 5:
        ref_image_path = "/home/kym/posco_vision/water_data_vertical/indy_5_ref.png"
    elif robot == 6:
        ref_image_path = "/home/kym/posco_vision/water_data_vertical/indy_6_ref.png"
    
    # 바운딩 박스 (x: 561~768, y: 299~476)

    if robot == 5:
        bbox = (595, 331, 728, 443)
    elif robot == 6:
        bbox = (595, 331, 728, 443)
    
    # SSIM 임계값 (0.6 이상이면 유사, 미만이면 다름)

    if robot == 5:
        ssim_threshold = 0.4
    elif robot == 6:
        ssim_threshold = 0.4
    
    # 로봇 및 카메라 연결
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
    
    # 실시간 물 검출 시작
    is_water(
        eye=eye,
        ref_image_path=ref_image_path,
        bbox=bbox,
        duration_sec=3000,  # 50분
        ssim_threshold=ssim_threshold
    )

if __name__ == "__main__":
    main() 