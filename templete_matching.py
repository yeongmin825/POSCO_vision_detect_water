import cv2
import numpy as np
import json
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class TemplateMatchingAnalyzer:
    def __init__(self, ref_path, cur_path, bbox):
        """
        템플릿 매칭 분석기 초기화
        Args:
            ref_path: 참조 이미지 경로
            cur_path: 현재 이미지 경로  
            bbox: (x1, y1, x2, y2) 형태의 바운딩 박스
        """
        self.ref_path = ref_path
        self.cur_path = cur_path
        self.bbox = bbox
        
        # 이미지 로드
        self.ref_img = cv2.imread(ref_path)
        self.cur_img = cv2.imread(cur_path)
        
        if self.ref_img is None:
            raise FileNotFoundError(f"참조 이미지를 로드할 수 없습니다: {ref_path}")
        if self.cur_img is None:
            raise FileNotFoundError(f"현재 이미지를 로드할 수 없습니다: {cur_path}")
        
        # 바운딩 박스 영역 추출
        x1, y1, x2, y2 = bbox
        self.ref_roi = self.ref_img[y1:y2, x1:x2].copy()
        self.cur_roi = self.cur_img[y1:y2, x1:x2].copy()
        
        print(f"ROI 크기: {self.ref_roi.shape}")
        print(f"바운딩 박스: x({x1}~{x2}), y({y1}~{y2})")
    
    def compute_metrics(self, img1, img2):
        """이미지 간 유사도 지표 계산"""
        img1_32 = img1.astype(np.float32)
        img2_32 = img2.astype(np.float32)
        
        # MSE 계산
        mse = float(np.mean((img1_32 - img2_32) ** 2))
        
        # SSIM 계산
        try:
            ssim_val, ssim_map = ssim(img1, img2, channel_axis=2, full=True, data_range=255)
        except TypeError:
            # 구버전 skimage 호환
            ssim_val, ssim_map = ssim(img1, img2, multichannel=True, full=True, data_range=255)
        
        # PSNR 계산
        if mse == 0:
            psnr_val = float('inf')
        else:
            psnr_val = float(cv2.PSNR(img1, img2))
        
        return mse, ssim_val, psnr_val, ssim_map
    
    def create_difference_visualizations(self, img1, img2):
        """차이 시각화 생성"""
        # 절대 차이
        absdiff = cv2.absdiff(img1, img2)
        
        # 그레이스케일 차이
        diff_gray = cv2.cvtColor(absdiff, cv2.COLOR_BGR2GRAY)
        
        # 히트맵 생성
        norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        
        # Otsu 임계값으로 변화 마스크 생성
        _, change_mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return absdiff, heatmap, change_mask, diff_gray
    
    def template_matching_analysis(self, method=cv2.TM_CCOEFF_NORMED):
        """템플릿 매칭 기반 분석 수행"""
        # 참조 이미지를 템플릿으로, 현재 이미지에서 매칭 위치 찾기
        ref_gray = cv2.cvtColor(self.ref_roi, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(self.cur_roi, cv2.COLOR_BGR2GRAY)
        
        # 크기가 같으면 직접 비교
        if ref_gray.shape == cur_gray.shape:
            print("ROI 크기가 동일하여 직접 비교를 수행합니다.")
            match_score = 1.0
            best_match_roi = self.cur_roi.copy()
            match_location = (0, 0)
        else:
            # 템플릿 매칭 수행
            result = cv2.matchTemplate(cur_gray, ref_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_score = min_val
                match_location = min_loc
            else:
                match_score = max_val
                match_location = max_loc
            
            # 매칭된 영역 추출
            h, w = ref_gray.shape
            x, y = match_location
            best_match_roi = self.cur_roi[y:y+h, x:x+w]
        
        # 지표 계산
        mse, ssim_val, psnr_val, ssim_map = self.compute_metrics(self.ref_roi, best_match_roi)
        
        # 차이 시각화
        absdiff, heatmap, change_mask, diff_gray = self.create_difference_visualizations(
            self.ref_roi, best_match_roi
        )
        
        return {
            'match_score': float(match_score),
            'match_location': match_location,
            'mse': float(mse),
            'ssim': float(ssim_val),
            'psnr_db': float(psnr_val),
            'ssim_map': ssim_map,
            'absdiff': absdiff,
            'heatmap': heatmap,
            'change_mask': change_mask,
            'diff_gray': diff_gray,
            'best_match_roi': best_match_roi
        }
    
    def save_results(self, results, output_prefix="template_match_result"):
        """결과 저장"""
        # 이미지 저장
        cv2.imwrite(f"{output_prefix}_ref_roi.png", self.ref_roi)
        cv2.imwrite(f"{output_prefix}_cur_roi.png", self.cur_roi)
        cv2.imwrite(f"{output_prefix}_matched_roi.png", results['best_match_roi'])
        cv2.imwrite(f"{output_prefix}_absdiff.png", results['absdiff'])
        cv2.imwrite(f"{output_prefix}_heatmap.png", results['heatmap'])
        cv2.imwrite(f"{output_prefix}_change_mask.png", results['change_mask'])
        
        # SSIM 맵 저장
        ssim_vis = (results['ssim_map'] * 255).clip(0, 255).astype(np.uint8)
        ssim_colored = cv2.applyColorMap(ssim_vis, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(f"{output_prefix}_ssim_map.png", ssim_colored)
        
        # 전체 이미지에 바운딩 박스 표시
        ref_with_bbox = self.ref_img.copy()
        cur_with_bbox = self.cur_img.copy()
        x1, y1, x2, y2 = self.bbox
        
        cv2.rectangle(ref_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(cur_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(ref_with_bbox, "Reference ROI", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(cur_with_bbox, "Current ROI", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(f"{output_prefix}_ref_with_bbox.png", ref_with_bbox)
        cv2.imwrite(f"{output_prefix}_cur_with_bbox.png", cur_with_bbox)
        
        # 결과 비교 시각화
        self.create_comparison_plot(results, output_prefix)
        
        # JSON 리포트 저장
        report = {
            'reference_image': self.ref_path,
            'current_image': self.cur_path,
            'bounding_box': {
                'x1': self.bbox[0], 'y1': self.bbox[1], 
                'x2': self.bbox[2], 'y2': self.bbox[3]
            },
            'roi_size': {
                'width': self.bbox[2] - self.bbox[0],
                'height': self.bbox[3] - self.bbox[1]
            },
            'template_match_score': results['match_score'],
            'match_location': results['match_location'],
            'similarity_metrics': {
                'mse': results['mse'],
                'ssim': results['ssim'],
                'psnr_db': results['psnr_db']
            },
            'interpretation': self.interpret_results(results)
        }
        
        with open(f"{output_prefix}_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def interpret_results(self, results):
        """결과 해석"""
        ssim_val = results['ssim']
        psnr_val = results['psnr_db']
        mse_val = results['mse']
        
        if ssim_val > 0.95:
            similarity_level = "매우 유사"
        elif ssim_val > 0.8:
            similarity_level = "유사"
        elif ssim_val > 0.6:
            similarity_level = "보통"
        elif ssim_val > 0.4:
            similarity_level = "다름"
        else:
            similarity_level = "매우 다름"
        
        return {
            'similarity_level': similarity_level,
            'ssim_score': f"{ssim_val:.4f} (1에 가까울수록 유사)",
            'psnr_score': f"{psnr_val:.2f} dB (클수록 유사)",
            'mse_score': f"{mse_val:.2f} (작을수록 유사)"
        }
    
    def create_comparison_plot(self, results, output_prefix):
        """비교 시각화 플롯 생성"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 참조 이미지
        axes[0, 0].imshow(cv2.cvtColor(self.ref_roi, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Reference ROI')
        axes[0, 0].axis('off')
        
        # 현재 이미지
        axes[0, 1].imshow(cv2.cvtColor(results['best_match_roi'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Current ROI')
        axes[0, 1].axis('off')
        
        # 절대 차이
        axes[0, 2].imshow(cv2.cvtColor(results['absdiff'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Absolute Difference')
        axes[0, 2].axis('off')
        
        # 히트맵
        axes[1, 0].imshow(cv2.cvtColor(results['heatmap'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Difference Heatmap')
        axes[1, 0].axis('off')
        
        # 변화 마스크
        axes[1, 1].imshow(results['change_mask'], cmap='gray')
        axes[1, 1].set_title('Change Mask')
        axes[1, 1].axis('off')
        
        # SSIM 맵
        axes[1, 2].imshow(results['ssim_map'], cmap='viridis')
        axes[1, 2].set_title('SSIM Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    # 파일 경로 설정
    ref_path = "/home/kym/posco_vision/water_data_vertical/ref.png"
    # cur_path = "/home/kym/posco_vision/water_data_vertical/cur_yes_water.png"
    cur_path = "/home/kym/posco_vision/water_data_vertical/cur_no_water.png"
    
    # 바운딩 박스 설정 (x: 561~768, y: 299~476)
    bbox = (561, 299, 768, 476)
    
    try:
        # 분석기 초기화
        analyzer = TemplateMatchingAnalyzer(ref_path, cur_path, bbox)
        
        # 템플릿 매칭 분석 수행
        print("템플릿 매칭 분석을 시작합니다...")
        results = analyzer.template_matching_analysis()
        
        # 결과 저장
        output_prefix = "/home/kym/posco_vision/water_data_vertical/template_match_result"
        report = analyzer.save_results(results, output_prefix)
        
        # 결과 출력
        print("\n=== 템플릿 매칭 분석 결과 ===")
        print(f"매칭 점수: {results['match_score']:.4f}")
        print(f"매칭 위치: {results['match_location']}")
        print(f"MSE: {results['mse']:.4f}")
        print(f"SSIM: {results['ssim']:.4f}")
        print(f"PSNR: {results['psnr_db']:.2f} dB")
        print(f"\n해석: {report['interpretation']['similarity_level']}")
        print(f"저장된 파일들: {output_prefix}_*.png, {output_prefix}_report.json")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 