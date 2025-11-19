# liveness_detector.py
"""
活體偵測系統 - 基於 Kinect 深度資訊
- 使用深度資訊判斷是否為真人（而非照片/螢幕）
- 結合人臉偵測與深度變化分析
"""

import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("警告: insightface 未安裝")

try:
    from kinect_data_receiver import KinectDataReceiver
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False
    print("注意: kinect_data_receiver.py 未找到")


class LivenessDetector:
    """活體偵測系統 - 基於深度資訊"""
    
    def __init__(self, use_gpu=True):
        """
        初始化活體偵測系統
        
        Args:
            use_gpu: 是否使用 GPU
        """
        print("=" * 70)
        print("初始化活體偵測系統 (深度分析)...")
        print("=" * 70)
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("請安裝 insightface: pip install insightface onnxruntime")
        
        # 初始化 InsightFace 用於人臉偵測
        print("\n載入人臉偵測模型...")
        try:
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("嘗試使用 GPU 加速...")
            else:
                providers = ['CPUExecutionProvider']
                print("使用 CPU 模式")
            
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            
            det_size = (640, 640) if use_gpu else (480, 480)
            self.app.prepare(ctx_id=0, det_size=det_size)
            
            actual_provider = self.app.models['detection'].session.get_providers()[0]
            if 'CUDA' in actual_provider:
                print("✓ 人臉偵測模型載入完成 (使用 GPU - CUDA)")
            else:
                print("✓ 人臉偵測模型載入完成 (使用 CPU)")
        except Exception as e:
            print(f"✗ 模型載入失敗: {e}")
            raise
        
        # 活體偵測參數
        self.depth_history = deque(maxlen=30)  # 深度歷史記錄（約1秒）
        self.face_depth_history = deque(maxlen=30)  # 人臉區域平均深度
        
        # 閾值設定
        self.MIN_DEPTH = 400        # 最小深度（mm）- 太近視為無效
        self.MAX_DEPTH = 4000       # 最大深度（mm）- 太遠視為無效
        self.VALID_DEPTH_RANGE = (600, 2500)  # 有效深度範圍（mm）
        
        # 深度變化閾值（用於判斷是否為平面）
        self.DEPTH_VARIANCE_THRESHOLD = 15.0   # 深度變異數閾值（mm²）
        self.DEPTH_RANGE_THRESHOLD = 40.0      # 深度範圍閾值（mm）
        
        # 活體判定參數
        self.min_samples_for_decision = 20     # 至少需要20幀才能判定
        self.liveness_score_threshold = 0.65   # 活體分數閾值
        
        # 統計資料
        self.frame_count = 0
        self.detection_count = 0
        self.is_live = None
        self.liveness_score = 0.0
        self.last_result = None
        
        print("\n✓ 活體偵測系統初始化完成")
        print(f"  深度範圍: {self.VALID_DEPTH_RANGE[0]}-{self.VALID_DEPTH_RANGE[1]} mm")
        print(f"  判定閾值: {self.liveness_score_threshold}")
        print("=" * 70 + "\n")
    
    def extract_face_depth(self, depth_frame, bbox):
        """
        提取人臉區域的深度資訊
        
        Args:
            depth_frame: 深度影像 (H, W)
            bbox: 人臉邊界框 [x1, y1, x2, y2]
            
        Returns:
            face_depth_stats: 人臉深度統計資訊字典
        """
        if depth_frame is None or bbox is None:
            return None
        
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = depth_frame.shape
            
            # 確保座標在範圍內
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # 提取人臉區域深度
            face_region = depth_frame[y1:y2, x1:x2].copy()
            
            # 處理深度資料（右移3位元移除 Player Index）
            face_region = (face_region >> 3).astype(np.uint16)
            
            # 過濾無效深度值
            valid_depths = face_region[(face_region > self.MIN_DEPTH) & (face_region < self.MAX_DEPTH)]
            
            if len(valid_depths) < 10:  # 至少需要10個有效點
                return None
            
            # 計算統計資訊
            mean_depth = float(np.mean(valid_depths))
            median_depth = float(np.median(valid_depths))
            std_depth = float(np.std(valid_depths))
            min_depth = float(np.min(valid_depths))
            max_depth = float(np.max(valid_depths))
            depth_range = max_depth - min_depth
            
            # 計算深度變異係數（CV）
            cv = (std_depth / mean_depth) if mean_depth > 0 else 0.0
            
            return {
                'mean': mean_depth,
                'median': median_depth,
                'std': std_depth,
                'min': min_depth,
                'max': max_depth,
                'range': depth_range,
                'cv': cv,
                'valid_points': len(valid_depths),
                'total_points': face_region.size
            }
        except Exception as e:
            print(f"提取人臉深度錯誤: {e}")
            return None
    
    def analyze_liveness(self):
        """
        分析活體特徵
        
        Returns:
            is_live: 是否為活體（True/False/None）
            score: 活體分數（0.0-1.0）
            reason: 判定原因
        """
        if len(self.face_depth_history) < self.min_samples_for_decision:
            return None, 0.0, "樣本不足"
        
        try:
            depths = np.array([d['median'] for d in self.face_depth_history if d is not None])
            
            if len(depths) < self.min_samples_for_decision:
                return None, 0.0, "有效樣本不足"
            
            # 特徵1: 深度變化（真人會有微小的頭部移動）
            depth_variance = float(np.var(depths))
            depth_std = float(np.std(depths))
            depth_range = float(np.max(depths) - np.min(depths))
            
            # 特徵2: 深度分佈（真人臉部有立體感）
            avg_face_variance = np.mean([d['std'] for d in self.face_depth_history if d is not None])
            avg_face_range = np.mean([d['range'] for d in self.face_depth_history if d is not None])
            avg_cv = np.mean([d['cv'] for d in self.face_depth_history if d is not None])
            
            # 計算活體分數（0.0-1.0）
            score = 0.0
            reasons = []
            
            # 評分項目1: 深度變化（權重 30%）
            # 真人會有自然的微小移動，但照片/螢幕則非常穩定
            if 2.0 < depth_std < 50.0:  # 適度的深度變化
                score += 0.3
                reasons.append(f"深度變化正常({depth_std:.1f}mm)")
            elif depth_std <= 2.0:
                reasons.append(f"深度過於穩定({depth_std:.1f}mm)-疑似照片")
            else:
                reasons.append(f"深度變化過大({depth_std:.1f}mm)")
            
            # 評分項目2: 臉部深度範圍（權重 40%）
            # 真人臉部有立體感，照片/螢幕則較平坦
            if avg_face_range > self.DEPTH_RANGE_THRESHOLD:
                range_score = min(1.0, avg_face_range / 100.0)  # 範圍越大分數越高
                score += 0.4 * range_score
                reasons.append(f"臉部立體感良好({avg_face_range:.1f}mm)")
            else:
                reasons.append(f"臉部過於平坦({avg_face_range:.1f}mm)-疑似照片")
            
            # 評分項目3: 深度變異數（權重 30%）
            # 真人臉部各點深度有變化，照片則較一致
            if avg_face_variance > self.DEPTH_VARIANCE_THRESHOLD:
                variance_score = min(1.0, avg_face_variance / 50.0)
                score += 0.3 * variance_score
                reasons.append(f"深度變異正常({avg_face_variance:.1f})")
            else:
                reasons.append(f"深度變異過小({avg_face_variance:.1f})-疑似照片")
            
            # 最終判定
            is_live = score >= self.liveness_score_threshold
            reason = " | ".join(reasons)
            
            return is_live, score, reason
            
        except Exception as e:
            print(f"活體分析錯誤: {e}")
            return None, 0.0, f"分析錯誤: {str(e)}"
    
    def detect(self, rgb_frame: np.ndarray, depth_frame: np.ndarray):
        """
        執行活體偵測
        
        Args:
            rgb_frame: RGB 影像 (H, W, 3)
            depth_frame: 深度影像 (H, W)
            
        Returns:
            result: 偵測結果字典
        """
        if rgb_frame is None or depth_frame is None:
            return None
        
        self.frame_count += 1
        
        try:
            # 確保格式正確
            if len(rgb_frame.shape) != 3 or rgb_frame.shape[2] != 3:
                return None
            
            # 轉換為 BGR (InsightFace 需要)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 偵測人臉
            faces = self.app.get(bgr_frame)
            
            if len(faces) == 0:
                return {
                    'has_face': False,
                    'is_live': None,
                    'score': 0.0,
                    'reason': '未偵測到人臉',
                    'samples_collected': len(self.face_depth_history),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 選擇最大的臉
            if len(faces) > 1:
                faces = sorted(
                    faces,
                    key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                    reverse=True
                )
            
            face = faces[0]
            bbox = [int(x) for x in face.bbox]
            self.detection_count += 1
            
            # 提取人臉深度資訊
            face_depth_stats = self.extract_face_depth(depth_frame, bbox)
            
            if face_depth_stats is None:
                return {
                    'has_face': True,
                    'bbox': bbox,
                    'is_live': None,
                    'score': 0.0,
                    'reason': '無法提取深度資訊',
                    'samples_collected': len(self.face_depth_history),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 檢查深度範圍
            mean_depth = face_depth_stats['mean']
            if not (self.VALID_DEPTH_RANGE[0] <= mean_depth <= self.VALID_DEPTH_RANGE[1]):
                reason = f"深度超出範圍({mean_depth:.0f}mm)"
                return {
                    'has_face': True,
                    'bbox': bbox,
                    'is_live': False,
                    'score': 0.0,
                    'reason': reason,
                    'depth_stats': face_depth_stats,
                    'samples_collected': len(self.face_depth_history),
                    'timestamp': datetime.now().isoformat()
                }
            
            # 記錄深度歷史
            self.face_depth_history.append(face_depth_stats)
            
            # 分析活體特徵
            is_live, score, reason = self.analyze_liveness()
            
            self.is_live = is_live
            self.liveness_score = score
            
            result = {
                'has_face': True,
                'bbox': bbox,
                'is_live': is_live,
                'score': score,
                'reason': reason,
                'depth_stats': face_depth_stats,
                'samples_collected': len(self.face_depth_history),
                'timestamp': datetime.now().isoformat()
            }
            
            self.last_result = result
            return result
            
        except Exception as e:
            print(f"活體偵測錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reset(self):
        """重置歷史資料"""
        self.depth_history.clear()
        self.face_depth_history.clear()
        self.is_live = None
        self.liveness_score = 0.0
        self.last_result = None
        print("活體偵測器已重置")
    
    def get_stats(self):
        """取得統計資訊"""
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'detection_rate': detection_rate,
            'samples_collected': len(self.face_depth_history),
            'is_ready': len(self.face_depth_history) >= self.min_samples_for_decision,
            'is_live': self.is_live,
            'liveness_score': self.liveness_score
        }
    
    def close(self):
        """關閉資源"""
        print("活體偵測系統已關閉")


# ==================== 測試用主程式 ====================

def main():
    """獨立測試用"""
    try:
        from kinect_data_receiver import KinectDataReceiver
    except ImportError:
        print("錯誤: 找不到 kinect_data_receiver.py")
        return
    
    receiver = KinectDataReceiver()
    detector = LivenessDetector(use_gpu=True)
    
    print("\n活體偵測測試啟動")
    print("=" * 70)
    print("站在 Kinect 前方 0.6-2.5 公尺處")
    print("系統將自動判斷是否為真人")
    print("按 'q' 退出, 'r' 重置")
    print("=" * 70)
    
    cv2.namedWindow("Liveness Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Liveness Detection", 480, 640)
    
    try:
        while True:
            rgb = receiver.get_rgb_frame()
            depth = receiver.get_depth_frame()
            
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue
            
            # 偵測
            result = detector.detect(rgb, depth)
            
            # 顯示
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            if result and result.get('has_face'):
                bbox = result.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = bbox
                    
                    # 繪製人臉框
                    if result.get('is_live') is True:
                        color = (0, 255, 0)  # 綠色 - 真人
                        status = "LIVE - REAL PERSON"
                    elif result.get('is_live') is False:
                        color = (0, 0, 255)  # 紅色 - 假冒
                        status = "FAKE - PHOTO/SCREEN"
                    else:
                        color = (255, 255, 0)  # 黃色 - 分析中
                        status = "ANALYZING..."
                    
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                    
                    # 顯示狀態
                    cv2.putText(display, status, (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # 顯示分數
                    score = result.get('score', 0.0)
                    score_text = f"Score: {score:.2f}"
                    cv2.putText(display, score_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 顯示進度
                    samples = result.get('samples_collected', 0)
                    needed = detector.min_samples_for_decision
                    progress_text = f"Samples: {samples}/{needed}"
                    cv2.putText(display, progress_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 顯示深度資訊
                    if 'depth_stats' in result and result['depth_stats']:
                        depth_text = f"Depth: {result['depth_stats']['mean']:.0f}mm"
                        cv2.putText(display, depth_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 顯示原因（如果有的話）
                    reason = result.get('reason', '')
                    if reason and len(reason) > 0:
                        # 分行顯示長文字
                        y_offset = 90
                        max_width = 50
                        words = reason.split()
                        line = ""
                        for word in words:
                            if len(line + word) < max_width:
                                line += word + " "
                            else:
                                cv2.putText(display, line.strip(), (10, y_offset),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                y_offset += 20
                                line = word + " "
                        if line:
                            cv2.putText(display, line.strip(), (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # 沒有偵測到人臉
                cv2.putText(display, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if result and result.get('reason'):
                    cv2.putText(display, result.get('reason', ''), (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Liveness Detection", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset()
                print("✓ 已重置")
    
    except KeyboardInterrupt:
        print("\n中斷...")
    finally:
        detector.close()
        receiver.close()
        cv2.destroyAllWindows()
        print("✓ 已安全退出")


if __name__ == "__main__":
    main()