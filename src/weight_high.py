# weight_high.py — 身高改採用 high.py 的估算方式；體重改為「深度輪廓逐列積分」法
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# 🔁 身高：沿用 high.py 的 3D 取點（中位數濾波與深度右移降噪邏輯由 high.py 處理）
from high import get_3d_point as high_get_3d_point

# 相機內參（請依你的 Kinect 做微調，須與 high.py 一致）
FX = 458.5
FY = 458.2
CX = 343.6
CY = 229.8

# 深度資料單位與處理設定
DEPTH_UNIT_MM = 1.0       # 你的深度是以毫米為單位
APPLY_SHIFT = True        # 是否套用 >>3 與 high.py 對齊
SHIFT_BITS = 3
INVALID_U16 = 65528       # 若你的原始深度傳此值代表無效，視實況調整
RHO_KG_PER_M3 = 985.0     # 人體等效密度（含衣物/含水），可視校正調整
ROW_FILL_COEF = 0.82      # 橫截面填充係數（矩形→橢圓/人體型，π/4≈0.785 ~ 0.85 之間取經驗值）

# 逐列積分的穩健門檻
MIN_MASK_PIXELS = 1200        # 人體分割最少像素
MIN_ROW_PIXELS = 15           # 一列上最少人體像素
MIN_VALID_ROWS = 80           # 最少有效列數
THICKNESS_MIN_MM = 60         # 單列厚度下限（mm）
THICKNESS_MAX_MM = 700        # 單列厚度上限（mm）
DEPTH_MIN_MM = 500            # 合理深度下限（mm）
DEPTH_MAX_MM = 4500           # 合理深度上限（mm）


class IntegratedBodyMeasurement:
    """整合身高體重測量類（身高＝high.py；體重＝深度輪廓逐列積分法）"""

    def __init__(self):
        # MediaPipe 初始化
        self.mp_pose = mp.solutions.pose
        self.mp_selfie = mp.solutions.selfie_segmentation

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.segmentation = self.mp_selfie.SelfieSegmentation(model_selection=1)

        # 歷史記錄與平滑
        self.height_history = deque(maxlen=12)   # 身高平滑（與 high.py 相容的遞增權重）
        self.weight_history = deque(maxlen=24)   # 體重短期平滑（顯示穩定）

        # 校正參數
        self.calibration_factor = 1.0
        self.is_calibrated = False
        self.calibration_samples = deque()
        self.target_weight = 66.0
        self.calibration_file = "weight_calibration.txt"

        self._load_calibration()

        print("身高體重測量系統初始化完成（身高=high.py；體重=逐列積分）")
        print(f"目標體重設定為: {self.target_weight:.1f} kg")
        if self.is_calibrated:
            print(f"已載入校正係數: {self.calibration_factor:.3f}")

    # ---------- 校正檔案存取 ----------
    def _load_calibration(self):
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                data = f.read().strip().split('\n')
                self.calibration_factor = float(data[0])
                if len(data) > 1:
                    self.target_weight = float(data[1])
                self.is_calibrated = True
                print(f"載入校正係數: {self.calibration_factor:.3f}")
        except Exception:
            print("未找到校正文件，使用默認設置")

    def _save_calibration(self):
        try:
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                f.write(f"{self.calibration_factor}\n{self.target_weight}\n")
            print(f"校正數據已保存 (係數: {self.calibration_factor:.3f})")
        except Exception as e:
            print(f"保存校正數據失敗: {e}")

    def add_calibration_sample(self, raw_weight):
        if raw_weight and raw_weight > 0:
            self.calibration_samples.append(float(raw_weight))
            print(f"校正樣本 #{len(self.calibration_samples)}: {raw_weight:.1f} kg")
            if len(self.calibration_samples) >= 10:
                self.calculate_calibration_factor()
                return True
        return False

    def calculate_calibration_factor(self):
        if len(self.calibration_samples) < 5:
            print("校正樣本不足，需要至少5個樣本")
            return False

        samples = np.array(self.calibration_samples, dtype=float)
        q1, q3 = np.percentile(samples, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filt = samples[(samples >= lb) & (samples <= ub)]
        if len(filt) < 3:
            print("有效校正樣本不足")
            return False

        avg_measured = float(np.mean(filt))
        self.calibration_factor = self.target_weight / max(avg_measured, 1e-6)
        self.is_calibrated = True

        print("校正完成!")
        print(f"平均測量值: {avg_measured:.1f} kg")
        print(f"目標體重: {self.target_weight:.1f} kg")
        print(f"校正係數: {self.calibration_factor:.3f}")
        print(f"使用樣本: {len(filt)}/{len(samples)}")

        self._save_calibration()
        self.calibration_samples.clear()
        return True

    def reset_calibration(self):
        self.calibration_factor = 1.0
        self.is_calibrated = False
        self.calibration_samples.clear()
        try:
            import os
            if os.path.exists(self.calibration_file):
                os.remove(self.calibration_file)
        except Exception:
            pass
        print("校正已重置")

    # ---------- 影像處理 ----------
    def segment_person(self, rgb_image):
        """人体分割（MediaPipe Selfie Segmentation）"""
        try:
            # MediaPipe 期望輸入是 RGB（本程式已保證）
            results = self.segmentation.process(rgb_image)
            if results.segmentation_mask is None:
                return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

            mask = (results.segmentation_mask > 0.5).astype(np.uint8)

            # 形態學清理：閉運算→開運算，移除孔洞與小斑點
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            return (mask * 255).astype(np.uint8)
        except Exception as e:
            print(f"分割錯誤: {e}")
            return np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    def _prepare_depth(self, depth_frame):
        """與 high.py 一致的深度預處理：位移降噪 + 無效值處理"""
        d = depth_frame.astype(np.uint16)

        # 無效值歸零
        if INVALID_U16 is not None:
            d = np.where(d == INVALID_U16, 0, d)

        if APPLY_SHIFT:
            d = (d >> SHIFT_BITS).astype(np.uint16)

        # 輕度中值濾波，抑制鹽椒雜訊
        d = cv2.medianBlur(d, 5)
        return d

    def _rowwise_volume_mm3(self, depth_u16, mask_u8):
        """
        逐列（row）積分估體積：
        - 對每一列，取前景像素列的 u 範圍與深度分位數（近端/遠端）估厚度
        - 將像素寬度、像素高度換算為該列的實距（mm），得到該列的近似截面積
        - 體積 = Σ( 截面積 * 切片高度 )
        """
        h, w = depth_u16.shape
        mask_bin = (mask_u8 > 0)
        total_mm3 = 0.0
        valid_rows = 0

        # 預先建立 u,v 座標格
        u_coords = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        v_coords = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)

        for v in range(h):
            row_mask = mask_bin[v, :]
            if row_mask.sum() < MIN_ROW_PIXELS:
                continue

            # 取該列有效深度
            z = depth_u16[v, row_mask].astype(np.float32)
            z = z[(z >= DEPTH_MIN_MM) & (z <= DEPTH_MAX_MM)]
            if z.size < MIN_ROW_PIXELS:
                continue

            # 前/後表面用分位數抑制異常值
            z_near = np.percentile(z, 10)   # 近端
            z_far  = np.percentile(z, 90)   # 遠端
            thickness = float(z_far - z_near)
            if thickness < THICKNESS_MIN_MM or thickness > THICKNESS_MAX_MM:
                continue

            # 該列人體的 u 範圍
            u_idx = np.where(row_mask)[0]
            u_min, u_max = int(u_idx.min()), int(u_idx.max())
            width_px = max(u_max - u_min + 1, 1)

            # 以該列的平均深度換算像素對應的實長
            z_mean = float(np.mean(z))
            # 每個像素在該深度的實際長度（mm/px）
            dx_mm = z_mean / FX
            dy_mm = z_mean / FY

            width_mm = width_px * dx_mm
            slice_h_mm = dy_mm

            # 該列橫截面積近似（矩形→橢圓/人體填充校正）
            area_mm2 = width_mm * thickness * ROW_FILL_COEF

            total_mm3 += area_mm2 * slice_h_mm
            valid_rows += 1

        if valid_rows < MIN_VALID_ROWS:
            return 0.0, valid_rows
        return total_mm3, valid_rows

    def estimate_weight(self, rgb_frame, depth_frame, height_cm):
        """
        體重估算（新方法）：
        1) 人體分割 → 2) 深度預處理 → 3) 逐列積分估體積 → 4) 以ρ=985 kg/m³換算體重
        5) 依校正係數修正；並回傳「未校正原值」供校正模式使用
        """
        try:
            # 1) 人體分割
            mask = self.segment_person(rgb_frame)
            if mask is None or (mask > 0).sum() < MIN_MASK_PIXELS:
                return None, "人體分割不足", None

            # 2) 深度預處理
            d = self._prepare_depth(depth_frame)

            # 3) 逐列積分
            vol_mm3, used_rows = self._rowwise_volume_mm3(d, mask)
            if vol_mm3 <= 0:
                return None, "體積計算失敗", None

            vol_m3 = vol_mm3 / 1e9

            # 4) 物理質量估算（人體密度）
            weight_phys = vol_m3 * RHO_KG_PER_M3

            # 5) 對身高做極輕微的尺度正規化（可關閉）
            if height_cm and height_cm > 0:
                h_norm = (height_cm / 170.0) ** 0.15
                weight_raw = float(weight_phys * h_norm)
            else:
                weight_raw = float(weight_phys)

            # 邊界限制
            weight_raw = float(np.clip(weight_raw, 30.0, 200.0))

            # 校正
            if self.is_calibrated:
                weight_cal = weight_raw * self.calibration_factor
                return weight_cal, f"預測成功（已校正；有效列={used_rows}）", weight_raw
            else:
                return weight_raw, f"預測成功（未校正；有效列={used_rows}）", weight_raw

        except Exception as e:
            return None, f"體重估算錯誤: {str(e)}", None

    # ---------- 統計工具 ----------
    def compute_confidence_interval(self, data, confidence=0.95):
        """簡易信賴區間（無 scipy 時採用常態近似）"""
        arr = np.asarray(list(data), dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        mean = float(np.mean(arr))
        if arr.size < 2:
            return mean, 0.0
        std = float(np.std(arr, ddof=1))
        sem = std / np.sqrt(arr.size)
        # 常態近似 1.96
        h = 1.96 * sem if confidence >= 0.95 else 1.64 * sem
        return mean, h

    # ---------- 主流程（單幀處理） ----------
    def process_frame(self, rgb_frame, depth_frame):
        """
        處理單幀數據
        ✅ 身高：用 high.py 的 get_3d_point() 與遞增權重平滑
        ✅ 體重：用新方法（逐列積分）
        """
        h, w, _ = rgb_frame.shape

        # MediaPipe 姿勢（輸入需 RGB）
        results = self.pose.process(rgb_frame)

        height_cm = None
        weight_kg = None
        raw_weight = None
        weight_msg = ""

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[self.mp_pose.PoseLandmark.NOSE]
            left_heel = lm[self.mp_pose.PoseLandmark.LEFT_HEEL]
            right_heel = lm[self.mp_pose.PoseLandmark.RIGHT_HEEL]

            nx, ny = int(nose.x * w), int(nose.y * h)
            hx = int((left_heel.x + right_heel.x) / 2 * w)
            hy = int((left_heel.y + right_heel.y) / 2 * h)

            # 高度 3D 取點（交給 high.py）
            head3d = high_get_3d_point(nx, ny, depth_frame)
            heel3d = high_get_3d_point(hx, hy, depth_frame)

            if head3d and heel3d:
                coord_head, _ = head3d
                coord_heel, _ = heel3d
                height_m = abs(coord_heel[1] - coord_head[1])
                cur_height_cm = height_m * 100.0

                # 遞增權重平滑（較新的權重較大）
                self.height_history.append(cur_height_cm)
                weights = np.linspace(1.0, 2.0, num=len(self.height_history))
                height_cm = float(np.average(self.height_history, weights=weights))

                # 體重估算（新方法）
                w_est, w_msg, w_raw = self.estimate_weight(rgb_frame, depth_frame, height_cm)
                weight_msg = w_msg
                raw_weight = w_raw
                if w_est:
                    # 簡單滑動平均讓顯示更穩
                    self.weight_history.append(float(w_est))
                    weight_kg = float(np.mean(self.weight_history))

        # 回傳原始 RGB（顯示端自己疊字）；以及數值與狀態字串
        return rgb_frame, height_cm, weight_kg, raw_weight, weight_msg

    def close(self):
        try:
            self.pose.close()
            self.segmentation.close()
        except Exception:
            pass


# ------------------------- 可獨立測試的 main -------------------------
def main():
    """獨立測試用主函數（保留原先顯示/校正/按鍵行為）"""
    try:
        from kinect_data_receiver import KinectDataReceiver
    except ImportError:
        print("錯誤: 找不到 kinect_data_receiver.py")
        print("此 main 僅供本地端獨立測試用")
        return

    receiver = KinectDataReceiver()
    measurement = IntegratedBodyMeasurement()

    cv2.namedWindow("身高體重測量系統", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("身高體重測量系統", 480, 640)

    last_stat_time = time.time()
    calibration_mode = False

    try:
        print("身高體重測量系統啟動")
        print("=" * 50)
        print("操作說明:")
        print("  C鍵 - 開始校正模式（收集10個樣本）")
        print("  R鍵 - 重置校正")
        print("  ESC鍵 - 退出程序")
        print("=" * 50)

        if measurement.is_calibrated:
            print(f"已載入校正: 係數={measurement.calibration_factor:.3f}")
        else:
            print("建議先進行校正以提高準確性（按C鍵開始）")

        while True:
            rgb = receiver.get_rgb_frame()
            depth = receiver.get_depth_frame()

            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            # 單幀處理
            img_rgb, height_cm, weight_kg, raw_weight, weight_msg = measurement.process_frame(rgb, depth)

            # 顯示：轉 BGR 疊字
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if height_cm:
                cv2.putText(img_display, f"Height: {height_cm:.1f} cm", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if weight_kg:
                cv2.putText(img_display, f"Weight: {weight_kg:.1f} kg", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if height_cm and weight_kg and height_cm > 0:
                bmi = weight_kg / ((height_cm / 100.0) ** 2)
                cv2.putText(img_display, f"BMI: {bmi:.1f}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # 校正模式提示
            if calibration_mode:
                cv2.putText(img_display, f"Calibrating... {len(measurement.calibration_samples)}/10",
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                if raw_weight:
                    if measurement.add_calibration_sample(raw_weight):
                        calibration_mode = False
                        print("校正完成！")

            # 每 3 秒輸出統計
            now = time.time()
            if now - last_stat_time >= 3.0:
                last_stat_time = now
                if height_cm:
                    # 以最近一段歷史近似統計
                    h_vals = list(self_val for self_val in measurement.height_history)
                    if len(h_vals) >= 2:
                        h_mean, h_ci = measurement.compute_confidence_interval(h_vals)
                        print(f"身高統計: {h_mean:.1f} ± {h_ci:.1f} cm (樣本:{len(h_vals)})")
                if weight_kg:
                    w_vals = list(self_val for self_val in measurement.weight_history)
                    if len(w_vals) >= 2:
                        w_mean, w_ci = measurement.compute_confidence_interval(w_vals)
                        status = "已校正" if measurement.is_calibrated else "未校正"
                        print(f"體重統計: {w_mean:.1f} ± {w_ci:.1f} kg ({status}, 樣本:{len(w_vals)})")
                        if height_cm:
                            avg_bmi = w_mean / ((height_cm / 100.0) ** 2)
                            print(f"平均BMI: {avg_bmi:.1f}")
                # 額外印出本幀狀態
                if weight_msg:
                    print(f"[體重] {weight_msg}")

            cv2.imshow("身高體重測量系統", img_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key in (ord('c'), ord('C')):
                calibration_mode = True
                measurement.calibration_samples.clear()
                print("開始校正模式 - 將收集10個體重樣本...")
            elif key in (ord('r'), ord('R')):
                measurement.reset_calibration()
                calibration_mode = False
                print("校正已重置")

    except KeyboardInterrupt:
        print("\n收到中斷信號，正在退出...")
    finally:
        measurement.close()
        receiver.close()
        cv2.destroyAllWindows()
        print("程序已安全退出")


if __name__ == "__main__":
    main()
