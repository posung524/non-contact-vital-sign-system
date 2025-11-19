# server_1.py
"""
MedicalCheckIn Backend Server
提供 Kinect 影像串流和 WebSocket API
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import cv2
import numpy as np
import json
import base64
from datetime import datetime
from typing import List
import uvicorn
import time
from pain_detection_system import RightHandOnlyDetection 

try:
    from kinect_data_receiver import KinectDataReceiver
    KINECT_AVAILABLE = True
except ImportError:
    print("警告: kinect_data_receiver.py 未找到,將使用模擬模式")
    KINECT_AVAILABLE = False
    
try:
    from weight_high import IntegratedBodyMeasurement
    WEIGHT_MODULE_AVAILABLE = True
except ImportError:
    print("警告: weight_high.py 未找到,將使用模擬資料")
    WEIGHT_MODULE_AVAILABLE = False

try:
    from age_gender_detector import AgeGenderDetector
    AGE_GENDER_AVAILABLE = True
except ImportError:
    print("警告: age_gender_detector.py 未找到")
    AGE_GENDER_AVAILABLE = False

kinect_receiver = None
body_measurement = None 
pain_system = None
age_gender_detector = None
active_module = "idle"

class MockKinectReceiver:
    """模擬 Kinect 接收器"""
    def __init__(self):
        self.frame_count = 0
    
    def get_rgb_frame(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.putText(img, "Mock Kinect Feed", (50, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Frame: {self.frame_count}", (50, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.frame_count += 1
        return img
    
    def get_depth_frame(self):
        return np.random.randint(500, 3000, (640, 480), dtype=np.uint16)
    
    def get_stats(self):
        return {
            'rgb_fps': 30,
            'depth_fps': 30,
            'color_depth_fps': 0,
            'rgb_buffer_size': 0,
            'depth_buffer_size': 0
        }
    
    def is_data_available(self):
        return True
    
    def close(self):
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kinect_receiver
    
    if KINECT_AVAILABLE:
        try:
            kinect_receiver = KinectDataReceiver()
            print("✓ Kinect 接收器已初始化")
        except Exception as e:
            print(f"⚠ Kinect 初始化失敗,使用模擬模式: {e}")
            kinect_receiver = MockKinectReceiver()
    else:
        kinect_receiver = MockKinectReceiver()
        print("✓ 使用模擬 Kinect 模式")
    
    yield
    
    global body_measurement, pain_system, age_gender_detector
    if kinect_receiver:
        kinect_receiver.close()
        print("✓ Kinect 接收器已關閉")
    if body_measurement:
        body_measurement.close()
        print("✓ 身高體重模組已關閉")
    if pain_system:
        try:
            pain_system.save_pain_records()
        except Exception:
            pass
        pain_system = None
        print("✓ 疼痛辨識模組已釋放")
    if age_gender_detector:
        age_gender_detector.close()
        print("✓ 年齡性別辨識模組已關閉")

app = FastAPI(title="MedicalCheckIn Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/module/activate/{name}")
async def activate_module(name: str):
    global body_measurement, pain_system, age_gender_detector, active_module, kinect_receiver
    name = name.lower().strip()

    def close_detection_modules():
        """只關閉檢測模組，保留 Kinect receiver"""
        global body_measurement, pain_system, age_gender_detector
        if body_measurement:
            try:
                body_measurement.close()
            except Exception:
                pass
            body_measurement = None
            print("✓ 身高體重模組已關閉")
        if pain_system:
            try:
                pain_system.save_pain_records()
            except Exception:
                pass
            pain_system.running = False
            pain_system = None
            print("✓ 疼痛辨識模組已釋放")
        if age_gender_detector:
            try:
                age_gender_detector.close()
            except Exception:
                pass
            age_gender_detector = None
            print("✓ 年齡性別辨識模組已關閉")

    if name == "age_gender":
        close_detection_modules()
        if AGE_GENDER_AVAILABLE:
            try:
                age_gender_detector = AgeGenderDetector(
                    use_gpu=True,
                    use_kinect=False,
                    min_samples_for_stable=5
                )
                active_module = "age_gender"
                print("✓ 年齡性別辨識模組已初始化 (GPU)")
                return {"ok": True, "active": active_module}
            except Exception as e:
                print(f"⚠ 年齡性別辨識模組初始化失敗: {e}")
                age_gender_detector = None
                active_module = "idle"
                return {"ok": False, "error": str(e)}
        else:
            return {"ok": False, "error": "age_gender module not available"}

    if name == "body":
        close_detection_modules()
        if WEIGHT_MODULE_AVAILABLE:
            try:
                body_measurement = IntegratedBodyMeasurement()
                print("✓ 身高體重模組已初始化")
            except Exception as e:
                print(f"⚠ 身高體重模組初始化失敗: {e}")
                body_measurement = None
        active_module = "body"
        return {"ok": True, "active": active_module}

    if name == "pain":
        close_detection_modules()
        try:
            ps = RightHandOnlyDetection()
            ps.kinect_receiver = kinect_receiver
            ps.pain_records = []
            ps.current_touch_start = None
            ps.current_body_part = None
            pain_system = ps
            print("✓ 疼痛辨識模組已初始化(右手)")
        except Exception as e:
            print(f"⚠ 疼痛辨識模組初始化失敗: {e}")
            pain_system = None
        active_module = "pain"
        return {"ok": True, "active": active_module}

    if name == "idle":
        close_detection_modules()
        active_module = "idle"
        return {"ok": True, "active": active_module}

    return {"ok": False, "error": f"unknown module '{name}'"}

@app.get("/")
async def root():
    return {
        "service": "MedicalCheckIn Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    stats = kinect_receiver.get_stats() if kinect_receiver else {}
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "kinect": {
            "available": kinect_receiver.is_data_available() if kinect_receiver else False,
            "rgb_fps": stats.get('rgb_fps', 0),
            "depth_fps": stats.get('depth_fps', 0),
            "mode": "real" if KINECT_AVAILABLE else "mock"
        },
        "modules": {
            "video_stream": "ready",
            "body_measurement": "ready" if WEIGHT_MODULE_AVAILABLE else "unavailable",
            "pain_detection": "ready",
            "age_gender": "ready" if AGE_GENDER_AVAILABLE else "unavailable"
        }
    }

def encode_frame_to_base64(frame):
    if frame is None:
        return None
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

@app.websocket("/stream/video")
async def websocket_video_stream(websocket: WebSocket):
    await websocket.accept()
    print("✓ 影像串流客戶端已連接")
    
    try:
        while True:
            if kinect_receiver and kinect_receiver.is_data_available():
                rgb_frame = kinect_receiver.get_rgb_frame()
                
                if rgb_frame is not None:
                    frame_base64 = encode_frame_to_base64(rgb_frame)
                    
                    await websocket.send_json({
                        "type": "video_frame",
                        "data": {
                            "frame": frame_base64,
                            "width": rgb_frame.shape[1],
                            "height": rgb_frame.shape[0],
                            "timestamp": datetime.now().isoformat()
                        }
                    })
            
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        print("✗ 影像串流客戶端已斷線")
    except Exception as e:
        if "WebSocket" not in str(e):
            print(f"✗ 影像串流錯誤: {e}")

@app.websocket("/stream/age_gender")
async def stream_age_gender(websocket: WebSocket):
    await websocket.accept()
    print("✓ 年齡性別辨識串流客戶端已連接")
    
    try:
        while True:
            if active_module == "age_gender" and age_gender_detector and kinect_receiver and kinect_receiver.is_data_available():
                rgb_frame = kinect_receiver.get_rgb_frame()
                
                if rgb_frame is not None:
                    result = age_gender_detector.detect(rgb_frame)
                    
                    if result:
                        await websocket.send_json({
                            "type": "age_gender_result",
                            "data": {
                                "age": result['age'],
                                "gender": result['gender_zh'],
                                "gender_en": result['gender_str'],
                                "confidence": result['confidence'],
                                "timestamp": result['timestamp']
                            }
                        })
                    
                    stable_result = age_gender_detector.get_stable_result_dict()
                    if stable_result:
                        await websocket.send_json({
                            "type": "age_gender_stable",
                            "data": {
                                "age": stable_result['age'],
                                "gender": stable_result['gender_zh'],
                                "gender_en": stable_result['gender_en'],
                                "confidence": stable_result['confidence'],
                                "is_stable": stable_result['is_stable'],
                                "sample_count": stable_result['sample_count']
                            }
                        })
            else:
                await websocket.send_json({
                    "type": "age_gender_result",
                    "data": None
                })
            
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print("✗ 年齡性別辨識串流客戶端已斷線")
    except Exception as e:
        if "WebSocket" not in str(e):
            print(f"✗ 年齡性別辨識串流錯誤: {e}")

@app.websocket("/stream/body")
async def websocket_body_metrics(websocket: WebSocket):
    await websocket.accept()
    print("✓ 身高體重串流客戶端已連接")
    
    try:
        while True:
            if active_module == "body" and body_measurement and kinect_receiver and kinect_receiver.is_data_available():
                rgb_frame = kinect_receiver.get_rgb_frame()
                depth_frame = kinect_receiver.get_depth_frame()
                
                if rgb_frame is not None and depth_frame is not None:
                    try:
                        result = body_measurement.process_frame(rgb_frame, depth_frame)
                        
                        if len(result) >= 5:
                            img_display, height_cm, weight_kg, raw_weight, weight_msg = result
                            
                            if height_cm and weight_kg:
                                bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm > 0 else None
                                
                                await websocket.send_json({
                                    "type": "body_metrics",
                                    "data": {
                                        "height_cm": round(height_cm, 1),
                                        "weight_kg": round(weight_kg, 1),
                                        "bmi": round(bmi, 1) if bmi else None,
                                        "status": weight_msg
                                    },
                                    "timestamp": datetime.now().isoformat()
                                })
                    except Exception as e:
                        print(f"body_measurement 處理錯誤: {e}")
            else:
                mock_data = {
                    "height_cm": 175.0 + np.random.uniform(-2, 2),
                    "weight_kg": 66.7 + np.random.uniform(-3, 3),
                    "bmi": 22.9,
                    "status": "模擬資料"
                }
                
                await websocket.send_json({
                    "type": "body_metrics",
                    "data": mock_data,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        print("✗ 身高體重串流客戶端已斷線")
    except Exception as e:
        if "WebSocket" not in str(e):
            print(f"✗ 身高體重串流錯誤: {e}")

@app.websocket("/stream/pain")
async def stream_pain(websocket: WebSocket):
    await websocket.accept()
    print("✓ 疼痛偵測串流客戶端已連接")
    
    try:
        last_body_part = None
        last_sent_len = 0
        last_list_sync_ts = 0.0
        
        while True:
            if active_module == "pain" and pain_system and kinect_receiver and kinect_receiver.is_data_available():
                rgb = kinect_receiver.get_rgb_frame()
                depth = kinect_receiver.get_depth_frame()
                if rgb is None or depth is None:
                    await asyncio.sleep(0.01)
                    continue

                _ = pain_system.process_frame(rgb, depth)
                
                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if len(rgb.shape) == 3 else rgb
                ok, buf = cv2.imencode(".jpg", frame_bgr)
                if not ok:
                    await asyncio.sleep(0.01)
                    continue
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                curr = getattr(pain_system, "current_body_part", None)
                if curr != last_body_part:
                    last_body_part = curr
                    if curr:
                        zh = pain_system.body_parts_chinese.get(curr, curr)
                        en = pain_system.body_parts_english.get(curr, curr)
                        print(f"[疼痛] 右手 → {zh} ({en})")
                    else:
                        print("[疼痛] 右手 → None")
                
                now = time.time()
                start = getattr(pain_system, "current_touch_start", None)
                thr = getattr(pain_system, "hold_time_threshold", 2.5)
                progress = 0
                if start and curr:
                    progress = int(max(0, min(100, (now - start) / max(thr, 1e-3) * 100)))

                await websocket.send_json({
                    "type": "pain_frame",
                    "data": {"frame": b64, "timestamp": datetime.now().isoformat()}
                })
                
                await websocket.send_json({
                    "type": "pain_progress",
                    "data": {
                        "progress": progress,
                        "current": curr,
                        "current_zh": pain_system.body_parts_chinese.get(curr, "") if curr else ""
                    }
                })
                
                total = len(pain_system.pain_records)
                if total > last_sent_len:
                    for idx, rec in enumerate(pain_system.pain_records[last_sent_len:], start=last_sent_len):
                        en = ""
                        zh = ""
                        
                        if isinstance(rec, dict):
                            en = str(rec.get("english_name") or rec.get("part_en") or rec.get("part") or rec.get("english") or "").strip()
                            zh = str(rec.get("chinese_name") or rec.get("part_zh") or rec.get("chinese") or "").strip()
                            if not zh and en:
                                zh = pain_system.body_parts_chinese.get(en, "")
                        elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            en = str(rec[0] or "").strip()
                            zh = str(rec[1] or "").strip()
                            if not zh and en:
                                zh = pain_system.body_parts_chinese.get(en, en)
                        elif isinstance(rec, str):
                            en = rec.strip()
                            zh = pain_system.body_parts_chinese.get(en, en)
                        
                        if zh:
                            await websocket.send_json({
                                "type": "pain_confirmed",
                                "data": {"english_name": en, "chinese_name": zh}
                            })
                    
                    last_sent_len = total

                    try:
                        zh_list = []
                        for rec in pain_system.pain_records:
                            en = ""
                            zh = ""
                            if isinstance(rec, dict):
                                en = str(rec.get("english_name") or rec.get("part_en") or rec.get("part") or rec.get("english") or "").strip()
                                zh = str(rec.get("chinese_name") or rec.get("part_zh") or rec.get("chinese") or "").strip()
                                if not zh and en:
                                    zh = pain_system.body_parts_chinese.get(en, "")
                            elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                en = str(rec[0] or "").strip()
                                zh = str(rec[1] or "").strip()
                                if not zh and en:
                                    zh = pain_system.body_parts_chinese.get(en, en)
                            elif isinstance(rec, str):
                                en = rec.strip()
                                zh = pain_system.body_parts_chinese.get(en, en)
                            
                            if zh and zh.strip():
                                zh_list.append(zh.strip())
                        
                        if zh_list:
                            await websocket.send_json({"type": "pain_list", "data": {"list": zh_list}})
                    except Exception as e:
                        print(f"[ERROR] pain_list 同步失敗: {e}")

                try:
                    if pain_system and len(pain_system.pain_records) > 0:
                        now_ts = time.time()
                        if now_ts - last_list_sync_ts >= 1.0:
                            zh_list = []
                            for rec in pain_system.pain_records:
                                en = ""
                                zh = ""
                                if isinstance(rec, dict):
                                    en = str(rec.get("english_name") or rec.get("part_en") or rec.get("part") or rec.get("english") or "").strip()
                                    zh = str(rec.get("chinese_name") or rec.get("part_zh") or rec.get("chinese") or "").strip()
                                    if not zh and en:
                                        zh = pain_system.body_parts_chinese.get(en, "")
                                elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
                                    en = str(rec[0] or "").strip()
                                    zh = str(rec[1] or "").strip()
                                    if not zh and en:
                                        zh = pain_system.body_parts_chinese.get(en, en)
                                elif isinstance(rec, str):
                                    en = rec.strip()
                                    zh = pain_system.body_parts_chinese.get(en, en)
                                
                                if zh and zh.strip():
                                    zh_list.append(zh.strip())
                            
                            if zh_list:
                                await websocket.send_json({"type": "pain_list", "data": {"list": zh_list}})
                            last_list_sync_ts = now_ts
                except Exception as e:
                    print(f"[ERROR] pain_list 定期補發失敗: {e}")

                await asyncio.sleep(0.03)
            else:
                await websocket.send_json({
                    "type": "pain_progress",
                    "data": {"progress": 0, "current": None, "current_zh": ""}
                })
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        print("✗ 疼痛偵測串流客戶端已斷線")
    except Exception as e:
        if "WebSocket" not in str(e):
            print(f"✗ 疼痛偵測串流錯誤: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("MedicalCheckIn Backend Server")
    print("=" * 60)
    print("啟動服務中...")
    print("API 文件: http://localhost:8000/docs")
    print("健康檢查: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )