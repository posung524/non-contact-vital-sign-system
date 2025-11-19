import socket
import struct
import numpy as np
import cv2
import threading
import time
from collections import defaultdict

class KinectDataReceiver:
    """Kinect數據接收器 - 專門負責接收和處理C#端傳來的數據"""
    
    def __init__(self, rgb_port=12345, depth_port=12346, color_depth_port=12347):
        self.rgb_port = rgb_port
        self.depth_port = depth_port
        self.color_depth_port = color_depth_port
        
        # 接收已旋轉的數據尺寸 480x640 (C#端已旋轉)
        self.rgb_width = 480
        self.rgb_height = 640
        self.depth_width = 480
        self.depth_height = 640
        
        # 數據緩衝區
        self.rgb_buffer = defaultdict(dict)
        self.depth_buffer = defaultdict(dict)
        self.color_depth_buffer = defaultdict(dict)
        
        # 最新幀數據 (已旋轉)
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_color_depth = None
        
        # 線程控制
        self.running = True
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.color_depth_lock = threading.Lock()
        
        # 統計信息
        self.rgb_fps = 0
        self.depth_fps = 0
        self.color_depth_fps = 0
        self.last_rgb_time = time.time()
        self.last_depth_time = time.time()
        self.last_color_depth_time = time.time()
        self.rgb_frame_count = 0
        self.depth_frame_count = 0
        self.color_depth_frame_count = 0
        
        print("初始化Kinect數據接收器...")
        self.setup_sockets()
        self.start_receiving()

    def setup_sockets(self):
        """設置UDP socket"""
        try:
            # RGB socket
            self.rgb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
            self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rgb_socket.bind(('127.0.0.1', self.rgb_port))
            self.rgb_socket.settimeout(0.5)
            
            # Depth socket
            self.depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
            self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.depth_socket.bind(('127.0.0.1', self.depth_port))
            self.depth_socket.settimeout(0.5)
            
            # Color Depth socket
            self.color_depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.color_depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16*1024*1024)
            self.color_depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.color_depth_socket.bind(('127.0.0.1', self.color_depth_port))
            self.color_depth_socket.settimeout(0.5)
            
            print(f"Socket設置完成 - RGB:{self.rgb_port}, 原始深度:{self.depth_port}, C#彩色深度:{self.color_depth_port}")
            
        except Exception as e:
            print(f"Socket設置失敗: {e}")
            raise

    def start_receiving(self):
        """啟動接收線程"""
        self.rgb_thread = threading.Thread(target=self.receive_rgb_loop, daemon=True)
        self.depth_thread = threading.Thread(target=self.receive_depth_loop, daemon=True)
        self.color_depth_thread = threading.Thread(target=self.receive_color_depth_loop, daemon=True)
        self.cleanup_thread = threading.Thread(target=self.cleanup_old_frames, daemon=True)
        
        self.rgb_thread.start()
        self.depth_thread.start()
        self.color_depth_thread.start()
        self.cleanup_thread.start()
        
        print("數據接收線程已啟動")

    def receive_rgb_loop(self):
        """RGB數據接收循環"""
        while self.running:
            try:
                data, addr = self.rgb_socket.recvfrom(65535)
                self.process_rgb_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"RGB接收錯誤: {e}")
                    time.sleep(0.01)

    def receive_depth_loop(self):
        """深度數據接收循環"""
        while self.running:
            try:
                data, addr = self.depth_socket.recvfrom(65535)
                self.process_depth_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"深度接收錯誤: {e}")
                    time.sleep(0.01)

    def receive_color_depth_loop(self):
        """彩色深度圖數據接收循環"""
        while self.running:
            try:
                data, addr = self.color_depth_socket.recvfrom(65535)
                self.process_color_depth_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"C#彩色深度接收錯誤: {e}")
                    time.sleep(0.01)

    def process_rgb_packet(self, data):
        """處理RGB數據包"""
        try:
            data_type = data[0]
            frame_id = struct.unpack('<I', data[1:5])[0]
            total_packets = struct.unpack('<H', data[5:7])[0]
            packet_index = struct.unpack('<H', data[7:9])[0]
            packet_data = data[9:]
            
            if data_type != 0x01:
                return
                
            with self.rgb_lock:
                self.rgb_buffer[frame_id][packet_index] = packet_data
                
                if len(self.rgb_buffer[frame_id]) == total_packets:
                    self.assemble_rgb_frame(frame_id, total_packets)
                    
        except Exception as e:
            print(f"RGB包處理錯誤: {e}")

    def process_depth_packet(self, data):
        """處理深度數據包"""
        try:
            data_type = data[0]
            frame_id = struct.unpack('<I', data[1:5])[0]
            total_packets = struct.unpack('<H', data[5:7])[0]
            packet_index = struct.unpack('<H', data[7:9])[0]
            packet_data = data[9:]
            
            if data_type != 0x02:
                return
                
            with self.depth_lock:
                self.depth_buffer[frame_id][packet_index] = packet_data
                
                if len(self.depth_buffer[frame_id]) == total_packets:
                    self.assemble_depth_frame(frame_id, total_packets)
                    
        except Exception as e:
            print(f"深度包處理錯誤: {e}")

    def process_color_depth_packet(self, data):
        """處理彩色深度圖數據包"""
        try:
            data_type = data[0]
            frame_id = struct.unpack('<I', data[1:5])[0]
            total_packets = struct.unpack('<H', data[5:7])[0]
            packet_index = struct.unpack('<H', data[7:9])[0]
            packet_data = data[9:]
            
            if data_type != 0x03:
                return
                
            with self.color_depth_lock:
                self.color_depth_buffer[frame_id][packet_index] = packet_data
                
                if len(self.color_depth_buffer[frame_id]) == total_packets:
                    self.assemble_color_depth_frame(frame_id, total_packets)
                    
        except Exception as e:
            print(f"C#彩色深度包處理錯誤: {e}")

    def assemble_rgb_frame(self, frame_id, total_packets):
        """組裝RGB幀 (接收已旋轉的480x640數據)"""
        try:
            # 組裝數據
            frame_data = b''
            for i in range(total_packets):
                if i in self.rgb_buffer[frame_id]:
                    frame_data += self.rgb_buffer[frame_id][i]
                else:
                    print(f"RGB幀{frame_id}缺少包{i}/{total_packets}")
                    return

            # 解碼PNG圖像 (C#端已旋轉為480x640)
            img_array = np.frombuffer(frame_data, dtype=np.uint8)
            rgb_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if rgb_image is not None:
                # 確保是C#端傳來的旋轉後尺寸 480x640
                if rgb_image.shape != (640, 480, 3):
                    rgb_image = cv2.resize(rgb_image, (480, 640), interpolation=cv2.INTER_CUBIC)
                
                # 轉換BGR到RGB
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
                # 直接使用C#端已旋轉的數據
                self.latest_rgb = rgb_image
                
                # 更新FPS統計
                self.rgb_frame_count += 1
                current_time = time.time()
                if current_time - self.last_rgb_time >= 1.0:
                    self.rgb_fps = self.rgb_frame_count
                    self.rgb_frame_count = 0
                    self.last_rgb_time = current_time
            
            del self.rgb_buffer[frame_id]
            
        except Exception as e:
            print(f"RGB幀組裝錯誤: {e}")
            if frame_id in self.rgb_buffer:
                del self.rgb_buffer[frame_id]

    def assemble_depth_frame(self, frame_id, total_packets):
        """組裝深度幀 (接收已旋轉的480x640數據)"""
        try:
            # 組裝數據
            frame_data = b''
            for i in range(total_packets):
                if i in self.depth_buffer[frame_id]:
                    frame_data += self.depth_buffer[frame_id][i]
                else:
                    print(f"深度幀{frame_id}缺少包{i}/{total_packets}")
                    return
            
            # 轉換為16位深度數據
            depth_array = np.frombuffer(frame_data, dtype=np.uint16)
            
            # C#端已旋轉，所以像素數量是 480x640
            expected_pixels = 480 * 640
            
            if len(depth_array) == expected_pixels:
                # 重塑為C#端旋轉後的尺寸 (640, 480)
                depth_image = depth_array.reshape((640, 480))
                self.latest_depth = depth_image
                
                # 更新FPS統計
                self.depth_frame_count += 1
                current_time = time.time()
                if current_time - self.last_depth_time >= 1.0:
                    self.depth_fps = self.depth_frame_count
                    self.depth_frame_count = 0
                    self.last_depth_time = current_time
                    
            else:
                print(f"深度數據大小錯誤，期望{expected_pixels}像素，實際{len(depth_array)}像素")
                # 嘗試修正數據
                if len(frame_data) >= expected_pixels * 2:
                    corrected_data = frame_data[:expected_pixels * 2]
                    depth_array = np.frombuffer(corrected_data, dtype=np.uint16)
                    depth_image = depth_array.reshape((640, 480))
                    self.latest_depth = depth_image
            
            del self.depth_buffer[frame_id]
            
        except Exception as e:
            print(f"深度幀組裝錯誤: {e}")
            if frame_id in self.depth_buffer:
                del self.depth_buffer[frame_id]

    def assemble_color_depth_frame(self, frame_id, total_packets):
        """組裝彩色深度圖幀 (用於統計監控)"""
        try:
            # 組裝數據
            frame_data = b''
            for i in range(total_packets):
                if i in self.color_depth_buffer[frame_id]:
                    frame_data += self.color_depth_buffer[frame_id][i]
                else:
                    print(f"C#彩色深度幀{frame_id}缺少包{i}/{total_packets}")
                    return

            # 解碼PNG圖像 (用於驗證數據完整性)
            img_array = np.frombuffer(frame_data, dtype=np.uint8)
            color_depth_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if color_depth_image is not None:
                # 儲存最新的彩色深度圖 (供未來可能的使用)
                self.latest_color_depth = color_depth_image
                
                # 更新FPS統計
                self.color_depth_frame_count += 1
                current_time = time.time()
                if current_time - self.last_color_depth_time >= 1.0:
                    self.color_depth_fps = self.color_depth_frame_count
                    self.color_depth_frame_count = 0
                    self.last_color_depth_time = current_time
            
            del self.color_depth_buffer[frame_id]
            
        except Exception as e:
            print(f"C#彩色深度幀組裝錯誤: {e}")
            if frame_id in self.color_depth_buffer:
                del self.color_depth_buffer[frame_id]

    def cleanup_old_frames(self):
        """清理舊的不完整幀"""
        while self.running:
            current_time = time.time()
            
            with self.rgb_lock:
                old_frames = [fid for fid in self.rgb_buffer.keys() 
                             if current_time - (fid / 1000.0) > 3]
                for fid in old_frames:
                    del self.rgb_buffer[fid]
            
            with self.depth_lock:
                old_frames = [fid for fid in self.depth_buffer.keys() 
                             if current_time - (fid / 1000.0) > 3]
                for fid in old_frames:
                    del self.depth_buffer[fid]
            
            with self.color_depth_lock:
                old_frames = [fid for fid in self.color_depth_buffer.keys() 
                             if current_time - (fid / 1000.0) > 3]
                for fid in old_frames:
                    del self.color_depth_buffer[fid]
            
            time.sleep(1)

    def get_rgb_frame(self):
        """獲取最新RGB幀 (C#端已旋轉的480x640)"""
        return self.latest_rgb

    def get_depth_frame(self):
        """獲取最新深度幀 (C#端已旋轉的480x640)"""
        return self.latest_depth

    def get_color_depth_frame(self):
        """獲取最新C#彩色深度圖幀 (C#端已旋轉的480x640)"""
        return self.latest_color_depth

    def get_depth_at_point(self, x, y):
        """獲取指定點的深度值"""
        if self.latest_depth is None:
            return None
        
        h, w = self.latest_depth.shape
        if 0 <= x < w and 0 <= y < h:
            raw_depth = self.latest_depth[y, x]
            actual_depth = raw_depth >> 3  # 移除Player Index位元
            return actual_depth if actual_depth > 0 else None
        return None

    def is_data_available(self):
        """檢查是否有可用數據"""
        return self.latest_rgb is not None and self.latest_depth is not None

    def get_stats(self):
        """獲取統計信息"""
        return {
            'rgb_fps': self.rgb_fps,
            'depth_fps': self.depth_fps,
            'color_depth_fps': self.color_depth_fps,
            'rgb_buffer_size': len(self.rgb_buffer),
            'depth_buffer_size': len(self.depth_buffer),
            'color_depth_buffer_size': len(self.color_depth_buffer)
        }

    def close(self):
        """關閉接收器"""
        print("正在關閉Kinect數據接收器...")
        self.running = False
        
        try:
            self.rgb_socket.close()
            self.depth_socket.close()
            self.color_depth_socket.close()
        except:
            pass
        
        print("Kinect數據接收器已關閉")