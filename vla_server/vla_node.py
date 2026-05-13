# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Junya Wada

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import os
import sys
import time
import torch
import numpy as np
import clip
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# 汎用パス探索関数
def get_workspace_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != '/':
        if os.path.isdir(os.path.join(current_dir, 'models')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.getcwd()

ws_root = get_workspace_root()
omnivla_path = os.path.join(ws_root, "extern/OmniVLA")
sys.path.insert(0, omnivla_path)
sys.path.insert(0, os.path.join(omnivla_path, "inference"))

# AIの道具をインポート
from utils_policy import load_model, transform_images_map, transform_images_PIL_mask

class VLAServer(Node):
    def __init__(self, initial_instruction):
        super().__init__('vla_server')
        self.get_logger().info('VLAノード起動中...')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_weights_path = os.path.join(ws_root, "models/omnivla-edge/omnivla-edge.pth")
        
        model_params = {
            "model_type": "omnivla-edge", "len_traj_pred": 8, "learn_angle": True,
            "context_size": 5, "obs_encoder": "efficientnet-b0", "encoding_size": 256,
            "obs_encoding_size": 1024, "goal_encoding_size": 1024, "late_fusion": False,
            "mha_num_attention_heads": 4, "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4, "clip_type": "ViT-B/32"
        }

        self.model, self.text_encoder, self.preprocess = load_model(
            self.model_weights_path, model_params, self.device
        )
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.get_logger().info('VLAモデルの読み込み完了')

        # 📡 ROS 2 通信設定
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 1)
        # cmd_velをパブリッシュ
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.inst_sub = self.create_subscription(String, '/vla/instruction', self.instruction_callback, 10)

        # ⚙️ AI用パラメータ
        self.imgsize_96 = (96, 96)
        self.imgsize_224 = (224, 224)
        self.current_instruction = initial_instruction # 言語指示
        self.metric_waypoint_spacing = 0.1 # 距離スケール

        # マスク（カメラ視野外の黒塗り用ですが、今回は全白で透過）
        self.mask_360_pil_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)

        # 🧠 記憶バッファ（最新の画像を6枚保持する）
        self.context_queue = []

        # 負荷対策
        self.last_inference_time = 0.0
        self.inference_interval = 1.0 

        self.get_logger().info('カメラ映像待機中')

    # 葉を受け取った時に呼ばれる関数
    def instruction_callback(self, msg):
        # 今の指示と違う言葉が来た時だけ更新する
        if self.current_instruction != msg.data:
            self.current_instruction = msg.data
            self.get_logger().info(f'新しい指示: [{self.current_instruction}]')

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_inference_time < self.inference_interval:
            return
        self.last_inference_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'画像変換エラー: {e}')
            return

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image_rgb)
        pil_image_96 = pil_image.resize(self.imgsize_96)
        pil_image_224 = pil_image.resize(self.imgsize_224)

        # 記憶バッファの更新（常に6枚を保つ）
        self.context_queue.append(pil_image_96)
        if len(self.context_queue) > 6:
            self.context_queue.pop(0)
        # 6枚溜まるまでは動かない
        if len(self.context_queue) < 6:
            self.get_logger().info(f'記憶バッファ充填中 ({len(self.context_queue)}/6)')
            return

        self.get_logger().info(f'推論中 指示: [{self.current_instruction}]')

        # ==========================================
        # 🏃 OmniVLA 推論プロセス
        # ==========================================
        with torch.no_grad(): # メモリ節約のおまじない
            # 1. 過去の映像履歴（obs_images）の準備
            obs_images = transform_images_PIL_mask(self.context_queue, self.mask_360_pil_96)
            obs_images = torch.split(obs_images.to(self.device), 3, dim=1)
            obs_image_cur = obs_images[-1].to(self.device) 
            obs_images = torch.cat(obs_images, dim=1).to(self.device)

            # 2. 現在の高解像度画像（cur_large_img）
            cur_large_img = transform_images_PIL_mask(pil_image_224, self.mask_360_pil_224).to(self.device)

            # 3. ダミーマップとダミーゴール（今回は「言語のみ」で動かすため）
            satellite_dummy = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
            current_map_image = transform_images_map(satellite_dummy)
            goal_map_image = transform_images_map(satellite_dummy)
            map_images = torch.cat((current_map_image.to(self.device), goal_map_image.to(self.device), obs_image_cur), axis=1)

            goal_image_dummy = transform_images_PIL_mask(pil_image_96, self.mask_360_pil_96).to(self.device)

            # 4. GPSダミー座標
            goal_pose_torch = torch.zeros((1, 4)).float().to(self.device)

            # 5. 言語指示のトークン化
            obj_inst_lan = clip.tokenize(self.current_instruction, truncate=True).to(self.device)
            feat_text_lan = self.text_encoder.encode_text(obj_inst_lan)

            # 6. モダリティ指定（7 = language only）
            modality_id_select = torch.tensor([7]).to(self.device)

            # 🚀 いざ、推論実行！
            predicted_actions, distances, mask_number = self.model(
                obs_images, 
                goal_pose_torch, 
                map_images, 
                goal_image_dummy, 
                modality_id_select, 
                feat_text_lan, 
                cur_large_img
            )

        # ==========================================
        # ⚙️ PDコントローラ（Action -> Twist変換）
        # ==========================================
        waypoints = predicted_actions.float().cpu().numpy()
        chosen_waypoint = waypoints[0][4].copy() # 少し先の未来(4)の座標を採用
        chosen_waypoint[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        EPS = 1e-8
        DT = 1.0 / 3.0 # 元のサンプリングレートベース
        
        # 速度計算
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * math.atan2(hy, hx) / DT
        elif np.abs(dx) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel = dx / DT
            angular_vel = np.arctan(dy / dx) / DT

        # モータへの過負荷を防ぐための制限（クリッピング）
        linear_vel = float(np.clip(linear_vel, 0.0, 0.3))
        angular_vel = float(np.clip(angular_vel, -0.5, 0.5))

        # Twist送信
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_vel
        cmd_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd_msg)
        
        self.get_logger().info(f'速度出力: 前進={linear_vel:.2f}, 旋回={angular_vel:.2f}')

def main(args=None):
    passed_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

    if len(passed_args) > 0:
        initial_instruction = " ".join(passed_args)
        print(f"指示: [{initial_instruction}]")
    else:
        user_input = input ("指示待ち中: ")
        initial_instruction = user_input.strip() if user_input.strip() else "sty"

    rclpy.init(args=args)
    vla_node = VLAServer(initial_instruction)

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
