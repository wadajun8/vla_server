import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist  # ★追加：ロボットの速度を表現する型
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import os
import sys
import time  # ★追加：時間を測るための道具
import torch

# ==========================================
# 🔍 汎用パス探索関数 (変更なし)
# ==========================================
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

from utils_policy import load_model

class VLAServer(Node):
    def __init__(self):
        super().__init__('vla_server')
        self.get_logger().info('🧠 VLAノードを起動中...')

        self.model_weights_path = os.path.join(ws_root, "models/omnivla-edge/omnivla-edge.pth")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        self.get_logger().info('🎉 AIモデルの読み込み完了！')

        # ==========================================
        # 📡 ROS 2 通信設定
        # ==========================================
        self.bridge = CvBridge()
        
        # 受信（Subscriber）：カメラ画像
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 1)
        
        # ★追加 送信（Publisher）：計算した速度指示
        self.cmd_vel_pub = self.create_publisher(Twist, '/vla/cmd_vel', 1)

        self.imgsize_96 = (96, 96)
        self.imgsize_224 = (224, 224)
        self.current_instruction = "blue trash bin"

        # ★追加 負荷対策：最後にAIが考えた時間を記録
        self.last_inference_time = 0.0
        # ★追加 考える間隔（秒）。VRAM 4GBならまずは1秒（1.0）か1.5秒間隔で様子見
        self.inference_interval = 1.0 

        self.get_logger().info('👀 カメラ映像待機中... 速度指令を /vla/cmd_vel に送信します')

    def image_callback(self, msg):
        current_time = time.time()

        # ★追加：前回考えてから、指定した時間（1.0秒）経っていなければ、画像を無視して処理を終わる
        if current_time - self.last_inference_time < self.inference_interval:
            return

        # 1.0秒以上経っていれば、画像を処理してAIに考えさせる！
        self.last_inference_time = current_time # 時間を更新

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'画像変換エラー: {e}')
            return

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image_rgb)

        pil_image_96 = pil_image.resize(self.imgsize_96)
        pil_image_224 = pil_image.resize(self.imgsize_224)

        self.get_logger().info(f'🤔 推論開始... 指示: [{self.current_instruction}]')

        # ==========================================
        # 🏃 推論＆運動（Twist）への変換
        # ==========================================
        # ※ここに先ほどの `run_omnivla_edge.py` にあったテンソル処理や
        # PDコントローラの計算をごっそり移植することになりますが、
        # まずは通信テストとして「ダミーの速度」を出力します！

        dummy_linear = 0.1  # 直進速度 (m/s)
        dummy_angular = -0.05 # 旋回速度 (rad/s)

        # Twistメッセージを作って送信
        cmd_msg = Twist()
        cmd_msg.linear.x = float(dummy_linear)
        cmd_msg.angular.z = float(dummy_angular)
        
        self.cmd_vel_pub.publish(cmd_msg)
        self.get_logger().info(f'🚀 速度送信: v={dummy_linear}, w={dummy_angular}')

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAServer()
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
