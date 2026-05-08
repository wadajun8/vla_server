import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import os
import sys
import torch

# modelsのパス探し
def get_workspace_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != '/':
        if os.path.isdir(os.path.join(current_dir, 'models')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.getcwd() # 見つからない場合の保険

# パスを自動計算して通す
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

        # モデルのロード
        self.model, self.text_encoder, self.preprocess = load_model(
            self.model_weights_path, model_params, self.device
        )
        self.model = self.model.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.get_logger().info('🎉 AIモデルの読み込み完了！')

        # ==========================================
        # カメラとの接続設定
        # ==========================================
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            1) # 最新の画像だけあればいいのでキューは1
        
        # VRAM対策：画像を小さく切り詰めるサイズ設定
        self.imgsize_96 = (96, 96)
        self.imgsize_224 = (224, 224)
        
        # AIへのテキスト指示（とりあえず固定）
        self.current_instruction = "blue trash bin"

        self.get_logger().info('👀 カメラからの映像を待機しています...')

    def image_callback(self, msg):
        """カメラから画像が届くたびに呼ばれる関数"""
        try:
            # 1. ROSの画像データ(sensor_msgs/Image)を、Pythonの画像(OpenCV)に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'画像変換エラー: {e}')
            return

        # 2. OpenCV(BGR)から、AIが好きなPIL(RGB)形式に変換
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image_rgb)

        # 3. 巨大な画像をAI用のサイズに圧縮（VRAM 4GBの命綱！）
        pil_image_96 = pil_image.resize(self.imgsize_96)
        pil_image_224 = pil_image.resize(self.imgsize_224)

        self.get_logger().info(f'画像を受信＆リサイズ完了 指示: [{self.current_instruction}]')
        
        # ⚠️ 注意: ここに推論処理を直書きすると、カメラの30fps（秒間30回）のペースで
        # AIがフル稼働してしまい、ノートPCが一瞬でフリーズします。
        # 次のステップで、これを「1秒に数回だけ考える」ように交通整理します。

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
