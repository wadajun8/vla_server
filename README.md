# ROS2でOmniVLAを使うためのノード
- まだ動かせるだけで出力が正しいとは限らないです・・・
## `vla_node`ノード
- トピック`/camera/color/image_raw`（sensor\_msgs/msg/Image型）を受け取り、リサイズ
- トピック`/vla/instruction`（std\_msg/msg/String型）のmsg.dataを受け取り、指示を更新
- OmniVLA-edgeに推論させる
- トピック`linear.x`(直進の速度)、`angular.z`（旋回の速度）をパブリッシュ
  - どちらもgeometry \_msgs/msg/Twist型
- 機能追加中…
