`RollingContact`サンプルは、通常のCPU上で動力学的に整合した剛体車輪接触を実装します。
差動二輪の`rolling_diff`、四輪操舵の`rolling_4s`、Tasks/TVMの両バックエンド、接触モード遷移、
ヘッドレスMuJoCo検証を含みます。CUDA、Torch、GPUは使用せず、必要でもありません。

## 接触座標系と拘束

各車輪は、キャリア中心フレーム、力を受ける車輪ボディ、駆動関節、任意の操舵関節、半径、幅、
摩擦係数、回転符号で記述します。地面法線`n`から世界座標系の右手系基底`[t, l, n]`を生成します。

```text
                 車軸・横方向 l
       lineStart o-----------o lineEnd
                         c        c: キャリア中心
                         | r
地面 --------------------p--------------------------> t
                         ^ n

l = n x t          p = c - r n          wrench順序=[角; 並進]
```

キャリアJacobianにはキャリア運動だけを含め、リム速度は名前で解決した駆動関節セレクタから一度だけ
加えます。理想転がりでは長手`(t^T J_c-r S_drive) alpha=0`、横滑り`l^T J_c alpha=0`、
法線`n^T J_c alpha=0`を課します。加速度レベルでは有限差分で検証した`A_dot alpha`と速度安定化項も
使用します。接触線の両端には4方向摩擦ピラミッドがあり、浮遊基部動力学・駆動トルク・仮想仕事と
同じQP内で整合します。

## C++ APIと設定

車輪記述には{% doxygen mc_rbdyn::RollingContactDescription %}、運動学拘束には
{% doxygen mc_solver::RollingContactConstraint %}、接触力と動力学には
{% doxygen mc_solver::RollingContactDynamicsConstraint %}を使用します。両制約はTasksとTVMで共通です。

```cpp
mc_rbdyn::RollingContactDescription wheel;
wheel.name = "left";
wheel.carrierFrame = "left_carrier";
wheel.wheelBody = "left_wheel";
wheel.driveJoint = "left_drive";
wheel.radius = 0.2;
wheel.width = 0.08;
wheel.friction = 0.8;
wheel.validate();

mc_solver::RollingContactConstraintOptions options;
options.differentialPlanar = true;
auto dynamics = std::make_unique<mc_solver::RollingContactDynamicsConstraint>(
    robots(), 0, solver().dt(), wheels);
auto rolling = std::make_unique<mc_solver::RollingContactConstraint>(robots(), 0, wheels, options);
solver().addConstraintSet(*dynamics);
solver().addConstraintSet(*rolling);
```

登録中はオブジェクトを生存させ、破棄時は`rolling`、`dynamics`の順で削除します。四輪操舵では各車輪の
`steeringJoint`を設定し、`steeringPlanar: true`と独立な2車輪の`steeringPlanarWheels`を指定します。
JSONスキーマは`RollingContactWheel`、`RollingContactConstraint`、`RollingContactDynamicsConstraint`、
`RollingContactController`として提供されています。

## CPUだけでビルド・実行

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 2
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure

CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  build/utils/mc_rtc_ticker \
  -f rolling-contact-report/config/mc_rtc-differential.yaml \
  --run-for 5 --no-sync
```

Tasks用設定は`mc_rtc-differential.yaml`と`mc_rtc-four-steering.yaml`、TVM用は名前に`-tvm`が付く設定です。
差動ロボットは直進・後退・その場旋回・円・正弦・不等半径・モード遷移、四輪操舵は直進・後退・
クラブ・Ackermann旋回・純ヨー・操舵速度・不整合指令・モード遷移を検証します。

## モードと安全な復帰

| モード | 運動学 | 接触力 |
| --- | --- | --- |
| `fixed` | 長手・横・法線を固定 | 全摩擦ピラミッド |
| `rolling` | 転がり長手・横・法線 | 全摩擦ピラミッド |
| `sliding` | 滑っている拘束を解除 | 滑りに逆らう確定済み生成方向 |
| `detached` | 接触行を解除 | 変数配置を保ち全接触力をゼロ固定 |

モード管理はヒステリシス、最小滞留時間、フィルタ、activationランプを使用します。無効・古い外部計測は
安全側の`detached`になります。復帰中は長手行をsoftにし、加速度残差が十分小さくなってからhardへ
昇格します。`recoveryRollingWeight`は`rollingWeight`以上でなければなりません。

ログには、ソルバ成否、バックエンド、CPU時間、動力学残差、浮遊基部努力、各車輪の残差・モード・
activation・QP接触力・摩擦余裕・駆動トルク余裕・外部計測有効性が含まれます。TVMでは駆動関節だけを
トルク変数にし、浮遊基部の6成分は常にゼロです。基部に架空の支持力を与えることはありません。

## CPU MuJoCo検証

```sh
rolling-contact-report/scripts/build-cpu-mujoco-runner.sh
python3 rolling-contact-report/scripts/run-mujoco-suite.py \
  --runner /tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner \
  --artifact-dir /tmp/rolling-contact-mujoco-suite \
  --backend both --cycles 1000 --warmup-cycles 100 --repetitions 5
```

スクリプトは`mc_mujoco`とMuJoCoを固定バージョンで`/tmp`へ構築し、GPUランタイムへのリンクを監査します。
平面、坂、低摩擦、外乱、接触喪失、復帰、全走行シナリオを描画なしで実行し、JSON/CSV証拠を出力します。

## 制限とトラブルシューティング

- フレーム・ボディ・関節名は構築時に検証されます。対象ロボットモジュールの名前と完全に一致させてください。
- `terrainNormal`は有限かつ非ゼロの世界座標法線です。オンライン地形推定は対象外です。
- 直ちにdetach/slidingになる場合は、重みを変更する前に計測有効性、法線力、摩擦・トルク余裕、閾値を確認します。
- hard復帰に失敗する場合はsoft復帰とhard昇格残差を確認します。
- MuJoCoモデルが見つからない場合は、提供のビルドスクリプトで隔離されたユーザー設定を再生成します。

対象は平面または一定勾配上の通常剛体車輪と多面体摩擦です。タイヤ変形、転がり抵抗、任意高さ地形、
相補性、キャスタ、オムニ/メカナム車輪、履帯、学習型滑り推定、ハードウェア固有センサは対象外です。
