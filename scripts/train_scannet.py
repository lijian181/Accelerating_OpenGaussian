import os
import subprocess

# 10个场景列表
scan_list = [
    #"scene0000_00",
    #"scene0062_00",
    "scene0070_00",
    "scene0097_00", "scene0140_00", "scene0200_00",
    "scene0347_00", "scene0400_00", "scene0590_00",
    "scene0645_00"
]

# GPU编号（根据实际情况修改）
gpu_num = 0

# 数据集根路径（Windows路径格式，根据实际情况修改）
data_root = r"E:\opengaussians\OpenGaussian-main\data\scannet"

for scan in scan_list:
    print(f"Training for {scan} .....")
    
    # 构建场景完整路径
    scene_path = os.path.join(data_root, scan)
    
    # 构建命令参数列表
    cmd = [
        "python", "../train.py",
        "--port", f"601{gpu_num}",
        "-s", scene_path,
        "-r", "2",
        "--frozen_init_pts",
        "--iterations", "90_000",
        "--start_ins_feat_iter", "30_000",
        "--start_root_cb_iter", "50_000",
        "--start_leaf_cb_iter", "70_000",
        "--sam_level", "0",
        "--root_node_num", "64",
        "--leaf_node_num", "5",
        "--pos_weight", "1.0",
        "--test_iterations", "30000",
        "--eval"
    ]

    print(cmd)

    # 设置CUDA可见设备环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    
    # 执行命令
    result = subprocess.run(cmd, env=env, check=True)
    if result.returncode != 0:
        print(f"Error occurred while training {scan}")
        break