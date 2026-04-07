import subprocess
import os

def run_training(scan, gpu_num, root_node_num, leaf_node_num, pos_weight, 
                 save_memory=False, loss_weight=None, data_base_path="E:\opengaussians\OpenGaussian-main\data\lerf_ovs"):
    """
    运行训练命令的函数
    :param scan: 场景名称
    :param gpu_num: GPU编号
    :param root_node_num: 根节点数量
    :param leaf_node_num: 叶节点数量
    :param pos_weight: 位置权重
    :param save_memory: 是否启用内存节省模式
    :param loss_weight: 损失权重（可选）
    :param data_base_path: 数据基础路径
    """
    print(f"Training for {scan} .....")
    
    # 构建命令列表
    cmd = [
        "python", "../train_feat.py",
        f"--port", f"601{gpu_num}",
        "-s", os.path.join(data_base_path, scan),
        "--iterations", "70_000",
        "--start_ins_feat_iter", "30_000",
        "--start_root_cb_iter", "40_000",
        "--start_leaf_cb_iter", "50_000",
        "--sam_level", "3",
        "--root_node_num", str(root_node_num),
        "--leaf_node_num", str(leaf_node_num),
        "--pos_weight", str(pos_weight),
        "--test_iterations", "30000",
        "--eval"
    ]
    
    # 添加可选参数
    if save_memory:
        cmd.append("--save_memory")
    if loss_weight is not None:
        cmd.extend(["--loss_weight", str(loss_weight)])
    
    # 设置CUDA可见设备环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    
    # 执行命令
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    # 1. figurines场景
    run_training(
        scan="figurines",
        gpu_num=0,
        root_node_num=64,
        leaf_node_num=10,
        pos_weight=0.5,
        save_memory=True
    )
    
    # #2. waldo_kitchen场景
    # run_training(
    #     scan="waldo_kitchen",
    #     gpu_num=0,
    #     root_node_num=64,
    #     leaf_node_num=10,
    #     pos_weight=0.5,
    #     save_memory=True
    # )
    #
    # # 3. teatime场景
    # run_training(
    #     scan="teatime",
    #     gpu_num=0,
    #     root_node_num=32,
    #     leaf_node_num=10,
    #     pos_weight=0.1,
    #     save_memory=True
    # )
    #
    # # 4. ramen场景
    # run_training(
    #     scan="ramen",
    #     gpu_num=0,
    #     root_node_num=64,
    #     leaf_node_num=10,
    #     pos_weight=0.5,
    #     save_memory=True,
    #     loss_weight=0.01
    # )