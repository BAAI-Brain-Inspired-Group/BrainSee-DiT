# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance, InceptionV3
import os



def get_fid_score(batch_size, dims, num_workers=1):
    data_dir = '/share/project/dataset/cfgnew_png'
    # items = ["zero", "0", "1", "2", "3", "4", "all"]
    items = [
        # "dit_fid_imagenet_czq_4/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.0",
        # "dit_fid_imagenet_czq_4/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.25",
        # "dit_fid_imagenet_czq_4/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.5",
        # "dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.0",
        # "dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.25",
        # "dit_fid_imagenet_czq_04/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/6930000_1.5",
        # # "dit_fid_imagenet_czq/000-mask-1.0-0-8-myDiT-XL-2/checkpoints/3000000_1.25",
        "dit_fid_imagenet_czq_4/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.0",
        "dit_fid_imagenet_czq_4/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.25",
        "dit_fid_imagenet_czq_4/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.5",
        "dit_fid_imagenet_czq_34/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.0",
        "dit_fid_imagenet_czq_34/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.25",
        "dit_fid_imagenet_czq_34/001-mask-0.94-60000-8-myDiT-XL-2/checkpoints/6930000_1.5",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # import pdb;pdb.set_trace()
    fid_value = []
    # m1, s1 = compute_statistics_of_path(os.path.join(data_dir, items[0], 'img'), model, batch_size,
    #                                     dims, device, num_workers)
    # m1, s1 = compute_statistics_of_path('/share/project/dataset/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/val', model, batch_size,
    #                                     dims, device, num_workers)
    m1, s1 = compute_statistics_of_path('/home/zqchen/code/mask_dit/fid/VIRTUAL_imagenet256_labeled.npz', model, batch_size,
                                        dims, device, num_workers)
    name = ['zero','all','0','1','2','3','4']
    idx = [1,1,1,1,1,1]
    # import pdb;pdb.set_trace()
    for i in range(len(items)):
        m2, s2 = compute_statistics_of_path(os.path.join(data_dir, items[i], name[idx[i]]), model, batch_size,
                                            dims, device, num_workers)
        value = calculate_frechet_distance(m1, s1, m2, s2)
        fid_value.append(value)
        print(os.path.join(data_dir, items[i], name[idx[i]]), ":", value)

    return fid_value

def plot():
    import numpy as np
    import matplotlib.pyplot as plt

    # 用户提供的数据列表
    data = [80.9564306988882, 55.05187762457376, 36.77178879719958, 29.547067587778884, 24.931210764108698, 
            83.61819099466067, 57.2197682477518, 38.79723251213153, 31.033452195804443, 26.337462249233113, 
            34.96314113521663, 28.396801984305114, 21.50994119392152, 17.55326905634098, 15.918280124306364, 
            26.757495982169303, 21.98245457208725, 17.343247082632274, 14.539818289967513, 12.999233987603702, 
            86.79733569431556, 62.1687900614138, 44.889572381215544, 37.27486245629234, 32.73474600542073, 
            33.610068580264624, 20.25691886994241, 13.851099086691192, 11.293411564214466, 9.892830309460123, 
            26.519676492058807, 16.750684722349888, 12.08113175566126, 9.866963387396424, 8.750219200923539]

    # 验证数据长度
    data_length = len(data)
    n = data_length // 5
    if data_length % 5 != 0:
        raise ValueError(f"数据长度 {data_length} 不是5的倍数，无法reshape为n×5矩阵")

    # 转换为5×n矩阵
    matrix = np.array(data).reshape(n, 5)
    print(matrix)

    # 横坐标
    x = [5, 10, 20, 30, 40]

    plt.figure(dpi=300)
    for i in range(n):
        plt.plot(x,matrix[i],'.-')

    plt.plot([5,40],[19.47,19.47],'--')
    plt.plot([5,40],[9.62,9.62])

    plt.savefig("fig_result/fid_cfg1_compare_5_40.jpg")

def plot_img():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    # 共同x坐标
    x = np.array([1., 1.2, 1.25, 1.3, 1.5])

    # 两组y值
    y1 = np.array([10.21,4.40,3.84,3.52,4.14])
    y2 = np.array([10.73,4.84,4.12,3.68,4.16])

    # 创建平滑曲线
    x_smooth = np.linspace(x.min(), x.max(), 300)  # 300个点用于平滑

    # 对第一组数据平滑
    spl1 = make_interp_spline(x, y1, k=3)  # 三次样条
    y1_smooth = spl1(x_smooth)

    # 对第二组数据平滑
    spl2 = make_interp_spline(x, y2, k=3)  # 三次样条
    y2_smooth = spl2(x_smooth)

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制原始点
    plt.plot(x, y1, 'bo', label='no', markersize=8)
    plt.plot(x, y2, 'rs', label='condition-0.94', markersize=8)

    # 绘制平滑曲线
    plt.plot(x_smooth, y1_smooth, 'b-', label='no', linewidth=2, alpha=0.7)
    plt.plot(x_smooth, y2_smooth, 'r--', label='condition-0.94', linewidth=2, alpha=0.7)

    # 添加图例和标题
    plt.legend(fontsize=12)
    plt.title('cfg', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig("con1.jpg")

if __name__ == "__main__":
    print(get_fid_score(batch_size=1000, dims=2048))
    # plot_img()
    # plot()