## 文件目录

```
sight/
├── README.md                          # 项目说明、指标公式、实验结果与运行命令
├── ALGORITHMS.md                      # HAP-Deblur / NRC-HybridNet 算法原理与实现说明
│
├── GOPRO_Large/                       # GoPro 运动模糊数据集（blur_gamma / sharp 配对）
│   ├── train/                         # 训练集，2103 张配对图像
│   └── test/                          # 测试集，1111 张配对图像
│
├── evaluate_baseline.py               # 对照组：原始 blur_gamma 相对 sharp 计算 PSNR/SSIM
├── evaluate_mean_filter.py            # 基础组：均值滤波评估
├── evaluate_median_filter.py          # 基础组：中值滤波评估
├── evaluate_gaussian_filter.py        # 基础组：高斯滤波评估
├── evaluate_freq_filter.py            # 频域组：理想/巴特沃斯低通、高通 FFT 滤波评估
├── evaluate_wiener.py                 # 改进组：HAP-Deblur 空频结合去模糊评估
├── evaluate_dl_deblur.py              # 深度学习组：NRC-HybridNet 评估（需 yolov8 环境）
│
├── freq_filters.py                    # 频域滤波核心：理想/巴特沃斯低通、高通 FFT 实现
├── deblur_wiener.py                   # HAP-Deblur 核心：频谱估核 + RL 反卷积 + 空域增强
├── deblur_dl.py                       # NRC-HybridNet 核心：MIMO-UNet 推理 + NR 融合 + 分块推理
├── process_media.py                   # 批量图像/视频去模糊（--method hap 或 dl）
├── train_deblur.py                    # MIMO-UNet 训练脚本（fine-tune，需 yolov8 环境）
│
├── models/
│   ├── mimo_unet.py                   # MIMO-UNet / MIMO-UNet+ 网络结构定义
│   ├── mimo_layers.py                 # 网络基础层：BasicConv、ResBlock
│   ├── MIMO-UNet.pkl                  # GoPro 预训练权重（推理默认加载）
│   ├── deblurring_nafnet_2025may.onnx # NAFNet ONNX 模型（备用）
│   └── finetune/
│       ├── best.pkl                     # 训练最佳权重
│       ├── last.pkl                     # 训练最新权重
│       └── train_log.csv                # 训练日志（loss / PSNR）
│
├── results/                           # 各实验逐张评估结果 CSV
│   ├── baseline_blur_gamma.csv        # 对照组结果
│   ├── mean_filter.csv                # 均值滤波结果
│   ├── median_filter.csv              # 中值滤波结果
│   ├── gaussian_filter.csv            # 高斯滤波结果
│   ├── freq_filter_summary.csv        # 频域四种滤波汇总
│   ├── freq_ideal_lpf.csv             # 理想低通结果
│   ├── freq_ideal_hpf.csv             # 理想高通结果
│   ├── freq_butterworth_lpf.csv       # 巴特沃斯低通结果
│   ├── freq_butterworth_hpf.csv       # 巴特沃斯高通结果
│   ├── hap_deblur.csv                 # HAP-Deblur 结果
│   └── dl_nrc_hybridnet.csv           # NRC-HybridNet 深度学习结果
│
└── image/                             # 实验效果对比图（README 插图）
    ├── image_1.png                    # 对照组：原始模糊
    ├── image_2.png ~ image_8.png      # 基础组：空域/频域滤波效果
    ├── image_9.png                    # 改进组：HAP-Deblur 效果
    └── image_10.png                   # 深度学习组：NRC-HybridNet 效果
```

## PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）
### 衡量像素级误差，单位 dB。
先计算参考图 $I_{\text{ref}}$（sharp）与待评图 $I_{\text{pred}}$（模糊/恢复结果）的均方误差 MSE，再换算为 dB：

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\bigl(I_{\text{ref}}(i) - I_{\text{pred}}(i)\bigr)^2
$$

$$
\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)
$$

其中 $N$ 为像素总数，$\text{MAX}$ 为像素最大值（8 位图像取 255）。

- PSNR 越高，像素越接近参考图
- 一般 >30 dB 较好，>40 dB 很好

## SSIM（Structural Similarity Index，结构相似度）

SSIM 从**亮度、对比度、结构**三方面衡量两幅图的相似性，比 PSNR 更接近人眼感受。

对参考图 $x$ 与待评图 $y$，在局部窗口内计算均值 $\mu_x, \mu_y$，方差 $\sigma_x^2, \sigma_y^2$，协方差 $\sigma_{xy}$，则：

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$
- 取值范围 $[0, 1]$：1 表示完全一致，0 表示完全不相似
- 彩色图像对 R/G/B 三通道分别计算 SSIM 后取平均（代码中 `channel_axis=2`）

## HAP-Deblur（Hybrid Adaptive Parallel Deblur） 
改进方案：空频结合 + 序列/视频共享核 + 批处理，避免传统滤波「越处理越糊」的问题。
| 模块 | 做法 |
| :--: | :--: |
| 频域 | 频谱估计运动方向 → Richardson-Lucy 反卷积 |
| 空域 | 双边滤波抑振铃 + Unsharp 增强细节 |
| 自适应 | 自动选择原图/去模糊融合比例（0~20%），错误核时锐度回退 |
| 序列批处理 | 同一 GOPRO 序列共享模糊核，只估计一次，大幅提速 |
| 视频 | 首帧锁定核 + 3 帧时域中值融合，提升稳定性

模糊图 → 运动方向估计（频域）→ 模糊核估计 → RL反卷积恢复 → 空间域去振铃 → 细节增强 → 自适应融合（避免翻车）→ 最终图像

先在频域估计模糊核，再反卷积恢复；用空间域去伪影；最后自适应回退避免翻车，并在序列里共享核保证速度与稳定性。

## NRC-HybridNet：

MIMO-UNet 多尺度 DL 去模糊（GoPro 预训练 / 可 fine-tune）：多输入多输出 U-Net，内部在 1/4、1/2、全分辨率三个尺度做监督
推理：只取最后一个全分辨率输出 preds[-1]
加速：GPU + FP16 混合精度。

NR 置信度融合（无需 GT 的后处理）：不需要清晰真值，用两个无参考指标评估 DL 输出是否可信。

频域残差注入（DL 去模糊 + 原图高频补纹理）：DL 容易过平滑。从原模糊图提取高频分量，按比例加回融合结果。

分块重叠推理 + 视频时序 EMA ： 图像大于 512×512 时，切成 512×512 块，32 px 重叠，Hann 窗加权融合，避免块边界伪影。视频模式下，对相邻帧 DL 结果做指数滑动平均，抑制闪烁。

## 1 对照组

原始模糊输入
![img.png](image\image_1.png)

## 2 基础组

均值滤波
![img.png](image\image_2.png)

中值滤波
![img.png](image\image_3.png)

高斯滤波
![img.png](image\image_4.png)

巴特沃斯低通（只跑100张）  
![img.png](image\image_5.png)

巴特沃斯高通（只跑100张）
![img.png](image\image_6.png)

理想低通（只跑100张）   
![img.png](image\image_7.png)

理想高通（只跑100张）
![img.png](image\image_8.png)


## 3 改进组
HAP-Deblur（Hybrid Adaptive Parallel Deblur） 
![img.png](image\image_9.png)

NRC-HybridNet
模型：MIMO-UNet.pkl
![img.png](image\image_10.png)
模型：自训练best.pth
![img.png](image\train_model.png)
![img.png](image\image_11.png)

## 4 滤波效果对比表

| 实验组别 | PSNR (dB) | SSIM | time (ms/img) |
| :--: | :--: | :--: | :--: |
| 对照组 | 25.2 | 0.78 | 332 |
| 均值滤波 | 24.5 | 0.75 | 391 |
| 中值滤波 | 24.8 | 0.76 | 394 |
| 高斯滤波 | 24.9 | 0.77 | 387 |
| 巴特沃斯低通 | 21.3 | 0.66 | 600 |
| 巴特沃斯高通 | 6.3 | 0.04 | 584 |
| 理想低通 | 21.1 | 0.63 | 585 |
| 理想高通 | 6.4 | 0.05 | 605 |
| HAP-Deblur | 24.1 | 0.75 | 3877 |
| NRC-HybridNet(1) | 27.45 | 0.898 | 769 |
| NRC-HybridNet(1) | 26.17 | 0.824 | 695 |
## 5 总结
均值滤波/中值滤波/高斯滤波：运动模糊无法改善、反而更糊。

低通：指标低于 baseline，且图像更糊，不适合运动去模糊。

高通：指标最低，输出为边缘提取结果，不适合去模糊，但可用于观察频域中高频（边缘）分布。 

巴特沃斯滤波：平滑过渡，距离越远：响应慢慢减小。

理想滤波 “一刀切”  小于截止频率，完全保留；大于截止频率，完全去掉。

HAP-Deblur：在 Richardson-Lucy 盲去模糊框架上，引入频谱方向估计、无参考自适应融合和锐度回退机制，并针对序列/视频做共享核与时域融合优化。

NRC-HybridNet：以 MIMO-UNet 为深度主干，设计 NR 置信度融合与频域残差注入后处理模块，在不增加训练成本的前提下提升鲁棒性与纹理保真度，并支持大图分块与视频时序平滑。

### 结论

传统空域/频域滤波不适合运动去模糊：它们针对噪声或频率成分，无法逆转方向性拖影，PSNR 均 ≤ baseline。

HAP-Deblur 思路正确但受限于估核精度：空频结合 + 自适应回退使结果稳定，但未超过 baseline，适合无 GPU 的可解释场景。

NRC-HybridNet 是唯一有效超越 baseline 的方案：PSNR +2.3 dB、SSIM +0.12，证明数据驱动的 DL 去模糊 + NR 后处理对 GoPro 运动模糊显著有效。

实验验证了「方法要与退化类型匹配」：运动模糊需要反卷积/学习型恢复，而非简单平滑或滤频。

## 6 运行命令

### HAP-Deblur
数据集评估（序列级共享核，推荐）：
```bash
D:\anaconda3\python.exe d:\work\sight\evaluate_wiener.py
```
批量图像（多进程）：
```bash
D:\anaconda3\python.exe d:\work\sight\process_media.py `
  --input d:\work\sight\GOPRO_Large\train\GOPR0372_07_00\blur_gamma `
  --output d:\work\sight\out_images `
  --workers 4
```
视频处理（首帧估核 + 时域融合）：
```bash
D:\anaconda3\python.exe d:\work\sight\process_media.py `
  --input input.mp4 --output output.mp4
```
启用细节增强阶段：
```bash
D:\anaconda3\python.exe d:\work\sight\evaluate_wiener.py --super-resolve
```

### NRC-HybridNet
数据集评估
```bash
D:\anaconda3\envs\yolov8\python.exe d:\work\sight\evaluate_dl_deblur.py
```
批量图像
```bash
D:\anaconda3\envs\yolov8\python.exe d:\work\sight\process_media.py `
  --method dl --input d:\work\sight\GOPRO_Large\train\GOPR0372_07_00\blur_gamma `
  --output d:\work\sight\out_dl
```

# 视频（限帧跑）
```bash
D:\anaconda3\envs\yolov8\python.exe d:\work\sight\process_media.py `
  --method dl --input d:\work\sight\1.mp4 --output d:\work\sight\output_dl.mp4 --max-frames 100
```