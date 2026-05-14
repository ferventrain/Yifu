# cFos Active Learning App

这个 app 用你的 `S:\Yifu\best_model.pt` 作为 3D cFos 分割模型，接到 MONAI Label 的 active learning 流程里。

## 支持内容

- 3D 推理：`NIfTI (.nii/.nii.gz)`、`TIFF (.tif/.tiff)`、`Zarr (.zarr)`
- 输出分割 mask
- 用 voxel-wise entropy 做 uncertainty scoring
- 用 `highest_entropy` 策略优先返回最值得标注的未标注样本

## 目录约定

给 `--studies` 一个 MONAI Label 本地数据目录，例如：

```text
datasets/
├── image/
│   ├── sample1.nii.gz
│   ├── sample2.tiff
│   └── sample3.zarr
└── label/
```

`label/` 可以一开始为空。MONAI Label 会把保存的预测或人工修正标签放进去。

## 安装依赖

在你现有环境里额外安装：

```bash
pip install monai monailabel nibabel
```

如果你想直接启动 server，通常还需要：

```bash
pip install uvicorn fastapi
```

## 启动

PowerShell:

```powershell
$env:CFOS_CHECKPOINT="S:\Yifu\best_model.pt"
$env:CFOS_DEVICE="cuda"
$env:CFOS_PATCH_SIZE="128,128,128"
$env:CFOS_INFER_BATCH_SIZE="2"
$env:CFOS_OUTPUT_DIR="S:\Yifu\apps\cfos_activelearning\output"

monailabel start_server `
  --app "S:\Yifu\apps\cfos_activelearning" `
  --studies "S:\Yifu\datasets" `
  --conf model_name cfos_unet
```

## 常用请求

先跑 uncertainty scoring：

```bash
POST /scoring/cfos_unet
```

拿下一个最值得标注的样本：

```bash
POST /activelearning/highest_entropy
```

直接推理某个样本：

```bash
POST /infer/cfos_unet
{
  "image": "sample1.nii.gz",
  "save_label": true,
  "label_tag": "original"
}
```

## 可调环境变量

- `CFOS_CHECKPOINT`: 模型权重路径
- `CFOS_DEVICE`: `cuda` / `cpu` / `auto`
- `CFOS_PATCH_SIZE`: 例如 `128,128,128`
- `CFOS_INFER_BATCH_SIZE`: sliding-window batch size
- `CFOS_OVERLAP`: patch overlap，默认 `0.25`
- `CFOS_THRESHOLD`: 前景阈值，默认 `0.5`
- `CFOS_OUTPUT_DIR`: 推理输出目录
- `CFOS_CACHE_DIR`: 预留缓存目录

## 说明

这个版本先实现的是 inference-driven active learning：

1. 用当前模型跑未标注样本
2. 计算平均 entropy
3. 优先把高 entropy 样本送去标注

如果你下一步想要，我可以继续把它补成：

- 自动把新标注样本写成训练 datalist
- 调你现有 `train_cfos_3d_mlflow.py` 做增量训练
- 每轮 active learning 自动更新 `best_model.pt`
