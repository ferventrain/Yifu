# Agent 运行规则（Yifu 仓库）

## 宗旨

1. **少命令**：运行已有流程只用标准入口，先 preflight、再提交一次；不写 run.py / watchdog 之类的临时脚本，不拼新命令，不改参数名和输出路径。
2. **不猜**：完成与失败只认任务状态和结构化错误（`error.code / step_id / retryable / suggestion`），不靠日志文字或退出码判断。
3. **少改**：一次普通失败不引发改代码；只有结构化错误明确指向配置或 harness 时，才做一次针对性修复。
4. **新代码先确认归属**：任务确实需要新代码时，先问用户——写进现有仓库，还是放到项目/数据文件夹下用完即删？
5. **少说话**：执行前只回一条将执行的命令，提交后回 run_id，不长篇解释，不逐步请求确认。

## 参数基线（cfos）

cfos 任务从 `config/config_cfos_template.json` 生成 config，只改每样本必查项：
`project_name`、`input.resolution_xyz`（从 ims 元数据取真实体素大小，MegaSpim 恒为 1.8/1.8/2.0）、
`input.channels`（cfos 惯例 signal=ch1/registration=ch0，非标准排布先核对）。其余参数保持 template
默认，要动阈值、模型、postprocess 等先问用户。全脑 normalization（normalize_scope=global）、
256³ 推理块、hemisphere 半球统计、postprocess（max_single_slice_voxels=300）是基线，不调。
血管任务不套 cfos template（等用户的血管 template 固化）。

## 组织选区（tissue ROI）

术语"组织选区"= 每个样本的组织包络 mask，固定存 `<sample>/tissue_roi.zarr`（样本级唯一名，不随通道变）。
一句话打开样本标注（自动找 QC 卷，画框→Segment All Slices→可选插值→Save）：

```bash
python -m pipeline_modules.visualization.annotate_tissue_sam_napari --sample-dir <sample_dir>
```

QC 卷不存在时错误里会给出 `surface_brightness_homogenize --qc_only` 的生成命令；消费用
`--tissue_mask <sample>/tissue_roi.zarr`。入口详情见 `pipeline_modules/visualization/capability_manifest.json`。

## 标准入口

```bash
python -m pipeline_modules.harness preflight --sample-dir "S:/path/to/sample"              # 只读校验
python -m pipeline_modules.harness run --sample-dir "S:/path/to/sample" \
  --config "S:/path/to/sample/config.json"                                                 # 提交，返回 run_id
python -m pipeline_modules.harness cancel <run_id>                                          # 取消
```

提交后在 Pipeline Monitor（http://127.0.0.1:8766，只读）查看。状态语义与错误字段见
`pipeline_modules/harness/capability_manifest.json`。拿不准时把结构化错误和日志尾部报告给用户，等指示。
