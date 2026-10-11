# Agent 运行规则（Yifu 仓库）

## 宗旨

1. **少命令**：运行已有流程只用标准入口，先 preflight、再提交一次；不写 run.py / watchdog 之类的临时脚本，不拼新命令，不改参数名和输出路径。
2. **不猜**：完成与失败只认任务状态和结构化错误（`error.code / step_id / retryable / suggestion`），不靠日志文字或退出码判断。
3. **少改**：一次普通失败不引发改代码；只有结构化错误明确指向配置或 harness 时，才做一次针对性修复。
4. **新代码先确认归属**：任务确实需要新代码时，先问用户——写进现有仓库，还是放到项目/数据文件夹下用完即删？
5. **少说话**：执行前只回一条将执行的命令，提交后回 run_id，不长篇解释，不逐步请求确认。

## cfos：config 与参数

**总原则：template 已定的参数一律不重复写。** run config 只写覆盖项。

1. **run config**：每 run 一个，放样本目录。`config.json` 是基底；补跑依次 `config_1.json`、`config_2.json`…（PM 下拉自动出现）。
   缺省键自动继承：`config_1 ← 同目录 config.json ← config/config_cfos_template.json`（深合并；list/标量整体覆盖）。
2. **首次建 config.json 必查三项**：`project_name`；`input.resolution_xyz`（从 ims 元数据取，MegaSpim 恒 1.8/1.8/2.0）；
   `input.channels`（cfos 惯例 signal=ch1 / registration=ch0，非标准排布先核对）。
3. **跳过环节**：顶层布尔 `"skip_registration": true`、`"only_tubule": true`、`"only_region_analysis": true`
   （等效命令行旗标，PM 一键开始可用；断点续跑靠完成标记自动跳过已完成步骤）。
4. **基线默认值全部定在 template 里，照用不调**；要动先问用户。当前基线要点：
   - 分割：cfos_unet + 全脑归一化（normalize_scope=global，1–99.5 百分位）+ 256³ 推理块 + rerun_if_model_updated。
   - 后处理：max_single_slice_voxels=300；edge_shrink_factor=0.99（脑表面收缩剔除，读注册产物 `<label>_ds_brain_mask.zarr` 缓存）。
   - 半球：save_upsampled_label_hemisphere_zarr=true（统计用 atlas_label_hemisphere.zarr）。
   - 配准：halo_mask 默认开；grad_step=0.15、syn_iterations=[70,50,30,10]；n4_preprocess 与 rigid_preflight_qc 默认开
     （产出 `<sample>/qc/registration_rigid_check.png`、`registration_final_check.png`，PM 卡片可点开）。
   - 已证伪不采用：CC 度量 + 大迭代档（AIND 对照实验，形变过度）。
5. 血管任务不套 cfos template（等血管 template 固化）。

## PM（Pipeline Monitor）

地址 http://127.0.0.1:8766。提交后在 PM 看板跟踪；拿不准时把结构化错误和日志尾部报告给用户。

**新写的程序必须让 PM 可读（三选一，从简到繁）**：
1. 用 **wrap** 包装（零改动，默认姿势）：`python -m pipeline_modules.harness wrap --sample-dir <样本> --title <标题> -- <任意命令>`。
   自动挂任务/日志/心跳/取消；程序 stdout 打 `PROGRESS <done>/<total> [phase]` 行就有进度条。
2. Python 程序内三行接入：`from pipeline_modules.harness.progress import report, report_done, report_failed`；
   `report(done=42, total=100, phase="写 ims", sample_dir=...)`。
3. 已在跑的外部进程用 attach（status 文件首行 `RUNNING/ALL DONE/FAILED/CANCELLED` + 第二行 started_at）。
   进度文件字段等完整契约见 `pipeline_modules/harness/capability_manifest.json` 的 progress_protocol。

**看板用法**：
1. 两级结构：项目（批=父目录）→ 样本卡；`_active` 下的批排前面，其余批展开排在后面（样式略暗）。
2. 批头部最右侧 P0-P3 设项目紧急度（存 `_active/pm_board.json`，P 高置顶，再点取消）；样本卡：config 下拉 +
   "开始/继续"（一键断点续跑）；"需求/复盘"笔记（存 `<样本>/pm_project.json`）；预计时长 chip；
   run 历史点击展开详情/QC 图/结果；"归档"删样本下 zarr/原始 ims/transforms（留 result*、visualization*、
   qc 和小文件）后把目录移入 `<root>/_archived`，并清掉该样本的任务记录与优先级。
3. 工具栏"环节耗时"：全部 run manifest 聚合的各模块中位/均值耗时（剔除 incomplete 与 >3 天离群）。

**不报错的要点**：
1. 提交/启动 worker 一律用 **micromamba yifu env 的 python**——worker 解释器跟随提交者，系统 python 会拉起无依赖
   worker，任务秒退 ModuleNotFoundError。
2. fresh shell 没设 `YIFU_ANALYSIS_ROOT` 时 harness 默认指向不存在的 `H:\`——命令前带
   `YIFU_ANALYSIS_ROOT=S:/Arivis_Analysis YIFU_ACTIVE_DIR=S:/Arivis_Analysis/_active`。
3. PM 启动（重启不杀任务；改后端需重启，改前端刷新 + Ctrl+F5）：
   `YIFU_ANALYSIS_ROOT=... micromamba run -n yifu python -m apps.pipeline_harness.main --port 8766`。
4. 刷新/GET 永不启动任务；开始、保存笔记、归档、取消才是显式操作。运行中 external 任务显"失联"= 进程死了或心跳断了。
5. 失败 run 的 manifest 会写 `status: incomplete`（含出错步骤），环节耗时统计自动排除。

## 组织选区（tissue ROI）

术语"组织选区"= 每个样本的组织包络 mask，固定存 `<sample>/tissue_roi.zarr`（样本级唯一名，不随通道变）。
一句话打开样本标注（自动找 QC 卷，画框→Segment All Slices→可选插值→Save）：

```bash
python -m pipeline_modules.visualization.annotate_tissue_sam_napari --sample-dir <sample_dir>
```

QC 卷不存在时错误里会给出 `surface_brightness_homogenize --qc_only` 的生成命令；消费用
`--tissue_mask <sample>/tissue_roi.zarr`。入口详情见 `pipeline_modules/visualization/capability_manifest.json`。

## 开发约定

改 main.py 流程、加新步骤或新模块接线后，必须跑一遍 stub 接线测试（不跑算法、几秒完成，
验证 6 步路径/键名/分支不断链），并让 `tests/test_main_stub_run.py` 的 EXPECTED_COMMANDS
覆盖新步骤的命令描述。手动跑法：在空样本目录放 config 后
`python main.py --config <cfg> --sample_dir <dir> --stub`，末行 `STUB_SUMMARY` 应列出全部命令。
新模块自身的算法测试照常进 `tests/`；失败测试不修就删或改，不留长期红测试。

## 标准入口

```bash
python -m pipeline_modules.harness preflight --sample-dir "S:/path/to/sample"              # 只读校验
python -m pipeline_modules.harness run --sample-dir "S:/path/to/sample" \
  --config "S:/path/to/sample/config.json"                                                 # 提交，返回 run_id
python -m pipeline_modules.harness cancel <run_id>                                          # 取消
```
