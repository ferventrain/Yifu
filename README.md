## utils
* channel_organizer.py: 整理不同通道图像到对应文件夹
    ```
    python channel_organizer.py
    ```
    然后输入包含Ch0，Ch1，Ch2的文件夹路径

* downsample.py: 下采样图像
    ```
    python downsample.py --input_folder INPUT_FOLDER --resolution_config JSON_PATH --method linear
    ```
    method: 下采样方法，可选linear, nearest, quadratic, cubic
    chunk_size: 每次处理的切片数量，根据内存情况调整
    INPUT_FOLDER: 输入文件夹，包含TIFF文件，不可以包含中文
    文件夹结构：
    SAMPLE_ID/
    ├── Ch0/(INPUT_FOLDER)
    ├── Ch1/
    ├── Ch2/
    ├── original_shape.json
    ├── Ch0_downsampled/

## Registration

* ANTs_registration.py
    使用ants包进行配准
    ```
    python ANTs_registration.py --target Ch0_downsampled/ --atlas_image atlas.tiff --atlas_label atlas_label.tiff --output_dir Ch0_downsampled_reg/
    ```
    mode: 配准方向，可选atlas2image, image2atlas，默认atlas2image
    registration_type: 配准类型，可选Rigid, Affine, SyN, SyNRA，默认SyN
    upsample_method: 上采样插值方法，可选nearest, linear, cubic, quintic，默认nearest
    chunk_size: 每次处理的切片数量，根据内存情况调整
    save_transforms: 是否保存变换文件
    文件夹结构：
    SAMPLE_ID/
    ├── Ch0/
    ├── Ch0_downsampled/
    ├── Ch0_downsampled_reg/
    ├── original_shape.json