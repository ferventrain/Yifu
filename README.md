## utils
* channel_organizer.py: 整理不同通道图像到对应文件夹
    ```
    python channel_organizer.py
    ```
    然后输入包含ch0，ch1，ch2的文件夹路径

* downsample.py: 下采样图像
    ```
    python downsample.py --input_folder INPUT_FOLDER --resolution_config JSON_PATH --method linear --downsample_mask
    ```
    method: 下采样方法，可选linear, nearest, quadratic, cubic
    chunk_size: 每次处理的切片数量，根据内存情况调整
    INPUT_FOLDER: 输入文件夹，包含TIFF文件，不可以包含中文
    downsample_mask: 是否下采样mask图像
    文件夹结构：
    ```
    SAMPLE_ID/
    ├── ch0/(INPUT_FOLDER)
    ├── ch1/
    ├── ch2/
    ├── ch0_mask/
    ├── original_shape.json
    ├── ch0_downsampled/
    ├── ch0_downsampled_mask/
    ```

## Registration

* ANTs_registration.py
    使用ants包进行配准
    ```
    python registration/ANTs_registration.py --mode image2atlas --target S:\tianzhenjun\nao_2/ch2_downsample --atlas_image Allen_brainatlas/atlas.nii.gz --atlas_label Allen_brainatlas/atlas_label.nii.gz --register_channel 2
    ```
    mode: 配准方向，可选atlas2image, image2atlas，默认atlas2image
    registration_type: 配准类型，可选Rigid, Affine, SyN, SyNRA，默认SyN
    register_channel: 用于配准的通道，默认0
    upsample_method: 上采样插值方法，可选nearest, linear, cubic, quintic，默认nearest
    chunk_size: 每次处理的切片数量，根据内存情况调整
    save_transforms: 是否保存变换文件
    文件夹结构：
    ```
    SAMPLE_ID/
    ├── ch0/
    ├── ch0_downsampled/
    ├── ch0_downsampled_reg/
    ├── original_shape.json
    ```