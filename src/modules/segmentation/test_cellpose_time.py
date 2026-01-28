from cellpose import models
import time
import numpy as np
import sys
import torch

def test_cellpose_performance():
    """
    Test Cellpose segmentation performance with different image sizes and configurations
    """
    print("Testing Cellpose performance...")

    # # Test different 2D image sizes first
    # image_sizes_2d = [(128, 128), (256, 256), (512, 512)]

    # # Ensure PyTorch uses the appropriate precision to avoid BFloat16 issues
    # torch.set_float32_matmul_precision('high')

    # # Set the default dtype to float32 to avoid BFloat16 precision issues
    # torch.set_default_dtype(torch.float32)

    # # Test 2D images
    # for size in image_sizes_2d:
    #     print(f"\nTesting with 2D image size: {size[0]}x{size[1]}")

    #     try:
    #         # Generate a more realistic test image with some structure
    #         img = np.random.rand(size[0], size[1], 3).astype(np.float32)

    #         # Add some simple structures that Cellpose might detect
    #         # Create a few bright spots that could resemble cells
    #         for _ in range(5):  # Add 5 potential "cells"
    #             center_x = np.random.randint(10, size[0]-10)
    #             center_y = np.random.randint(10, size[1]-10)
    #             radius = np.random.randint(5, 15)
    #             y, x = np.ogrid[:size[0], :size[1]]
    #             mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    #             img[mask] = 1.0  # Make these areas brighter

    #         # Load model (try both GPU and CPU)
    #         print("  Loading model...")
    #         try:
    #             # Set the device explicitly and ensure proper precision
    #             model = models.CellposeModel(gpu=True)
    #             device_used = "GPU"
    #         except Exception as e:
    #             print(f"  GPU not available: {e}, trying CPU...")
    #             try:
    #                 model = models.CellposeModel(gpu=False)
    #                 device_used = "CPU"
    #             except Exception as e2:
    #                 print(f"  Error loading model: {e2}")
    #                 continue

    #         print(f"  Using {device_used} for processing")

    #         # Record start time
    #         print(f"  Processing image...")
    #         start_time = time.time()

    #         # Run Cellpose evaluation with explicit precision handling
    #         # Handle the case where the function might return different number of values
    #         result = model.eval(
    #             img,
    #             diameter=30,
    #             channels=[0, 0])

    #         # Unpack results based on what's returned
    #         if len(result) >= 4:
    #             masks, flows, styles, diams = result
    #         elif len(result) == 3:
    #             masks, flows, styles = result
    #             diams = None
    #         else:
    #             print(f"  Unexpected number of return values: {len(result)}")
    #             continue

    #         # Record end time
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time

    #         print(f"  {device_used} Time: {elapsed_time:.2f} seconds")
    #         print(f"  Masks shape: {masks.shape if masks is not None else 'None'}")
    #         print(f"  Number of unique masks: {len(np.unique(masks)) if masks is not None else 0}")

    #     except KeyboardInterrupt:
    #         print("  Interrupted by user")
    #         break
    #     except Exception as e:
    #         print(f"  Error during processing: {e}")
    #         import traceback
    #         traceback.print_exc()

    # Test 3D image
    print(f"\nTesting with 3D image (z-stack)...")
    try:
        # Create a 3D image (z, y, x, channels)
        z_size, y_size, x_size = 256, 256, 256  # Reasonable 3D size that won't be too slow
        img_3d = np.random.rand(z_size, y_size, x_size, 1).astype(np.float32)

        # Add some structures in 3D
        for _ in range(5):  # Add 5 potential "cells" in 3D
            center_z = np.random.randint(5, z_size-5)
            center_y = np.random.randint(10, y_size-10)
            center_x = np.random.randint(10, x_size-10)
            radius = np.random.randint(3, 8)
            z, y, x = np.ogrid[:z_size, :y_size, :x_size]
            mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
            img_3d[mask] = 1.0  # Make these areas brighter

        print(f"  3D image shape: {img_3d.shape}")

        # Load model for 3D processing
        print("  Loading model for 3D processing...")
        try:
            model = models.CellposeModel(gpu=True)
            device_used = "GPU"
        except Exception as e:
            print(f"  GPU not available: {e}, trying CPU...")
            try:
                model = models.CellposeModel(gpu=False)
                device_used = "CPU"
            except Exception as e2:
                print(f"  Error loading model: {e2}")
                return

        print(f"  Using {device_used} for 3D processing")

        # Record start time
        print(f"  Processing 3D image...")
        start_time = time.time()

        # Run Cellpose evaluation for 3D image
        result = model.eval(
            img_3d,
            diameter=30,
            channel_axis=3,
            z_axis=0,
            do_3D=True,
            )  # Enable 3D processing

        # Unpack results based on what's returned
        if len(result) >= 4:
            masks, flows, styles, diams = result
        elif len(result) == 3:
            masks, flows, styles = result
            diams = None
        else:
            print(f"  Unexpected number of return values: {len(result)}")
            return
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"  {device_used} 3D Time: {elapsed_time:.2f} seconds")
        print(f"  3D Masks shape: {masks.shape if masks is not None else 'None'}")
        print(f"  Number of unique masks in 3D: {len(np.unique(masks)) if masks is not None else 0}")

    except KeyboardInterrupt:
        print("  3D processing interrupted by user")
    except Exception as e:
        print(f"  Error during 3D processing: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting completed.")

if __name__ == "__main__":
    test_cellpose_performance()