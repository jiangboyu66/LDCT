import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_mayo_ima_to_npy(input_dir: str,
                            output_dir: str = None,
                            dtype: str = 'int16'):
    """
    专为 Mayo Clinic LDCT Siemens .IMA 优化版
    自动计算最佳 header 大小
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    ima_files = sorted(input_path.glob("*.IMA")) + sorted(input_path.glob("*.ima"))
    print(f"找到 {len(ima_files)} 个 .IMA 文件\n")

    target_pixels = 512 * 512
    success = 0
    header_stats = []

    for ima_file in tqdm(ima_files, desc="转换进度"):
        try:
            raw = np.fromfile(ima_file, dtype=dtype)
            total_elements = len(raw)

            # ==================== 自动适配 header 大小 ====================
            # 尝试常见 header（4016, 4020, 4096 等）
            possible_headers = [4016, 4020, 4096, 0]
            best_header = 4016
            best_remaining = 0

            for h in possible_headers:
                remaining = total_elements - (h // 2)
                if target_pixels - 10 <= remaining <= target_pixels + 100:
                    best_header = h
                    best_remaining = remaining
                    break

            image_data = raw[best_header // 2:]
            n = len(image_data)

            if n >= target_pixels:
                data = image_data[:target_pixels].reshape(512, 512)
            else:
                # 补零
                data = np.zeros(target_pixels, dtype=dtype)
                data[:n] = image_data
                print(f"⚠️  {ima_file.name} | header={best_header} | 补零 {target_pixels - n} 个元素")

            npy_path = output_path / f"{ima_file.stem}.npy"
            np.save(npy_path, data)
            success += 1
            header_stats.append(best_header)

        except Exception as e:
            print(f"❌ 失败 {ima_file.name}: {e}")

    # 统计信息
    print(f"\n🎉 转换完成！成功 {success}/{len(ima_files)} 个文件")
    print(f"输出路径: {output_path}")
    if header_stats:
        most_common = max(set(header_stats), key=header_stats.count)
        print(f"最常用 header 大小: {most_common} 字节")


# ========================== 配置 ==========================
if __name__ == "__main__":
    # ==================== 修改这里 ====================
    input_folder = r"C:\Users\JIANG\Desktop\dataset2\3mm D45\QD_3mm_sharp\quarter_3mm_sharp\L506\quarter_3mm_sharp"  # 你的路径

    convert_mayo_ima_to_npy(
        input_dir=input_folder,
        output_dir=r"C:\Users\JIANG\Desktop\dataset2\3mm D45\QD_3mm_sharp\quarter_3mm_sharp\L506",  # None = 保存到原文件夹
        dtype='int16'
    )

# ========================== 配置区（只需修改这里） ==========================
