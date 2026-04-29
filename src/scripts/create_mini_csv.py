"""
从原始CSV中选取12张图片（6正常+6患癌），生成mini版CSV用于本地测试。

仅依赖: torch + Python标准库

用法:
    python create_mini_csv.py

输出:
    G:\data\train_with_test_data_mini.csv
"""

import csv
import random
import torch


def main():
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    random.seed(42)

    # 读取原始CSV
    csv_path = r"G:\data\train_with_test_data.csv"
    output_path = r"G:\data\train_with_test_data_mini.csv"

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        all_rows = list(reader)

    print(f"原始数据总量: {len(all_rows)}")
    print(f"列名: {headers}")

    # 按 (patient_id, image_id) 去重
    seen = set()
    unique_rows = []
    for row in all_rows:
        key = (row['patient_id'], row['image_id'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    print(f"去重后数据量: {len(unique_rows)}")

    # 分离正常和患癌
    normal_rows = [r for r in unique_rows if r['cancer'] == '0']
    cancer_rows = [r for r in unique_rows if r['cancer'] == '1']

    print(f"正常样本（去重后）: {len(normal_rows)}")
    print(f"患癌样本（去重后）: {len(cancer_rows)}")

    # 使用torch随机打乱
    normal_indices = torch.randperm(len(normal_rows)).tolist()
    cancer_indices = torch.randperm(len(cancer_rows)).tolist()

    # 选取6正常 + 6患癌，尽量来自不同patient
    def select_diverse(rows, indices, count):
        selected = []
        seen_patients = set()
        # 第一轮：优先不同patient
        for idx in indices:
            if len(selected) >= count:
                break
            pid = rows[idx]['patient_id']
            if pid not in seen_patients:
                selected.append(idx)
                seen_patients.add(pid)
        # 第二轮：如果不够，补充
        for idx in indices:
            if len(selected) >= count:
                break
            if idx not in selected:
                selected.append(idx)
        return selected

    sel_normal = select_diverse(normal_rows, normal_indices, 6)
    sel_cancer = select_diverse(cancer_rows, cancer_indices, 6)

    selected_normal_rows = [normal_rows[i] for i in sel_normal]
    selected_cancer_rows = [cancer_rows[i] for i in sel_cancer]

    print(f"\n=== 选中正常样本 ({len(selected_normal_rows)} 张) ===")
    for r in selected_normal_rows:
        print(f"  patient_id={r['patient_id']}, image_id={r['image_id'][:50]}...")

    print(f"\n=== 选中患癌样本 ({len(selected_cancer_rows)} 张) ===")
    for r in selected_cancer_rows:
        print(f"  patient_id={r['patient_id']}, image_id={r['image_id'][:50]}...")

    # 划分 split：
    # 训练集: 5正常 + 5患癌 (共10张)
    # 测试集: 1正常 + 1患癌 (共2张)
    for r in selected_normal_rows:
        r['split'] = 'training'
    for r in selected_cancer_rows:
        r['split'] = 'training'

    # 各取第1个作为测试集
    selected_normal_rows[0]['split'] = 'test'
    selected_cancer_rows[0]['split'] = 'test'

    # 合并
    mini_rows = selected_normal_rows + selected_cancer_rows

    # 统计
    train_count = sum(1 for r in mini_rows if r['split'] == 'training')
    test_count = sum(1 for r in mini_rows if r['split'] == 'test')
    train_normal = sum(1 for r in mini_rows if r['split'] == 'training' and r['cancer'] == '0')
    train_cancer = sum(1 for r in mini_rows if r['split'] == 'training' and r['cancer'] == '1')
    test_normal = sum(1 for r in mini_rows if r['split'] == 'test' and r['cancer'] == '0')
    test_cancer = sum(1 for r in mini_rows if r['split'] == 'test' and r['cancer'] == '1')

    print(f"\n=== 划分结果 ===")
    print(f"总样本数: {len(mini_rows)}")
    print(f"训练集: {train_count} 张 (正常:{train_normal}, 患癌:{train_cancer})")
    print(f"测试集: {test_count} 张 (正常:{test_normal}, 患癌:{test_cancer})")

    # 保存
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mini_rows)

    print(f"\n已保存到: {output_path}")

    # 打印完整CSV内容预览
    print(f"\n=== CSV 内容预览 ===")
    print(','.join(headers))
    for r in mini_rows:
        print(f"{r['patient_id']},{r['image_id']},{r['split']},{r['cancer']},{r['cohert_num']}")

    print(f"\n完成！后续使用方式:")
    print(f"  python prepare_folds.py --csv_path \"{output_path}\" --output_path \"G:\\data\\train_with_test_mini_folds.csv\" --n_folds 5 --seed 42")
    print(f"  python run_finetune.py --csv_path \"{output_path}\" --data_dir \"G:\\data\" --n_folds 5")


if __name__ == "__main__":
    main()