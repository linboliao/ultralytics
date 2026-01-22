import argparse
import json
import os
import stat
import shutil
import subprocess
import time
import traceback
import concurrent.futures
import pandas as pd
import openslide
from typing import Dict, List, Any, Optional
from collections import defaultdict

conda_path = r'C:\Users\MXZY-AI\.conda\envs'

work_dir = {
    'prepath': r'D:\Users\MXZY-AI\PycharmProjects\PrePATH',
    'ultralytics': r'D:\Users\MXZY-AI\PycharmProjects\ultralytics',
    'mil': r'D:\Users\MXZY-AI\PycharmProjects\MIL_BASELINE'
}


def run_command(p, command, task_name):
    """执行命令行任务并处理异常"""
    try:
        print(f'开始执行{task_name}任务')
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{p}"
        print(command)
        result = subprocess.run(
            command,
            cwd=p,
            env=env,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=720000
        )
        print(f"✅ [{task_name}] 执行成功")
        print(f"输出摘要: {result.stdout[:100]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [{task_name}] 执行失败 (code={e.returncode})\n错误信息: {e.stderr}")
        traceback.print_exc()
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ [{task_name}] 执行超时")
        traceback.print_exc()
        return False
    except FileNotFoundError:
        print(f"🔍 [{task_name}] 文件未找到")
        traceback.print_exc()
        return False


def generate_cls_csv(csv_dir, coord_dir, wsi_dir, num):
    """生成CSV分割文件（含两列：case_id 和 slide_id），均匀分成num份"""
    csv_path = os.path.join(csv_dir, 'csv')
    os.makedirs(csv_path, exist_ok=True)
    slides = []
    for root, dirs, files in os.walk(wsi_dir):
        for file in files:
            if file.endswith('.svs'):
                slides.append(file)
    base_names = []
    for slide in slides:
        base = os.path.splitext(slide)[0]
        h5_path = os.path.join(coord_dir, 'patches', f'{base}.h5')
        if os.path.exists(h5_path) and os.path.getsize(h5_path) > 0:
            base_names.append(base)

    df = pd.DataFrame({"case_id": base_names, "slide_id": base_names})
    total_rows = len(df)

    part_size = total_rows // num
    remainder = total_rows % num

    csv_files = []
    start_index = 0

    for i in range(num):
        current_part_size = part_size + (1 if i < remainder else 0)
        end_index = start_index + current_part_size

        part_df = df.iloc[start_index:end_index]
        part_csv_path = os.path.join(csv_path, f'part_{i}.csv')
        part_df.to_csv(part_csv_path, index=False)

        csv_files.append(part_csv_path)
        start_index = end_index

    return csv_files


def get_patch_csv(args):
    wsi_format = {'svs', 'ndpi', 'tif', 'tiff', 'mrxs'}
    output_40 = os.path.join(args.coord_dir, "40x_slides.csv")
    output_20 = os.path.join(args.coord_dir, "20x_slides.csv")
    output_10 = os.path.join(args.coord_dir, "10x_slides.csv")

    slides_40 = []
    slides_20 = []
    slides_10 = []

    for root, dirs, filenames in os.walk(args.wsi_dir):
        for filename in filenames:
            postfix = filename.split(".")[-1].lower()
            if postfix in wsi_format:
                slide_path = os.path.join(root, filename)
                try:
                    with openslide.OpenSlide(slide_path) as wsi:
                        objective = wsi.properties.get('openslide.objective-power', '20')
                        objective_int = int(objective)
                        if objective_int == 40:
                            slides_40.append(slide_path)
                        elif objective_int == 20:
                            slides_20.append(slide_path)
                        elif objective_int == 10:
                            slides_10.append(slide_path)
                except Exception as e:
                    print(f"处理文件 {slide_path} 时出错: {e}")
                    continue

    pd.DataFrame({'slide_path': slides_40}).to_csv(output_40, index=False, encoding='utf-8')
    pd.DataFrame({'slide_path': slides_20}).to_csv(output_20, index=False, encoding='utf-8')
    pd.DataFrame({'slide_path': slides_10}).to_csv(output_10, index=False, encoding='utf-8')

    return (output_40, output_20, output_10), ('896', '448', '224')


def create_patch_cls(args):
    path = work_dir.get('prepath')
    coord_dir = os.path.join(args.output_dir, 'patches_cls')
    os.makedirs(coord_dir, exist_ok=True)
    args.coord_dir = coord_dir
    csvs, patch_sizes = get_patch_csv(args)
    for csv, patch_size in zip(csvs, patch_sizes):
        patch_cmd = [
            os.path.join(conda_path, "clam/python.exe"),
            os.path.join(path, 'create_patches_fp.py'),
            "--source", args.wsi_dir,
            "--csv_path", csv,
            "--save_dir", coord_dir,
            "--preset", "maixin.csv",
            "--patch_level", '0',
            "--patch_size", patch_size,
            "--step_size", patch_size,
            "--wsi_format", 'svs',
            "--seg", "--patch", "--stitch", "--use_mp"
        ]
        if not run_command(path, patch_cmd, f"WSI生成 0 {patch_size} coords"):
            return False


def extract_features(csv_path, path, args, conda_path, coord_dir):
    """处理单个CSV文件的函数，用于并行执行"""
    feat_dir = os.path.join(args.output_dir, 'feat_cls')
    task_name = f"WSI特征提取_{os.path.basename(csv_path)}"
    print(csv_path)

    scr = 'extract_features_fp_stains.py' if args.normal else 'extract_features_fp_fast.py'
    feat_cmd = [
        os.path.join(conda_path, "clam/python.exe"),
        os.path.join(path, scr),
        "--data_coors_dir", coord_dir,
        "--data_slide_dir", args.wsi_dir,
        "--slide_ext", '.svs',
        "--csv_path", csv_path,
        "--feat_dir", feat_dir.replace('\\', '/'),
        "--batch_size", '48',
        "--model", args.model,
    ]

    return run_command(path, feat_cmd, task_name)


def extract_features_parallel(args):
    prepath = work_dir.get('prepath')
    coord_dir = os.path.join(args.output_dir, 'patches_cls')

    csv_paths = generate_cls_csv(args.output_dir, coord_dir, args.wsi_dir, 2)
    csv_paths = [p for p in csv_paths if os.path.exists(p)]
    if not csv_paths: return False

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_to_csv = {executor.submit(extract_features, p, prepath, args, conda_path, coord_dir): p for p in
                         csv_paths}
        for future in concurrent.futures.as_completed(future_to_csv):
            try:
                future.result()
            except Exception as e:
                print(f"处理失败: {future_to_csv[future]}, 错误: {e}")
                return False
    return True


def generate_yolo_csv(csv_path, output_dir, num_parts=2):
    csv_output_dir = os.path.join(output_dir, 'csv/yolo')
    os.makedirs(csv_output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, dtype={'slide_id': str})

    if 'prediction' not in df.columns:
        raise ValueError(f"CSV文件 {csv_path} 中未找到'prediction'列")

    filtered_df = df[df['prediction'] == 1].copy()
    total_rows = len(filtered_df)

    if total_rows == 0:
        print("未找到prediction=1的行，无需分割")
        return []

    part_size = total_rows // num_parts
    remainder = total_rows % num_parts

    csv_files = []
    start_index = 0

    for i in range(num_parts):
        current_part_size = part_size + (1 if i < remainder else 0)
        end_index = start_index + current_part_size

        part_df = filtered_df.iloc[start_index:end_index]
        part_csv_path = os.path.join(csv_output_dir, f'part_{i}.csv')
        part_df.to_csv(part_csv_path, index=False)

        csv_files.append(part_csv_path)
        start_index = end_index

    return csv_files


def create_patch_yolo(args):
    path = work_dir.get('prepath')
    coord_dir = os.path.join(args.output_dir, 'patches_yolo')
    patch_cmd = [
        os.path.join(conda_path, "clam/python.exe"),
        os.path.join(path, 'create_patches_fp.py'),
        "--source", args.wsi_dir,
        "--save_dir", coord_dir,
        "--preset", "maixin.csv",
        "--patch_level", '0',
        "--patch_size", '2048',
        "--step_size", '2048',
        "--wsi_format", 'svs',
        "--seg", "--patch", "--stitch", "--use_mp"
    ]
    if not run_command(path, patch_cmd, "WSI生成 0 2048 coords"):
        return False


def run_yolo(args, csv_file):
    path = work_dir.get('ultralytics')
    coord_dir = os.path.join(args.output_dir, 'patches_yolo')
    yolo_cmd = [
        os.path.join(conda_path, "ultralytics/python.exe"),
        os.path.join(path, 'infer/yolo2x.py'),
        "--model", 'yolo',
        "--task", 'detect',
        "--data_coors_dir", coord_dir,
        "--data_slide_dir", args.wsi_dir,
        "--csv_path", csv_file,
        "--ckpts",
        'runs/detect/yolo11s_0512/weights/best.pt;runs/detect/yolo11s_0702/weights/best.pt;runs/detect/cbam/weights/best.pt;runs/detect/pki/weights/best.pt',
        "--slide_ext", '.svs',
        "--batch_size", '2',
        "--output_dir", os.path.join(args.output_dir, 'yolo'),
    ]
    return run_command(path, yolo_cmd, f"YOLO检测（{os.path.basename(csv_file)}）")


def run_yolo_parallel(args):
    """YOLO目标检测任务"""
    csvs = generate_yolo_csv(args.csv_path, args.output_dir, num_parts=3)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_yolo, args, csv) for csv in csvs]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                return False
    return True


def gen_test_csv(args):
    test_csv = os.path.join(args.output_dir, 'test.csv')
    if os.path.exists(test_csv):
        os.remove(test_csv)
    feat_dir = os.path.join(args.output_dir, f'feat_cls/pt_files/{args.model}')
    feat_files = [
        entry.path for entry in os.scandir(feat_dir)
        if entry.is_file() and os.path.getsize(entry.path) > 0
    ]
    df = pd.DataFrame({
        "test_slide_path": feat_files,
        "test_label": [0 for _ in range(len(feat_files))],
    })
    df.to_csv(test_csv, index=False)
    return test_csv


def merge_predictions_by_voting(output_dir, model_indices=[0, 1, 2, 3, 4]):
    slide_predictions = defaultdict(list)

    for idx in model_indices:
        result_dir = os.path.join(output_dir, f'cancer/{idx}')
        result_file = os.path.join(result_dir, 'Infer_Result_CLAM_MB_MIL.csv')

        if not os.path.exists(result_file):
            raise FileNotFoundError(f"模型{idx}的结果文件不存在: {result_file}")

        df = pd.read_csv(result_file, dtype={'slide_id': str})

        required_cols = ['slide_id', 'prediction']
        if not set(required_cols).issubset(df.columns):
            raise ValueError(f"结果文件{result_file}缺少必要列，需要: {required_cols}")

        for _, row in df.iterrows():
            slide_id = row['slide_id']
            prediction = row['prediction']
            slide_predictions[slide_id].append(int(prediction))

    merged_results = []
    for slide_id, preds in slide_predictions.items():
        count_0 = preds.count(0)
        count_1 = preds.count(1)

        final_pred = 1 if count_1 >= count_0 else 0
        merged_results.append({
            'slide_id': slide_id,
            'prediction': final_pred,
            'votes_0': count_0,  # 可选：保留票数统计
            'votes_1': count_1
        })

    merged_df = pd.DataFrame(merged_results)
    merged_df = merged_df[['slide_id', 'prediction', 'votes_0', 'votes_1']]  # 调整列顺序
    merged_df = merged_df.sort_values(by='slide_id').reset_index(drop=True)

    output_file = os.path.join(output_dir, 'cancer', 'merged_voting_result.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"投票合并结果已保存至: {output_file}")

    return merged_df


def run_cls(args):
    path = work_dir.get('mil')
    cancer_dir = os.path.join(args.output_dir, 'cancer/0')
    ckpt = '10x-normal' if args.normal else '10x'
    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, f'ckpts/cancer/{ckpt}/best_f1.pth'),
        "--test_log_dir", cancer_dir
    ]
    if not run_command(path, test_cmd, "癌症诊断"):
        return False
    cancer_dir = os.path.join(args.output_dir, 'cancer/1')
    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, f'ckpts/cancer/{ckpt}/best_f2.pth'),
        "--test_log_dir", cancer_dir
    ]
    if not run_command(path, test_cmd, "癌症诊断"):
        return False
    cancer_dir = os.path.join(args.output_dir, 'cancer/2')
    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, f'ckpts/cancer/{ckpt}/best_f3.pth'),
        "--test_log_dir", cancer_dir
    ]
    if not run_command(path, test_cmd, "癌症诊断"):
        return False
    cancer_dir = os.path.join(args.output_dir, 'cancer/3')
    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, f'ckpts/cancer/{ckpt}/best_f4.pth'),
        "--test_log_dir", cancer_dir
    ]
    if not run_command(path, test_cmd, "癌症诊断"):
        return False
    cancer_dir = os.path.join(args.output_dir, 'cancer/4')
    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, f'ckpts/cancer/{ckpt}/best_f5.pth'),
        "--test_log_dir", cancer_dir
    ]
    if not run_command(path, test_cmd, "癌症诊断"):
        return False
    merge_predictions_by_voting(args.output_dir)
    return True


def run_isup(args):
    path = work_dir.get('mil')

    isup_dir = os.path.join(args.output_dir, 'isup')

    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/isup/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, 'ckpts/isup/best.pth'),
        "--test_log_dir", isup_dir
    ]
    return run_command(path, test_cmd, "isup 诊断")


def run_gleason(args):
    path = work_dir.get('mil')

    isup_dir = os.path.join(args.output_dir, 'gleason')

    test_cmd = [
        os.path.join(conda_path, f'clam/python.exe'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/gleason/CLAM_MB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, 'ckpts/gleason/best.pth'),
        "--test_log_dir", isup_dir
    ]
    return run_command(path, test_cmd, "gleason 诊断")


grade_mapping = {
    0: "3+3",
    1: "3+4",
    2: "4+3",
    3: "4+4",
    4: "3+5",
    5: "4+5",
    6: "5+4",
    7: "5+5"
}
isup_mapping = {
    "3+3": 1,
    "3+4": 2,
    "4+3": 3,
    "4+4": 4,
    "3+5": 4,
    "5+3": 4,
    "4+5": 5,
    "5+4": 5,
    "5+5": 5

}


def execute_phase_parallel(tasks, task_names, args, max_workers=2):
    """并行执行阶段任务"""
    print(f"🚀 开始并行执行 {len(tasks)} 个任务: {', '.join(task_names)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到执行器
        future_to_task = {executor.submit(task, args): name for task, name in zip(tasks, task_names)}

        results = {}
        # 等待所有任务完成并收集结果
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                results[task_name] = result
                print(f"✅ [{task_name}] 执行完成")
            except Exception as e:
                results[task_name] = False
                print(f"❌ [{task_name}] 执行失败: {e}")

    return results


def run_medical_image_pipeline(wsi_dir: str, output_dir: str, slide_list=None, patch_level: int = 0,
                               wsi_format: str = "svs", model: str = "h-optimus-1", normal=False) -> Dict[str, Any]:
    args = argparse.Namespace()
    args.wsi_dir = wsi_dir
    args.slide_list = slide_list
    args.output_dir = output_dir
    args.patch_level = patch_level
    args.wsi_format = wsi_format
    args.model = model
    args.normal = normal

    if slide_list:
        slide_list_items = slide_list.split(';')
        tmp_dir = os.path.join(output_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        for slide in slide_list_items:
            shutil.copy(os.path.join(wsi_dir, slide), tmp_dir)
        args.wsi_dir = tmp_dir

    all_results = {}
    execution_stats = {}
    st = time.time()
    try:
        create_patch_cls(args)
        extract_features_parallel(args)

        args.test_csv = gen_test_csv(args)

        st_phase1 = time.time()
        phase1_tasks = [run_cls, run_gleason, create_patch_yolo]
        phase1_names = ["癌症诊断", "Gleason诊断", "yolo patching"]

        phase1_results = execute_phase_parallel(phase1_tasks, phase1_names, args, max_workers=3)
        all_results.update(phase1_results)

        phase1_time = time.time() - st_phase1
        execution_stats["phase1_time"] = phase1_time
        print(f"⏱️ 阶段1执行时间: {phase1_time:.2f}秒")
        args.csv_path = os.path.join(args.output_dir, 'cancer/merged_voting_result.csv')
        run_yolo_parallel(args)
        total_time = time.time() - st
        print(f"⏱️ 总执行时间: {total_time:.2f}秒")

        if all_results.get("癌症诊断", True):
            _generate_result_files(output_dir)

        return {
            "success": True,
            "results": all_results,
            "statistics": execution_stats,
            "output_dir": output_dir
        }

    except Exception as e:
        print(f"❌ 流水线执行失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "statistics": execution_stats
        }


def _generate_result_files(output_dir: str) -> None:
    """生成最终的结果文件"""
    result_json = os.path.join(output_dir, 'exist_cancer.json')
    cancer_csv = os.path.join(output_dir, 'cancer/merged_voting_result.csv')
    tissue_csv = os.path.join(output_dir, 'yolo/area.csv')
    gleason_csv = os.path.join(output_dir, 'gleason/Infer_Result_CLAM_MB_MIL.csv')
    # isup_csv = os.path.join(output_dir, 'isup/Infer_Result_CLAM_MB_MIL.csv')

    results = []

    cancer_df = pd.read_csv(cancer_csv, dtype={'slide_id': str})
    tissue_df = pd.read_csv(tissue_csv, dtype={'slide_id': str})
    gleason_df = pd.read_csv(gleason_csv, dtype={'slide_id': str})
    # isup_df = pd.read_csv(isup_csv, dtype={'slide_id': str})
    import geopandas as gpd
    for slide_id, pred, votes_0 in zip(cancer_df['slide_id'], cancer_df['prediction'], cancer_df['votes_0']):
        file_path = os.path.join(output_dir, 'yolo', f'{slide_id}-detect.geojson')
        num = 0
        if os.path.exists(file_path):
            gdf = gpd.read_file(file_path)
            gdf = gdf[
                gdf['classification'].apply(
                    lambda x: json.loads(x.replace("'", '"')).get('name') == 'Malignant'
                )
            ]
            gdf.to_file(os.path.join(output_dir, f'{slide_id}.geojson'))
            num = len(gdf)
        if pred == 0 or num == 0:
            typ = 'Benign'
        else:
            typ = 'Malignant'

        conf = 'weak' if votes_0 in [2, 3] else 'strong'
        tissue = tissue_df[tissue_df['slide_id'].astype(str) == str(slide_id)]
        gleason = gleason_df[gleason_df['slide_id'].astype(str) == str(slide_id)]
        # isup = isup_df[isup_df['slide_id'].astype(str) == str(slide_id)]

        tissue_area = tissue['area'].iloc[0] if not tissue.empty else "N/A"
        gleason_grade = grade_mapping[gleason['prediction'].iloc[0]] if not gleason.empty else 'N/A'
        isup_grade = isup_mapping[gleason_grade] if gleason_grade != 'N/A' else "N/A"

        result = {
            "filename": f'{slide_id}.geojson',
            "type": typ,
            "conf": conf,
            "percentage": tissue_area,
            "Gleason": f"{gleason_grade}",
            "ISUP": f"{isup_grade}"
        }
        results.append(result)

    try:
        with open(result_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"geojson_files": []}

    for new_result in results:
        found = False
        for i, existing_result in enumerate(data['geojson_files']):
            if existing_result['filename'] == new_result['filename']:
                # 更新已存在的记录
                data['geojson_files'][i] = new_result
                found = True
                break

        if not found:
            data['geojson_files'].append(new_result)

    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


parser = argparse.ArgumentParser(description="医学图像处理流水线 v1.0")

# 路径参数组
path_group = parser.add_argument_group("路径配置")
path_group.add_argument("--wsi_dir", type=str, default=r"D:/MXZY-AI/UI_withAI[v1.1]/input", help="WSI图像目录")
path_group.add_argument("--slide_list", type=str, help="slide 列表，使用;分隔")
path_group.add_argument("--output_dir", default=r"D:/MXZY-AI/UI_withAI[v1.1]/input/test", help="合并结果输出目录")

# WSI参数组
wsi_group = parser.add_argument_group("WSI处理参数")
wsi_group.add_argument("--patch_level", type=int, default=0, help="提取层级")
wsi_group.add_argument("--wsi_format", default="svs;kfb", help="slide 格式，使用;分隔")
wsi_group.add_argument("--model", default="h-optimus-1", help="基础模型")

wsi_group.add_argument("--normal", type=bool, default=True, help="归一化")

# 任务控制组
control_group = parser.add_argument_group("任务控制")
control_group.add_argument("-j", "--jobs", type=int, default=-1, help="并行任务数（-1=自动使用所有核心）")


def handle_remove_readonly(func, path, exc_info):
    """处理只读文件的删除回调"""
    os.chmod(path, stat.S_IWRITE)  # 去除只读属性
    func(path)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        print(args)
        if args.slide_list:
            slide_list_items = args.slide_list.split(';')
            tmp_dir = os.path.join(args.output_dir, 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            for slide in slide_list_items:
                shutil.copy(os.path.join(args.wsi_dir, slide), tmp_dir)
            args.wsi_dir = tmp_dir

        all_results = {}
        execution_stats = {}
        st = time.time()
        create_patch_cls(args)
        extract_features_parallel(args)

        args.test_csv = gen_test_csv(args)

        st_phase1 = time.time()
        phase1_tasks = [run_cls, run_gleason, create_patch_yolo]
        phase1_names = ["癌症诊断", "Gleason诊断", "yolo patching"]

        phase1_results = execute_phase_parallel(phase1_tasks, phase1_names, args, max_workers=3)
        all_results.update(phase1_results)

        phase1_time = time.time() - st_phase1
        execution_stats["phase1_time"] = phase1_time
        print(f"⏱️ 阶段1执行时间: {phase1_time:.2f}秒")
        cancer_csv = os.path.join(args.output_dir, 'cancer/merged_voting_result.csv')
        args.csv_path = os.path.join(args.output_dir, 'cancer/merged_voting_result.csv')
        run_yolo_parallel(args)
        total_time = time.time() - st
        print(f"⏱️ 总执行时间: {total_time:.2f}秒")

        if all_results.get("癌症诊断", True):
            _generate_result_files(args.output_dir)

    finally:
        pass
        # shutil.rmtree(tmp_dir, onerror=handle_remove_readonly)
        # print(f"\n已成功删除文件夹: {tmp_dir}")
