import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
import concurrent.futures
import pandas as pd
from joblib import Parallel, delayed

# conda_path = 'C:/Users/MXZY-AI/.conda/envs'
#
# work_dir = {
#     'prepath': r'D:\Users\MXZY-AI\PycharmProjects\PrePATH',
#     'ultralytics': r'D:\Users\MXZY-AI\PycharmProjects\ultralytics',
#     'mil': r'D:\Users\MXZY-AI\PycharmProjects\MIL_BASELINE'
# }

conda_path = '/home/lbliao/anaconda3/envs'

work_dir = {
    'prepath': r'/data2/lbliao/Code/PrePATH',
    'ultralytics': r'/NAS2/Data1/lbliao/Code/ultralytics',
    'mil': r'/data2/lbliao/Code/MIL_BASELINE'
}


def run_command(p, command, task_name):
    """执行命令行任务并处理异常"""
    try:
        print(f'开始执行{task_name}任务')
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{p}"
        env['LD_LIBRARY_PATH'] = '/home/lbliao/anaconda3/envs/ultralytics/lib'
        result = subprocess.run(
            command,
            cwd=p,
            env=env,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=3600  # 1小时超时
        )
        print(f"✅ [{task_name}] 执行成功")
        print(f"输出摘要: {result.stdout[:20]}...")
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


def generate_csv_files(csv_dir, coord_dir, wsi_dir):
    """生成CSV分割文件（含两列：case_id 和 slide_id）"""
    csv_path = os.path.join(csv_dir, 'csv')
    os.makedirs(csv_path, exist_ok=True)

    base_names = []
    for slide in os.listdir(wsi_dir):
        base_name = os.path.splitext(slide)[0]
        coord_path = os.path.join(coord_dir, 'patches', f'{base_name}.h5')
        if os.path.exists(coord_path):
            base_names.append(base_name)

    df = pd.DataFrame({
        "case_id": base_names,
        "slide_id": base_names
    })
    csv_path = f"{csv_path}/part.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8', errors='replace')

    return csv_path


def run_wsi_task(args):
    """WSI处理流水线（补丁生成+特征提取）"""
    path = work_dir.get('prepath')
    coord_dir = os.path.join(args.output_dir, 'patches_0_224')
    patch_cmd = [
        f"{conda_path}/clam/bin/python",
        os.path.join(path, 'create_patches_fp.py'),
        "--source", args.wsi_dir,
        "--save_dir", coord_dir,
        "--preset", "maixin.csv",
        "--patch_level", '0',
        "--patch_size", '224',
        "--step_size", '224',
        "--wsi_format", 'svs;kfb',
        "--seg", "--patch", "--stitch", "--use_mp"
    ]
    if not run_command(path, patch_cmd, "WSI生成coords"):
        return False

    tmp_csv = generate_csv_files(args.output_dir, coord_dir, args.wsi_dir)
    feat_dir = os.path.join(args.output_dir, 'feat_0_224')
    feat_cmd = [
        f"{conda_path}/clam/bin/python",
        os.path.join(path, 'extract_features_fp_fast.py'),
        "--data_coors_dir", coord_dir,
        "--data_slide_dir", args.wsi_dir,
        "--slide_ext", '.svs;.kfb',
        "--csv_path", tmp_csv,
        "--feat_dir", feat_dir,
        "--batch_size", '128',
        "--model", args.model,
    ]
    return run_command(path, feat_cmd, "WSI特征提取")


def run_yolo(args):
    """YOLO目标检测任务"""
    path = work_dir.get('prepath')
    coord_dir = os.path.join(args.output_dir, f'patches_0_1024')
    patch_cmd = [
        f"{conda_path}/clam/bin/python",
        os.path.join(path, 'create_patches_fp.py'),
        "--source", args.wsi_dir,
        "--save_dir", coord_dir,
        "--preset", "maixin.csv",
        "--patch_level", '0',
        "--patch_size", '1024',
        "--step_size", '1024',
        "--wsi_format", 'svs;kfb',
        "--seg", "--patch", "--stitch", "--use_mp"
    ]
    if not run_command(path, patch_cmd, "WSI生成coords"):
        return False

    path = work_dir.get('ultralytics')
    yolo_cmd = [
        f"{conda_path}/ultralytics/bin/python",
        os.path.join(path, 'infer/yolo2x.py'),
        "--model", 'yolo',
        "--task", 'segment',
        "--data_coors_dir", coord_dir,
        "--data_slide_dir", args.wsi_dir,
        "--ckpts", 'runs/segment/yolo12n/weights/best.pt',
        "--slide_ext", '.kfb;.svs',
        "--batch_size", '32',
        "--output_dir", os.path.join(args.output_dir, 'yolo'),
    ]
    return run_command(path, yolo_cmd, "YOLO检测")


def gen_test_csv(args):
    test_csv = os.path.join(args.output_dir, 'test.csv')
    feat_dir = os.path.join(args.output_dir, f'feat_0_224/pt_files/{args.model}')
    feat_files = [entry.path for entry in os.scandir(feat_dir)]
    # feat_files = [os.path.join(feat_dir, f) for f in os.listdir(feat_dir)]
    df = pd.DataFrame({
        "test_slide_path": feat_files,
        "test_label": [0 for _ in range(len(feat_files))],
    })
    df.to_csv(test_csv, index=False)
    return test_csv


def run_cls(args):
    path = work_dir.get('mil')

    cancer_dir = os.path.join(args.output_dir, 'cancer')

    test_cmd = [
        os.path.join(conda_path, f'clam/bin/python'),
        os.path.join(path, 'infer_mil.py'),
        "--yaml_path", os.path.join(path, f'configs/cancer/AB_MIL-{args.model}.yaml'),
        "--test_dataset_csv", args.test_csv,
        "--model_weight_path", os.path.join(path, 'ckpts/cancer/best.pth'),
        "--test_log_dir", cancer_dir
    ]
    return run_command(path, test_cmd, "癌症诊断")


def run_isup(args):
    path = work_dir.get('mil')

    isup_dir = os.path.join(args.output_dir, 'isup')

    test_cmd = [
        os.path.join(conda_path, f'clam/bin/python'),
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
        os.path.join(conda_path, f'clam/bin/python'),
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


def execute_phase_parallel(tasks, task_names, max_workers=2):
    """并行执行阶段任务"""
    print(f"🚀 开始并行执行 {len(tasks)} 个任务: {', '.join(task_names)}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到执行器
        future_to_task = {executor.submit(task,args): name for task, name in zip(tasks, task_names)}

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


parser = argparse.ArgumentParser(description="医学图像处理流水线 v1.0")

# 路径参数组
path_group = parser.add_argument_group("路径配置")
path_group.add_argument("--wsi_dir", type=str, required=True, help="WSI图像目录")
path_group.add_argument("--slide_list", type=str, help="slide 列表，使用;分隔")
path_group.add_argument("--output_dir", required=True, help="合并结果输出目录")

# WSI参数组
wsi_group = parser.add_argument_group("WSI处理参数")
wsi_group.add_argument("--patch_level", type=int, default=0, help="提取层级")
wsi_group.add_argument("--wsi_format", default="svs;kfb", help="slide 格式，使用;分隔")
wsi_group.add_argument("--model", default="h-optimus-1", help="基础模型")

# 任务控制组
control_group = parser.add_argument_group("任务控制")
control_group.add_argument("-j", "--jobs", type=int, default=-1, help="并行任务数（-1=自动使用所有核心）")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.slide_list:
        slide_list = args.slide_list.split(';')
        tmp_dir = os.path.join(args.output_dir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        for slide in slide_list:
            shutil.copy(os.path.join(args.wsi_dir, slide), tmp_dir)
        args.wsi_dir = tmp_dir
    # all_results = {}
    # st = time.time()
    # run_wsi_task(args)
    # print(time.time() - st)
    # st = time.time()
    # run_yolo(args)
    # print(time.time() - st)
    # st = time.time()
    # args.test_csv = gen_test_csv(args)
    # run_cls(args)
    # print(time.time() - st)
    # run_isup(args)
    # run_gleason(args)

    all_results = {}
    st = time.time()

    # 阶段1：并行执行 run_wsi_task 和 run_yolo
    phase1_tasks = [run_wsi_task, run_yolo]
    phase1_names = ["特征提取", "YOLO检测"]

    phase1_results = execute_phase_parallel(phase1_tasks, phase1_names, max_workers=2)
    all_results.update(phase1_results)

    phase1_time = time.time() - st
    print(f"⏱️ 阶段1执行时间: {phase1_time:.2f}秒")

    # 生成测试CSV（必须在阶段1完成后执行）
    st = time.time()
    args.test_csv = gen_test_csv(args)
    csv_gen_time = time.time() - st
    print(f"⏱️ CSV生成时间: {csv_gen_time:.2f}秒")

    # 阶段2：并行执行 run_cls, run_isup, run_gleason
    st = time.time()
    phase2_tasks = [run_cls, run_isup, run_gleason]
    phase2_names = ["癌症诊断", "ISUP诊断", "Gleason诊断"]

    phase2_results = execute_phase_parallel(phase2_tasks, phase2_names, max_workers=3)
    all_results.update(phase2_results)

    phase2_time = time.time() - st
    print(f"⏱️ 阶段2执行时间: {phase2_time:.2f}秒")

    # 总执行时间
    total_time = phase1_time + csv_gen_time + phase2_time
    print(f"⏱️ 总执行时间: {total_time:.2f}秒")

    if all_results.get("癌症诊断", True) and all_results.get("癌症诊断", True) and all_results.get("癌症诊断", True):
        result_json = os.path.join(args.output_dir, 'exist_cancer.json')
        cancer_csv = os.path.join(args.output_dir, 'cancer/Infer_Result_AB_MIL.csv')
        tissue_csv = os.path.join(args.output_dir, 'yolo/area.csv')
        gleason_csv = os.path.join(args.output_dir, 'gleason/Infer_Result_CLAM_MB_MIL.csv')
        isup_csv = os.path.join(args.output_dir, 'isup/Infer_Result_CLAM_MB_MIL.csv')

        results = []

        cancer_df = pd.read_csv(cancer_csv)
        tissue_df = pd.read_csv(tissue_csv)
        gleason_df = pd.read_csv(gleason_csv)
        isup_df = pd.read_csv(isup_csv)

        for slide_id, pred in zip(cancer_df['slide_id'], cancer_df['prediction']):
            if pred == 1:
                tissue = tissue_df[tissue_df['slide_id'] == slide_id]
                gleason = gleason_df[gleason_df['slide_id'] == slide_id]
                isup = isup_df[isup_df['slide_id'] == slide_id]

                result = {
                    "filename": f'{slide_id}.geojson',
                    "percentage": tissue['area'].iloc[0] if not tissue.empty else "N/A",
                    "Gleason": f"{gleason['prediction'].iloc[0] if not gleason.empty else 'N/A'};ISUP: {isup['prediction'].iloc[0] if not isup.empty else 'N/A'}"
                }
                results.append(result)

        try:
            with open(result_json, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"geojson_files": []}

        data['geojson_files'].extend(results)

        with open(result_json, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


