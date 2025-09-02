import numpy as np

def load_and_print_npy(file_path):
    data = np.load(file_path, allow_pickle=True)
    
    print(f"Raw data type: {type(data)}")
    print(f"Raw data shape: {np.shape(data)}")

    # 试着转换成 Python 对象
    try:
        data_dict = data.item()
        print("\nLoaded as Python dict-like object:")
    except Exception as e:
        print("\nCould not convert to dict:", e)
        return

    # 遍历所有键
    for key, value in data_dict.items():
        print(f"\nKey: {key}")
        print(f"  Type: {type(value)}")
        print(f"  Shape: {np.shape(value)}")
        if isinstance(value, np.ndarray):
            print(f"  Sample content: {value[:3]}")
        else:
            print(f"  Content: {value}")

    # 特别打印 'grasp_qpos' 的 shape
    if 'grasp_qpos' in data_dict:
        qpos = data_dict['grasp_qpos']
        print("\n✅ 'grasp_qpos' found!")
        print(f"   Shape of 'grasp_qpos': {np.shape(qpos)}")
    else:
        print("\n⚠️  'grasp_qpos' key not found in this file.")

def main():
    file_path = '/home/rose/legograsp/flex/results/6d/grasp_poses.npy'
    load_and_print_npy(file_path)

if __name__ == "__main__":
    main()
