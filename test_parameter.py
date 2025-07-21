FOLDER_NAME = 'Test'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/trajectory'
length_path = f'results/length'
INPUT_DIM = 7
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors
USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1 # the number of GPUs
NUM_META_AGENT = 1  # the number of processes
NUM_TEST = 100
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs
SAVE_TRAJECTORY = True  # do you want to save per-step metrics
SAVE_LENGTH = True  # do you want to save per-episode metrics

# 新增详细指标收集配置
SAVE_DETAILED_METRICS = True  # 是否保存详细指标数据
METRICS_DIR = 'results/detailed_metrics'  # 详细指标保存目录
ROBOT_RADIUS = 10  # 机器人半径（像素），用于计算覆盖面积
COLLISION_COOLDOWN = 0.1  # 碰撞检测冷却时间（秒）
PRINT_STEP_METRICS = False  # 是否打印每步的指标（调试用）