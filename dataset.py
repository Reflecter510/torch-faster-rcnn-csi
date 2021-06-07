# 三种数据集相关的配置

# S1数据集（注：S1所有数据）
class S1P1():
    # 数据集名称
    name = "192S1ALL"
    # CSI尺寸
    image_shape = [90,192,1]
    # 动作类别数
    num_classes = 12
    # 锚框设置
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 108
    test_batch = 215
    # 使用kaggle进行训练时的数据集目录
    kaggle_dir = "../input/mydata/S1"
    # 动作的名称列表
    actions = ['none', 'jump', 'pick', 'throw', 'pull', 'clap', 'box', 'wave', 'lift', 'kick', 'squat', 'turnRound', 'checkWatch']
    

class TEMPORAL():
    # 数据集名称
    name = "TEMPORAL"
    # CSI尺寸
    image_shape = [52,192,1]
    # 动作类别数
    num_classes = 6
    # 锚框设置
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 36
    test_batch = 278
    # 使用kaggle进行训练时的数据集目录
    kaggle_dir = "../input/my"
    # 动作的名称列表
    actions = ['none', "hand_up", "hand_down", "hand_left", "hand_right", "hand_circle", "hand_cross"]

class S2():
    # 数据集名称
    name = "192S2"
    # CSI尺寸
    image_shape = [90,192,1]
    # 动作类别数
    num_classes = 12
    # 锚框设置
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 48
    test_batch = 72
    # 使用kaggle进行训练时的数据集目录
    kaggle_dir = "../input/mydata/S2"
    # 动作的名称列表
    actions = ['none', 'jump', 'pick', 'throw', 'pull', 'clap', 'box', 'wave', 'lift', 'kick', 'squat', 'turnRound', 'checkWatch']
    