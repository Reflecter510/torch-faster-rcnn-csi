
class S1P1():
    name = "192S1ALL"
    image_shape = [90,192,1]
    num_classes = 12
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 108
    test_batch = 215
    kaggle_dir = "../input/mydata/S1"
    actions = ['none', 'jump', 'pick', 'throw', 'pull', 'clap', 'box', 'wave', 'lift', 'kick', 'squat', 'turnRound', 'checkWatch']
    

class TEMPORAL():
    name = "TEMPORAL"
    image_shape = [52,192,1]
    num_classes = 6
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 36
    test_batch = 278
    kaggle_dir = "../input/my"
    actions = ['none', "hand_up", "hand_down", "hand_left", "hand_right", "hand_circle", "hand_cross"]

class S2():
    name = "192S2"
    image_shape = [90,192,1]
    num_classes = 12
    anchor =  ((4*16,5*16,6*16,7*16,8*16,9*16,10*16),)
    train_batch = 288
    test_batch = 72
    kaggle_dir = "../input/mydata/S2"
    actions = ['none', 'jump', 'pick', 'throw', 'pull', 'clap', 'box', 'wave', 'lift', 'kick', 'squat', 'turnRound', 'checkWatch']
    