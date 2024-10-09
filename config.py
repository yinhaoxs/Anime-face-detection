cfg_nano = {
    'name': 'nano',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96, 128]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1.5,
    'landm_weight': 0.5,
    'gpu_train': True,
    'batch_size': 1280,
    'ngpu': 1,
    'epoch': 50,
    'decay1': 30,
    'decay2': 40,
    'image_size': 150
}
