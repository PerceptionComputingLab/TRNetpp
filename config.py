class DefaultConfig (object) :

    exp_file_root = 'exp file path'
    train_dataset_root = 'train_dataset path'
    test_dataset_root = 'test_dataset path'

    cubic_sequence_length = 30
    cube_side_length = 25

    batch_size = 4
    max_epoch = 200
    learning_rate = 1e-6
    decay_LR = (20, 0.1)
    train_index = 1
    save_step = 10

    num_cross_fold = 10
    use_gpu = False

    in_channels = 1

    local_proj_shape = 2
    local_dim_hidden = (96, 128, 128)
    local_num_layers = (1, 2, 1)
    local_num_heads = (3, 4, 4)
    local_head_dim = 32
    local_patch_shape = 4
    local_switch_position = True,

    global_dim_seq = 1024
    global_num_heads = 4
    global_head_dim = 32
    global_num_encoders = 6

    CLS_num_linear = 3
    CLS_num_class = 2


opt  = DefaultConfig()