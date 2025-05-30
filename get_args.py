import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    # Neural Network setting
    parser.add_argument('-cout', type=int, default  = 24)
    parser.add_argument('-cfeat', type=int, default  = 256)

    # The relay channel
    parser.add_argument('-num_hops', default = 3)
    parser.add_argument('-relay_mode', default  = 'AF')

    parser.add_argument('-channel_mode', default = 'awgn')
    parser.add_argument('-fading', default = False)
    parser.add_argument('-link_qual', default  = 8)
    parser.add_argument('-link_qual2', default  = 8)
    parser.add_argument('-precode', default = False)
    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-link_rng', default  = 5)

    # add the successive refinement setting
    parser.add_argument('-L', default = 2)
    parser.add_argument('-sel_L', default = 2)

    # compressor setting
    parser.add_argument('-lamda', type=int, default  = 6400)
    parser.add_argument('-btneck_feat', type=int, default  = 256)
    parser.add_argument('-btneck_sz', type=int, default  = 192)

    # training setting
    parser.add_argument('-epoch', type=int, default  = 400)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 45)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-comm_iter', type=int, default  = 1)
    parser.add_argument('-compress_iter', type=int, default  = 2)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args