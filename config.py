import argparse


def imgcap_adaptive_argument_parser():

    parser = argparse.ArgumentParser(description='Train imgcaptioining arguments')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='flickr30k')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scale_min', type=float, default=0.9)
    parser.add_argument('--scale_max', type=float, default=1.1)
    parser.add_argument('--rotate_min', type=float, default=-10)
    parser.add_argument('--rotate_max', type=float, default=10)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=224,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=224,
                        help="width of an image (default: 224)")
    parser.add_argument('--test_split', type=str, default='test')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--encoder_lr', default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument('--decoder_lr', default=0.0005, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--epochs', default=30, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--grad_clip', type=float, default=0.1)

    parser.add_argument('--finetune_encoder', type=bool, default=False)
    parser.add_argument('--cider_tune', type=bool, default=False)
    parser.add_argument('--epochs_since_improvement', type=int, default=0)
    parser.add_argument('--ss_prob', type=float, default=0.2, help='the probability to use the model prediction during training instead of teacher force')
    # parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
    #                     help="stepsize to decay learning rate, valid if optimizer is sgd")
    # parser.add_argument('--LUT_lr', default=[(60, 0.1), (70, 0.006), (80, 0.0012), (90,0.00024)],
    #                     help="multistep to decay learning rate if using sgd")

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--encoder', type=str, default='vgg16')
    parser.add_argument('--embed_dim', type=int, default=512, help='the embedding dim of word')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dim of decoder RNN')
    parser.add_argument('--model_type', type=str, default='adaptive', help="the model type of decoder, 'adaptive/gridtd/aoa'")
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save_path', type=str, default='./output/adaptive/')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='', help='for evaluation')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0', type=str)


    return parser


def imgcap_gridTD_argument_parser():

    parser = argparse.ArgumentParser(description='Train imgcaptioining arguments')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scale_min', type=float, default=0.9)
    parser.add_argument('--scale_max', type=float, default=1.1)
    parser.add_argument('--rotate_min', type=float, default=-10)
    parser.add_argument('--rotate_max', type=float, default=10)
    parser.add_argument('-j', '--workers', default=1, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=224,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=224,
                        help="width of an image (default: 224)")
    parser.add_argument('--test_split', type=str, default='test')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adamax',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--encoder_lr', default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument('--decoder_lr', default=0.0005, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--epochs', default=20, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--grad_clip', type=float, default=0.1)

    parser.add_argument('--finetune_encoder', type=bool, default=False)
    parser.add_argument('--epochs_since_improvement', type=int, default=0)
    parser.add_argument('--cider_tune', type=bool, default=False)
    parser.add_argument('--lrp_tune', type=bool, default=False)
    parser.add_argument('--lrp_cider_tune',type=bool, default=False)
    parser.add_argument('--ss_prob', type=float, default=None, help='the probability to use the model prediction during training instead of teacher force')
    # parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
    #                     help="stepsize to decay learning rate, valid if optimizer is sgd")
    # parser.add_argument('--LUT_lr', default=[(60, 0.1), (70, 0.006), (80, 0.0012), (90,0.00024)],
    #                     help="multistep to decay learning rate if using sgd")

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--encoder', type=str, default='vgg16')
    parser.add_argument('--embed_dim', type=int, default=512, help='the embedding dim of word')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dim of decoder RNN')
    parser.add_argument('--model_type', type=str, default='gridtd', help="the model type of decoder, 'adaptive/gridtd/aoa'")
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save_path', type=str, default='./output/gridTD/')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='', help='for evaluation')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0', type=str)

    return parser


def imgcap_aoa_argument_parser():

    parser = argparse.ArgumentParser(description='Train imgcaptioining arguments')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--scale_min', type=float, default=0.9)
    parser.add_argument('--scale_max', type=float, default=1.1)
    parser.add_argument('--rotate_min', type=float, default=-10)
    parser.add_argument('--rotate_max', type=float, default=10)
    parser.add_argument('-j', '--workers', default=1, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=224,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=224,
                        help="width of an image (default: 224)")
    parser.add_argument('--test_split', type=str, default='test')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adamax',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--encoder_lr', default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument('--decoder_lr', default=0.0005, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--epochs', default=50, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--grad_clip', type=float, default=0.1)

    parser.add_argument('--finetune_encoder', type=bool, default=False)
    parser.add_argument('--epochs_since_improvement', type=int, default=0)
    parser.add_argument('--cider_tune', type=bool, default=False)
    parser.add_argument('--lrp_tune', type=bool, default=False)
    parser.add_argument('--lrp_cider_tune',type=bool, default=False)
    parser.add_argument('--ss_prob', type=float, default=None, help='the probability to use the model prediction during training instead of teacher force')

    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--encoder', type=str, default='vgg16')
    parser.add_argument('--embed_dim', type=int, default=512, help='the embedding dim of word')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dim of decoder RNN')
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--model_type', type=str, default='aoa', help="the model type of decoder, 'adaptive/gridtd/aoa'")
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save_path', type=str, default='./output/aoa/')
    parser.add_argument('--print_freq', type=int, default=500)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='', help='for evaluation')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0', type=str)

    return parser