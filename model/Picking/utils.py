import argparse
import seisbench.data as sbd

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name',         default='test',                 help='Checkpoint name')
    parser.add_argument('--checkpoint_path',    default='None',                 help='Pretrain weight path')
    parser.add_argument('--batch_size',         default=128,        type=int,   help='Training batch size')
    parser.add_argument('--num_workers',        default=4,          type=int,   help='Training num workers')
    parser.add_argument('--epochs',             default=200,        type=int,   help='Training epochs')
    parser.add_argument('--test_mode',          default='false',                help='Input true to enter test mode')
    parser.add_argument('--resume',             default='false',                help='Input true to enter resume mode')
    parser.add_argument('--noise_need',         default='true',                 help='Input false to disable noise data')
    parser.add_argument('--decoder_type',       default='linear',               help='linear,cnn,transformer')
    parser.add_argument('--weight',             default='1.0',      type=float, help='Training P weight')
    parser.add_argument('--early_stop',         default='7',        type=int,   help='Training early stop')
    parser.add_argument('--loss_type',          default='BCELoss',              help='Training Loss = BCELoss, Phasenet')
    parser.add_argument('--model_type',         default='onlyp',                help='onlyp, ps')
    parser.add_argument('--test_set',           default='test',                 help='test or dev')
    parser.add_argument('--threshold',          default='0.2',      type=float, help='Picking threshold')
    parser.add_argument('--train_model',        default='wav2vec2',             help='wav2vec2, phasenet, eqt')
    parser.add_argument('--parl',               default='y',                    help='y or n, parller training')
    parser.add_argument('--task',               default='pick',                 help='pick or detect')
    parser.add_argument('--freeze',             default='y',                    help='y or n')
    parser.add_argument('--lr',                 default='0.0005',   type=float, help='Learning rate')
    parser.add_argument('--weighted_sum',       default='n',                    help='y or n')
    parser.add_argument('--dataset',            default='tw',                   help='tw or stead')
    parser.add_argument('--result',             default='n',                    help='picking result')
    parser.add_argument('--image',              default='n',                    help='y for picking image')

    args = parser.parse_args()
    return args

def get_dataset(args):

    data_path = "/mnt/nas5/johnn9/dataset/"
    cwb_path = data_path + "cwbsn/"
    tsm_path = data_path + "tsmip/"
    noi_path = data_path + "cwbsn_noise/"
    stead_path = data_path + "stead/"

    if args.dataset == 'tw':
        cwb = sbd.WaveformDataset(cwb_path, sampling_rate=100)
        c_mask = cwb.metadata["trace_completeness"] == 4
        cwb.filter(c_mask)
        tsm = sbd.WaveformDataset(tsm_path, sampling_rate=100)
        t_mask = tsm.metadata["trace_completeness"] == 1
        tsm.filter(t_mask)
        if args.task == 'detect':
            cp_mask = cwb.metadata["trace_p_arrival_sample"].notna()
            cs_mask = cwb.metadata["trace_s_arrival_sample"].notna()
            cwb.filter(cp_mask)
            cwb.filter(cs_mask)
            tp_mask = tsm.metadata["trace_p_arrival_sample"].notna()
            ts_mask = tsm.metadata["trace_s_arrival_sample"].notna()
            tsm.filter(tp_mask)
            tsm.filter(ts_mask)
        c_train, c_dev, c_test = cwb.train_dev_test()
        t_train, t_dev, t_test = tsm.train_dev_test()
        if args.noise_need == 'true':
            noise = sbd.WaveformDataset(noi_path, sampling_rate=100)
            n_train, n_dev, n_test = noise.train_dev_test()
            train = c_train + t_train + n_train
            dev = c_dev + t_dev + n_dev
            test = c_test + t_test + n_test
        elif args.noise_need == 'false':
            train = c_train + t_train
            dev = c_dev + t_dev
            test = c_test + t_test
        print('tw dataset')
        print('c_train = ', len(c_train))
        print('t_train = ', len(t_train))
        if args.noise_need == 'true':
            print('n_train = ', len(n_train))
    elif args.dataset == 'stead':
        stead = sbd.WaveformDataset(stead_path, sampling_rate=100)
        train, dev, test = stead.train_dev_test()
        print('stead dataset')
        print('train = ', len(c_train))
    return train,dev,test
