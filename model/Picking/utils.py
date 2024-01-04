import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='test',
        help='Checkpoint name',
    )
    parser.add_argument(
        '--checkpoint_path',
        default='',
        help='Pretrain weight path'
    )
    parser.add_argument(
        '--batch_size', 
        default=128,
        type=int,
        help='Training batch size',
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Training num workers',
    )
    parser.add_argument(
        '--epochs',
        default=200,
        type=int,
        help='Training epochs'
    )
    parser.add_argument(
        '--test_mode',
        default='false',
        help='Input true to enter test mode'
    )
    parser.add_argument(
        '--resume',
        default='false',
        help='Input true to enter resume mode'
    )
    parser.add_argument(
        '--noise_need',
        default='true',
        help='Input n to disable noise data'
    )
    parser.add_argument(
        '--decoder_type',
        default='linear',
        help='linear,cnn,transformer',
    )
    parser.add_argument(
        '--weight',
        default='1.0',
        type=float,
        help='Training P weight'
    )
    parser.add_argument(
        '--early_stop',
        default='10',
        type=int,
        help='Training early stop'
    )
    parser.add_argument(
        '--loss_type',
        default='BCELoss',
        help='Training Loss = BCELoss, Phasenet'
    )
    args = parser.parse_args()
    
    return args