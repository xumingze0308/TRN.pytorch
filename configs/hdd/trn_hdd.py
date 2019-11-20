from configs import parse_base_args, build_data_info

__all__ = ['parse_trn_args']

def parse_trn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/HDD', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='multimodal', type=str)
    parser.add_argument('--hidden_size', default=2000, type=int)
    parser.add_argument('--camera_feature', default='inceptionresnetv2', type=str)
    parser.add_argument('--enc_steps', default=90, type=int)
    parser.add_argument('--dec_steps', default=6, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    return build_data_info(parser.parse_args())
