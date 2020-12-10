from char import character
from mmcv import Config
from addict import Dict
import logging
opt = Dict()
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp_name', help='Where to store logs and models')
opt.exp_name = "Weight"
# parser.add_argument('--train_data', required=True,
#                     help='path to training dataset')
opt.train_data = "table_lmdb_dataset/train"
# parser.add_argument('--valid_data', required=True,
#                     help='path to validation dataset')
opt.valid_data = "table_lmdb_dataset/val"
# parser.add_argument('--manualSeed', type=int,
#                     default=1111, help='for random seed setting')
opt.manualSeed = 11111
# parser.add_argument('--workers', type=int,
#                     help='number of data loading workers', default=4)
opt.workers = 4
# parser.add_argument('--batch_size', type=int,
#                     default=192, help='input batch size')
opt.batch_size = 192*2
# parser.add_argument('--num_iter', type=int, default=300000,
#                     help='number of iterations to train for')
opt.num_iter = 300000
# parser.add_argument('--valInterval', type=int, default=2000,
#                     help='Interval between each validation')
opt.valInterval = 200
# parser.add_argument('--saved_model', default='',
#                     help="path to model to continue training")
opt.saved_model = "saved_models/Weight/best_norm_ED.pth"
# parser.add_argument('--FT', action='store_true',
#                     help='whether to do fine-tuning')
opt.FT = False
# parser.add_argument('--adam', action='store_true',
#                     help='Whether to use adam (default is Adadelta)')
opt.adam = True
# parser.add_argument('--lr', type=float, default=1,
#                     help='learning rate, default=1.0 for Adadelta')
opt.lr = 1e-4
# parser.add_argument('--beta1', type=float, default=0.9,
#                     help='beta1 for adam. default=0.9')
opt.beta1 = 0.9
# parser.add_argument('--rho', type=float, default=0.95,
#                     help='decay rate rho for Adadelta. default=0.95')
opt.rho = 0.95
# parser.add_argument('--eps', type=float, default=1e-8,
#                     help='eps for Adadelta. default=1e-8')
opt.eps = 1e-8
# parser.add_argument('--grad_clip', type=float, default=5,
#                     help='gradient clipping value. default=5')
opt.grad_clip = 50
# parser.add_argument('--baiduCTC', action='store_true',
#                     help='for data_filtering_off mode')
opt.baiduCTC = False
# """ Data processing """
opt.select_data = None
# parser.add_argument('--select_data', type=str, default='MJ-ST',
#                     help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
opt.batch_ratio = '1.0'
# parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
#                     help='assign ratio for each selected data in the batch')
opt.total_data_usage_ratio = '1.0'
# parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
#                     help='total data usage ratio, this ratio is multiplied to total number of data.')
opt.batch_max_length = 35
# parser.add_argument('--batch_max_length', type=int,
#                     default=25, help='maximum-label-length')
opt.imgH = 32
# parser.add_argument('--imgH', type=int, default=32,
#                     help='the height of the input image')
opt.imgW = 200
# parser.add_argument('--imgW', type=int, default=100,
#                     help='the width of the input image')
opt.rgb = False
# parser.add_argument('--rgb', action='store_true', help='use rgb input')
opt.character = character
# parser.add_argument('--character', type=str,
#                     default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
opt.sensitive = True
# parser.add_argument('--sensitive', action='store_true',
#                     help='for sensitive character mode')
opt.PAD = False
# parser.add_argument('--PAD', action='store_true',
#                     help='whether to keep ratio then pad for image resize')
opt.data_filtering_off = False
# parser.add_argument('--data_filtering_off',
#                     action='store_true', help='for data_filtering_off mode')
# """ Model Architecture """
opt.Transformation = None
# parser.add_argument('--Transformation', type=str,
#                     required=True, help='Transformation stage. None|TPS')
opt.FeatureExtraction = "ResNet"
# parser.add_argument('--FeatureExtraction', type=str, required=True,
#                     help='FeatureExtraction stage. VGG|RCNN|ResNet')
opt.SequenceModeling = None
# parser.add_argument('--SequenceModeling', type=str,
#                     required=True, help='SequenceModeling stage. None|BiLSTM')
opt.Prediction = "CTC"
# parser.add_argument('--Prediction', type=str,
#                     required=True, help='Prediction stage. CTC|Attn')
opt.num_fiducial = 20
# parser.add_argument('--num_fiducial', type=int, default=20,
#                     help='number of fiducial points of TPS-STN')
opt.input_channel = 1
# parser.add_argument('--input_channel', type=int, default=1,
#                     help='the number of input channel of Feature extractor')
opt.output_channel = 512
# parser.add_argument('--output_channel', type=int, default=512, 
#                     help='the number of output channel of Feature extractor')
opt.hidden_size = 256
# parser.add_argument('--hidden_size', type=int, default=256,
#                     help='the size of the LSTM hidden state')
opt = Config(opt,)
#  Base Config