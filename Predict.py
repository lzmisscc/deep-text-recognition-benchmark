import string
import argparse
import uuid
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from utils import CTCLabelConverter, AttnLabelConverter
# from dataset import LmdbDataset
from model import Model
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

compose = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.88, 0.18)
])
with open("../table_ocr/abs_val.txt", "r") as f:
    lines = f.readlines()

names,labels = [], []
for line in lines:
    name, label = line.strip("\n").split("\t")
    names.append(name), labels.append(label)

def pre_pre(img, opt):
    import math
    image = img
    w, h = image.size
    # name = uuid.uuid4().hex
    # image.save(f"VIS/{name}-gt.jpg")
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) <= opt.imgW:
        resized_w = opt.imgW
        resized_image = image.resize(
            (resized_w, opt.imgH), Image.BICUBIC)

        new_PAD = Image.new(size=(opt.imgW, opt.imgH), color=(255),mode='L')
        new_PAD.paste(resized_image, (0,0))
        img = new_PAD  
        # img.save(f"VIS/{name}.jpg") 
        return img
    return image


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    
    model.eval()
    with torch.no_grad():
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\t{"T/F":5s}\tconfidence score'
        dashed_line = '-' * 80
        print(f'{dashed_line}\n{head}\n{dashed_line}')
        for image_tensors, image_path_list in zip(names, labels):
            im = Image.open(image_tensors)
            im = pre_pre(im, opt)
            image_tensors = compose(im)
            image_tensors = torch.unsqueeze(image_tensors, 0).float()
            batch_size = image_tensors.size(0)
            image_path_list = [image_path_list, ]
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')

            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{img_name==pred}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

if __name__ == '__main__':
    from CONFIG2TRAIN.config import opt
    opt.saved_model = "saved_models/Weight/best_accuracy.pth"
    opt.batch_size = 1
    opt.imgW = 32
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
