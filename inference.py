from __future__ import print_function
from tqdm import tqdm
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from config import cfg_nano
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.net_nano import Nano
from utils.box_utils import decode, decode_landm


def arg_parse():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-m', '--trained_model', default='./weights/nano_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='nano', help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--long_side',type=int, default=160, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()
    return args

args = arg_parse()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def init_model(model_path,device):
    cfg = cfg_nano
    net = Nano(cfg = cfg, phase = 'test')
    

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')

    net = net.to(device)
    return net

def preprocess(img):
    img = np.float32(img_raw)

    # testing scale
    target_size = args.long_side
    max_size = args.long_side
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    
    print("args.origin_size:",args.origin_size)
    if args.origin_size:
        
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape

    print(img.shape)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    return img,scale,resize,im_height, im_width

def post_process(loc, conf, landms,priorbox):
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets

def draw_result(in_img,dets):
    img = in_img.copy()
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    return img

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = cfg_nano

    print("args.cpu:",args.cpu)
    device = torch.device("cpu" if args.cpu else "cuda")
    
    net = init_model(model_path="./weights/nano_Final.pth",device=device)

    image_path = "./img/test.jpg"
    if not os.path.exists(image_path):
        print("img not exists")
        exit()
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    
    img,scale,resize,im_height, im_width = preprocess(img_raw)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    
    dets = post_process(loc, conf, landms,priorbox=priorbox)
    
    

    # show image
    if args.save_image:
        
        img_save = draw_result(img_raw,dets=dets)
        # save image
        save_path = "outputs"
        os.makedirs(save_path,exist_ok=True)
        name = os.path.basename(image_path)
        flag = cv2.imwrite(os.path.join(save_path,name), img_save)
        if flag:
            print(f'img save: {os.path.join(save_path,name)} ')
        
            
