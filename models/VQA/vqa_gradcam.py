import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import pdb

class GradCam:
    def __init__(self, model, use_cuda=True):
        self.model = model
        self.cuda = use_cuda

    def forward(self, input):
        return self.model(*input)

    def __call__(self, coco_ids, qtexts, index=None, cam_size=(7,7)):
        self.model.model.zero_grad()
        
        try:
            output, feature, feature_7x7 = self.model.vqa_cam(coco_ids, qtexts)
        except:
            pdb.set_trace()

        index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1.0
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.model.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.model.model.module.grad_7x7[-1].cpu().data.numpy()

        target = feature_7x7
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        #pdb.set_trace()
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, cam_size)
        cam = cam - np.min(cam)
        delta = 1e-4
        cam = cam / (np.max(cam)+delta)

        #pdb.set_trace()
        return cam