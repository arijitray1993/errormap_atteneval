import sys 
#sys.path.append("../")
import json
import os
import numpy as np
import pickle as pkl
from scipy.stats import entropy
import random
import torch.nn.functional as F
#from ImageEncoders.MaskRCNN import model as modellib
#rom ImageEncoders.MaskRCNN.config import Config
#from ImageEncoders.MaskRCNN.visualize import *
from PIL import Image
import cv2
from io import BytesIO
import bert_vqa_deploy_7x7_v2 as bert_vqa_deploy_7x7
import torch
import pdb 
import time


'''
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
'''

def same_ques(ques1, ques2):
    q1_words = ques1.lower().strip().split(" ")
    q2_words = ques2.lower().strip().split(" ")

    q1_words = [lmtzr.lemmatize(w) for w in q1_words]
    q2_words = [lmtzr.lemmatize(w) for w in q2_words]

    q1_dict = dict()
    for w in q1_words:
        q1_dict[w] = 1

    allCount = len(q2_words)
    count=0
    for w in q2_words:
        if w in q1_dict:
            count+=1

    if float(count)/allCount > 0.7:
        return True
    else:
        return False

def closest_question_lstm(embed, qas, og_ques):
    ques = []
    for q in qas:
        if q in ["kj", "&7^()"]:
            continue

        if same_ques(og_ques, q):
            continue

        dist = np.sqrt(sum(np.square(qas[q][0] - embed[0])))

        ques.append((q, dist))

    ques.sort(key = lambda x:x[1])

    #return list(zip(*ques))[0]
    return ques

def apply_mask(image, mask, color, alpha=0.7):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

#@jit(nopython=True, parallel=True)
def apply_obj_mask(masked_image, mask, actual_image, weight):

    mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    obj_image = np.ones(actual_image.shape)*255

    np.copyto(obj_image, actual_image, where=(mask==1))

    white_image = np.ones(actual_image.shape)*255

    if weight< 0.3:
        weight=weight+0.15
    obj_img_weighted = weight*obj_image + (1-weight)*white_image

    np.copyto(masked_image, obj_img_weighted, where=(mask==1))

    return masked_image

def computeIOU(box1, box2):
    #boxes should be in (y1, x1, y2, x2)
    box1 = np.asarray(box1).astype(np.float32)
    box2 = np.asarray(box2).astype(np.float32)
    
    iou_box_x1 = max(box1[1], box2[1])
    iou_box_y1 = max(box1[0], box2[0])
    iou_box_x2 = min(box1[3], box2[3])
    iou_box_y2 = min(box1[2], box2[2])
    
    iou_h = max(0, iou_box_y2-iou_box_y1)
    iou_w = max(0, iou_box_x2 - iou_box_x1)
    
    roi_area = (iou_h * iou_w)
    
    box1_area = np.absolute((box1[3] - box1[1]) * (box1[2] - box1[0]))
    box2_area = np.absolute((box2[3] - box2[1]) * (box2[2] - box2[0]))
    
    iou = roi_area/float(box1_area + box2_area - roi_area)
    
    return iou

def compute_box_distance(box1, box2):
    #boxes in (y1, x1, y2, x2)
    box1 = np.asarray(box1).astype(np.float32)
    box2 = np.asarray(box2).astype(np.float32)
    
    cntr_box1_x = int((box1[1] + box1[3])/2)
    cntr_box1_y = int((box1[0] + box1[2])/2)
    
    cntr_box2_x = int((box2[1] + box2[3])/2)
    cntr_box2_y = int((box2[0] + box2[2])/2)
    
    dist = np.sqrt((cntr_box1_x - cntr_box2_x)**2 + (cntr_box1_y - cntr_box2_y)**2)
    
    return dist

def computeWeights(mrcnn_boxes, rpn_boxes, box_weights):
    epsilon = 1e-5

    rcnn_box_weights = []
    for ind, rcnn_box in enumerate(mrcnn_boxes):
        max_area = 0
        all_iou = []
        all_weights = []
        for rpn_ind, rpn_box in enumerate(rpn_boxes):
            iou_area = computeIOU(rcnn_box, rpn_box)
            # distance = compute_box_distance(rpn_box, rcnn_box)
            # norm_dist = distance / (1024 * np.sqrt(2))
            # if iou_area <= 0 and (norm_dist > 0.3):
            #    continue;

            # if iou_area <= 0:
            #    iou_area = (1 - norm_dist)/float(4)
            all_iou.append(iou_area)
            all_weights.append(box_weights[rpn_ind])

            # print(all_iou)
        # print(all_weights)
        if len(all_iou) >= 1 and np.sum(all_iou)>0:
            final_weight = np.exp(np.log(np.sum(np.exp(np.log(np.asarray(all_iou)) + np.log(np.asarray(all_weights))))) -
                                  (np.log(float(np.sum(all_iou)+ epsilon))))

            rcnn_box_weights.append(final_weight)
        else:
            rcnn_box_weights.append(0)

    return rcnn_box_weights

def make_rpn_attention_im(img_name, im_file, attention_rpn, bboxes, token_ind):
    box_weights = attention_rpn.numpy()  # np.average(attention_rpn, axis=-1)
    '''
    r = all_rcnn_preds

    mrcnn_boxes = r['rois']
    mrcnn_obj_classes = r['class_ids']
    mrcnn_masks = r['masks']
    mrcnn_probs = r['scores']
    '''

    im_boxes = bboxes
    im_boxes = im_boxes * 256
    final_obj_weights = box_weights #computeWeights(im_boxes, im_boxes, box_weights)
    if len(final_obj_weights) != 0:
        # end = time.time()
        if np.max(final_obj_weights) > 0:
            final_obj_weights = np.exp(np.log(final_obj_weights) - np.log(np.max(final_obj_weights)))
    # print("compute time :"+str(end-start))
    #N = len(final_obj_weights)

    #pdb.set_trace()
    actual_image = Image.open(im_file)
    
    actual_image = actual_image.resize((256,  256))

    img_arr = np.asarray(actual_image)
    if len(img_arr.shape)<3:
            img_arr = img_arr[:,:,np.newaxis]
            img_arr = np.repeat(img_arr, 3, axis=2)
    elif img_arr.shape[2]==1:
            img_arr = np.repeat(img_arr, 3, axis=2)
    masked_image = np.ones(img_arr.shape) * 255
    masked_image = img_arr * 0.1 + masked_image * 0.9
    
    if len(final_obj_weights) != 0:
        obj_atten_inds = np.argsort(final_obj_weights)
    else:
        obj_atten_inds = []
    obj_atten_inds = obj_atten_inds[::-1]
    top_N = 5  # int(N * float(3) / 4)
    for i in obj_atten_inds[:top_N][::-1]:
    
        if final_obj_weights[i] > 0:
            mask = np.zeros((256,256)) #mrcnn_masks[:, :, i]
            im_boxes = im_boxes.astype(np.int32)
            x0, y0, x1, y1 = im_boxes[i]
            mask[y0:y1, x0:x1]=np.ones((y1-y0, x1-x0))
            # masked_image = apply_mask(masked_image, mask, rgba)
            masked_image = apply_obj_mask(masked_image, mask, img_arr,
                                        float(final_obj_weights[i]))

    ## draw origin box (clicked box and draw arrows from that box to attended boxes)
    ## will only work for cases where we have such box to box attention, think about generalizing this later
    if token_ind>29 and token_ind<66:
        origin_box = im_boxes[token_ind-30]
        ox0, oy0, ox1, oy1 = origin_box
        cv2.rectangle(masked_image, (origin_box[0], origin_box[1]), (origin_box[2], origin_box[3]), (100,100,100), 5)

        for i in obj_atten_inds[:top_N]:
            x0, y0, x1, y1 = im_boxes[i]
            cv2.rectangle(masked_image, (x0, y0), (x1, y1), (50, 50, 50), 1)
            pt1, pt2 = compute_closest_corner(origin_box, im_boxes[i])
            #xc = int((x0+x1)/2.0)
            #yc = int((y0+y1)/2.0)
            cv2.arrowedLine(masked_image, pt1, pt2, (100,100,100), 2,8,0,0.05)

    
    #pdb.set_trace() 
    masked_im = Image.fromarray(masked_image.astype(np.uint8))
    #masked_im = masked_im.resize((224,224))
    masked_im.save(img_name) #, dpi=(100, 100))
    
    actual_image.close()

def compute_closest_corner(box1, box2):

    ax0, ay0, ax1, ay1 = box1
    bx0, by0, bx1, by1 = box2
    min_d = float("inf")
    for ax in [ax0, ax1]:
        for bx in [bx0, bx1]:
            d = abs(ax-bx)
            if d<min_d:
                ax_c = ax
                bx_c = bx
                min_d = d

    min_d = float("inf")
    for ay in [ay0, ay1]:
        for by in [by0, by1]:
            d = abs(ay-by)
            if d<min_d:
                ay_c = ay
                by_c = by
                min_d = d

    return (ax_c, ay_c), (bx_c, by_c)


def make_spatial_attention_im(img_sp_name, im_file, attention_sp):
    epsilon = 1e-4
    attention_sp = np.reshape(attention_sp, (7,7))  
    actual_image = Image.open(im_file)
    # 'attention_sp' in session['explType']:
    processed_img = actual_image.resize((224, 224))
    # print(2)
    processed_img = np.asarray(processed_img).astype(np.float32)
    img_arr= processed_img
    if len(img_arr.shape)<3:
        img_arr = img_arr[:,:,np.newaxis]
        img_arr = np.repeat(img_arr, 3, axis=2)
    elif img_arr.shape[2]==1:
        img_arr = np.repeat(img_arr, 3, axis=2)
    processed_img = img_arr

    # print(3)
    attention_spatial = np.squeeze(attention_sp)
    att_map = cv2.resize(attention_spatial, (224, 224))
    # print(4)
    att_map = (att_map - np.min(att_map)+epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
    # att_map = np.sqrt(att_map)
    
    # print(5)
    att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
    alpha = 0.5
    # print(6)
    #white_image = alpha * np.ones((224, 224, 3)) * 255 + (1 - alpha) * processed_img
    output_image = (1 - alpha) * att_heatmap + alpha * processed_img
    #output_image = (1 - np.expand_dims(att_map, 2)) * white_image + np.expand_dims(att_map,
    #                                                                               2) * processed_img        
    
    actual_image.close()
    cv2.imwrite(img_sp_name, output_image)



def compute_allCombination_attention_im_file(answer, im_file, dense_att_weights, bboxes, question, q_tokens, attention_folder='attention_images/', prefix=""):
    
    formattedImQ = "_".join(question.split(" "))
    formattedImA = "_".join(answer.split(" "))
    chosen_imgid = im_file.split("/")[-1].split(".")[0]
    
    all_img_sps = []
    all_img_rpns=[]
    
    #tic= time.time()
    #### RPN object-based attention ######
    tc=0
    for token in dense_att_weights[-1, -1, :,:]:
        img_name = attention_folder + str(chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_"+str(tc)+"rpnregion"+str(prefix)+".png"

        if not os.path.exists(img_name):
            attention_rpn = token[66-len(bboxes):66]
            make_rpn_attention_im(img_name, im_file, attention_rpn, bboxes, tc)
        tc+=1
        all_img_rpns.append(img_name)
    #toc = time.time()
    #print("all rpn time : "+str(toc-tic)) 

    #tic = time.time()
    ##### Compute the average attention over all words in question for rpn attention #####
    img_name = attention_folder + str(chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_average_rpnregion"+str(prefix)+".png"
    attention_rpn = dense_att_weights[-1, -1, :len(q_tokens), 66-len(bboxes):66].mean(0)
    make_rpn_attention_im(img_name, im_file, attention_rpn, bboxes, token_ind=-1)
    all_img_rpns.append(img_name)
    #toc = time.time()
    #print("one avg rpn time : "+str(toc-tic))
    
    #tic = time.time()
    ###### Spatial attention #########
    tc=0    
    for token in dense_att_weights[-1, :, :, :].mean(0): 
        img_sp_name = attention_folder + str(chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_"+str(tc)+"spatial"+str(prefix)+".png"   
        
        if not os.path.exists(img_sp_name):
            attention_sp = token[66:].numpy()
            make_spatial_attention_im(img_sp_name, im_file, attention_sp)      
        tc+=1
        all_img_sps.append(img_sp_name)
    #toc = time.time()
    #print("spatial time : "+str(toc-tic))

    #tic = time.time()
    ###### Compute average attention over all words for spatial attention ###
    img_sp_name = attention_folder + str(chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_average_spatial"+str(prefix)+".png"   
    attention_sp = dense_att_weights[-1, :, :len(q_tokens), 66:].mean(0).mean(0).numpy()
    make_spatial_attention_im(img_sp_name, im_file, attention_sp)
    all_img_sps.append(img_sp_name)
    #toc = time.time()
    #print("one avg spatial time : "+str(toc-tic))

    return all_img_sps, all_img_rpns

def compute_attention_im_file(answer, im_file, dense_att_weights, bboxes, question, q_tokens, attention_folder='attention_images/', prefix=""):
    epsilon = 1e-4
    formattedImQ = "_".join(question.split(" "))
    formattedImA = "_".join(answer.split(" "))
    chosen_imgid = im_file.split("/")[-1].split(".")[0]
    img_name = attention_folder + str(
        chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_"+str(prefix)+"rpnregion.png"
    img_sp_name = attention_folder + str(
        chosen_imgid) + "_" + formattedImQ + "_" + formattedImA + "_"+str(prefix)+"spatial.png"

    if os.path.exists(img_name) and os.path.exists(img_sp_name):
        return img_sp_name, img_name
    else:
        attention_rpn = np.average(dense_att_weights[-1, -1, :len(q_tokens),66-len(bboxes):66], axis=0)
        attention_sp = np.average(dense_att_weights[-1, :, :len(q_tokens), 66:].mean(0), axis=0)

        attention_sp = np.reshape(attention_sp, (7,7))
        
        box_weights = attention_rpn  # np.average(attention_rpn, axis=-1)
        '''
        r = all_rcnn_preds

        mrcnn_boxes = r['rois']
        mrcnn_obj_classes = r['class_ids']
        mrcnn_masks = r['masks']
        mrcnn_probs = r['scores']
        '''

        im_boxes = bboxes
        im_boxes = im_boxes * 1024
        final_obj_weights = computeWeights(im_boxes, im_boxes, box_weights)
        if len(final_obj_weights) != 0:
            # end = time.time()
            if np.max(final_obj_weights) > 0:
                final_obj_weights = np.exp(np.log(final_obj_weights) - np.log(np.max(final_obj_weights)))
        # print("compute time :"+str(end-start))
        N = len(final_obj_weights)

        #pdb.set_trace()
        actual_image = Image.open(im_file)
        
        actual_image = actual_image.resize((1024, 1024))

        if True:  # 'attention_sp' in session['explType']:
            processed_img = actual_image.resize((224, 224))
            # print(2)
            processed_img = np.asarray(processed_img).astype(np.float32)
            img_arr= processed_img
            if len(img_arr.shape)<3:
                img_arr = img_arr[:,:,np.newaxis]
                img_arr = np.repeat(img_arr, 3, axis=2)
            elif img_arr.shape[2]==1:
                img_arr = np.repeat(img_arr, 3, axis=2)
            processed_img = img_arr

            # print(3)
            attention_spatial = np.squeeze(attention_sp)
            att_map = cv2.resize(attention_spatial, (224, 224))
            # print(4)
            att_map = (att_map - np.min(att_map) + epsilon) / (np.max(att_map) - np.min(att_map)+epsilon)
            # att_map = np.sqrt(att_map)
            

            # print(5)
            att_heatmap = cv2.applyColorMap(np.uint8(255 * att_map), cv2.COLORMAP_JET)
            alpha = 0.5
            # print(6)
            #white_image = alpha * np.ones((224, 224, 3)) * 255 + (1 - alpha) * processed_img
            output_image = (1 - alpha) * att_heatmap + alpha * processed_img
            #output_image = (1 - np.expand_dims(att_map, 2)) * white_image + np.expand_dims(att_map,
            #                                                                               2) * processed_img

        img_arr = np.asarray(actual_image)
        if len(img_arr.shape)<3:
                img_arr = img_arr[:,:,np.newaxis]
                img_arr = np.repeat(img_arr, 3, axis=2)
        elif img_arr.shape[2]==1:
                img_arr = np.repeat(img_arr, 3, axis=2)
        masked_image = np.ones(img_arr.shape) * 255
        masked_image = img_arr * 0.1 + masked_image * 0.9
        
        if len(final_obj_weights) != 0:
            obj_atten_inds = np.argsort(final_obj_weights)
        else:
            obj_atten_inds = []
        obj_atten_inds = obj_atten_inds[::-1]
        top_N = 5  # int(N * float(3) / 4)
        for i in obj_atten_inds[:top_N][::-1]:
        
            if final_obj_weights[i] > 0:
                mask = np.zeros((1024,1024))#mrcnn_masks[:, :, i]
                im_boxes = im_boxes.astype(np.int32)
                x0, y0, x1, y1 = im_boxes[i]
                mask[y0:y1, x0:x1]=np.ones((y1-y0, x1-x0))
                # masked_image = apply_mask(masked_image, mask, rgba)
                masked_image = apply_obj_mask(masked_image, mask, img_arr,
                                            float(final_obj_weights[i]))
        #pdb.set_trace() 
        masked_im = Image.fromarray(masked_image.astype(np.uint8))
        
        masked_im.save(img_name, dpi=(20, 20))
        actual_image.close()
        cv2.imwrite(img_sp_name, output_image)
        return img_sp_names, img_names



class BertVQA:

    def __init__(self, model_path="../simple_bert_7x7_4", data_val='data_vqa_val.pt', gpu_id=0):

        #self.model=bert_vqa_deploy.load('../simple_bert_12','../data_vqa_val.pt')
        self.model=bert_vqa_deploy_7x7.load(model_path, data_val=data_val, gpuid=gpu_id)


    def getVQAAns_batch(self, imgs, ques, score=False):
        ## doesn't support attention generation yet
        with torch.no_grad():
            #pdb.set_trace()
            scores, attns=self.model.vqa_batch(imgs,ques)  #imgs are coco_ids
        scores = F.softmax(scores)
        max_scores, ans_ids = scores.max(1)

        answers=[]
        for a_id in ans_ids:
            answers.append(self.model.answer_dictionary[a_id])

        if score:
            return ans_ids, answers, attns, max_scores, scores
        else:
            return ans_ids, answers, attns


    
    def getVQAAns(self, img, ques, expl=True, attention_folder='attention_images/', prefix=""):
        # get image features
        '''
        image = img #cv2.imread(img)
        ques = ques.strip().lower().split("?")[0].split(".")[0]
        mrcnn_image = cv2.resize(image, (1024, 1024))  # do this or mold inputs fail.
        #print(mrcnn_image.shape)
        molded_images, image_metas, _ = mrcnn_model.mold_inputs([mrcnn_image])
        out_mrcnn_feat = mrcnn_model_feats.predict([molded_images, image_metas])
        rpn_boxes, rpn_feats = out_mrcnn_feat[0], out_mrcnn_feat[3]
        results, mrcnn_class, mrcnn_bbox, mrcnn_mask, P2, P3, P4, P5 = \
            mrcnn_model.detect([mrcnn_image], verbose=0)

        r = results[0]
        mrcnn_boxes = r['rois']
        mrcnn_obj_classes = r['class_ids']
        mrcnn_masks = r['masks']
        mrcnn_probs = r['scores']
        '''
        coco_id = img #check this

        #tic = time.time()
        with torch.no_grad():
            #pdb.set_trace()
            score,attn=self.model.vqa(coco_id,ques)
        _,apred=score.max(0);
        apred=int(apred);
        #toc = time.time()

        #print("VQA Answer time : "+str(toc-tic))

        q_tokens, q_tensor = self.model.parse_question(ques)

        answer = self.model.answer_dictionary[apred]
        attn = attn.cpu()

        if False:
            with open("debug_attention/"+str(coco_id)+"_"+str(ques)+".pkl", "wb") as f:
                pkl.dump(attn, f)
        

        coco_dir='/data/diva-6/sangwoo/data/mscoco/val2014'
        im_file=os.path.join(coco_dir,'COCO_val2014_%012d.jpg'%coco_id)

        if expl:
            if not os.path.isdir(attention_folder):
                os.mkdir(attention_folder)
            #ques_atten = attn[-1, -1, :, :len(q_tokens)].numpy()
            bboxes = self.model.get_bbox_by_coco_id(coco_id).cpu().numpy()  # bboxes are in x1, y1, x2, y2 

            #tic = time.time()
            atten_im_file_sps, atten_im_file_rpns = compute_allCombination_attention_im_file(answer, im_file, attn, bboxes , ques, q_tokens, attention_folder, prefix)
            #toc = time.time()
            #print("expl gen time : "+str(toc-tic))

            bboxes = self.model.get_bbox_by_coco_id(coco_id).cpu().tolist()
            ques_atten = attn[-1, -1, :, :len(q_tokens)]
            return answer, score, q_tensor, atten_im_file_sps, atten_im_file_rpns, ques_atten, bboxes
        
        return answer, score, q_tensor



