
import torch
import os
import numpy
import scipy
import re
import db as db
import math
from pytorch_transformers import BertTokenizer
import copy
import model_7x7 as base_model
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets.folder
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from PIL import Image, ImageOps, ImageEnhance


class load:
    def __init__(self,root,data_val,gpuid=0, cam_mode=False):
        self.args=torch.load(os.path.join(root,'args.pt'));
        self.answer_dictionary=torch.load(os.path.join(root,'answer_dictionary.pt'));
        self.args.cam_mode = cam_mode
        #Instantiate model
        model=base_model.simple_vqa_model(self.args).cuda();
        self.model=nn.DataParallel(model).cuda();
        #self.model=model.module.cuda(gpuid)
        
        #Load model checkpoint
        if os.path.exists(os.path.join(root, 'model_checkpoint.pt')):
            checkpoint = torch.load(os.path.join(root, 'model_checkpoint.pt'))
        else:
            for i in range(self.args.epochs,0,-1):
                try:
                    print('Try loading checkpoint %d'%i)
                    checkpoint=torch.load(os.path.join(root,'model_epoch%02d.pth'%i));
                    break;
                except:
                    pass;
            
        self.model.load_state_dict(checkpoint['model_state'])
        
        self.model.eval();
        
        if isinstance(data_val,str):
            self.data_val=db.DB.load(data_val);
        else:
            self.data_val=data_val;
        
        #Prepare BERT tokenizer
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased');
    
    #def cuda(self,gpuid=0):
    #    return;
    
    def answer_dictionary(self):
        return self.answer_dictionary;
    
    def nbox(self):
        return self.data_val['table_ifvs']['features'].shape[1];
    
    def nspatial(self):
        return self.data_val['table_ifvs']['features_7x7'].shape[2]*self.data_val['table_ifvs']['features_7x7'].shape[3];
    
    def parse_question(self,qtext):
        qtokens=self.tokenizer.tokenize(qtext);
        if len(qtokens)>self.args.max_qlength-2:
            qtokens=qtokens[:self.args.max_qlength-2];
        
        qtokens=['[CLS]']+qtokens+['[SEP]'];
        question=self.tokenizer.convert_tokens_to_ids(qtokens);
        question=question+[0]*(self.args.max_qlength-len(question));
        question=torch.LongTensor(question);
        return qtokens,question;
    
    def vqa(self,coco_id,qtext):
        #Lookup image id in table_iid
        id=self.data_val['table_iid']['coco_id'].index(coco_id);
        iid=self.data_val['table_iid']['iid'][id];
        
        #Get feature and question
        feature=self.data_val['table_ifvs']['features'][iid:iid+1,:,:];
        feature_7x7=self.data_val['table_ifvs']['features_7x7'][iid:iid+1,:,:,:].clone().permute(0,2,3,1).clone();
        _,q=self.parse_question(qtext);
        q=q.view(1,-1);
        
        scores,attn=self.model(feature,feature_7x7,q);
        attn=torch.cat(attn,dim=0)
        return scores.view(-1),attn;
    
    def vqa_batch(self,coco_ids,qtexts):
        #Lookup image id in table_iid
        iids=[];
        for coco_id in coco_ids:
            id=self.data_val['table_iid']['coco_id'].index(coco_id);
            iids.append(self.data_val['table_iid']['iid'][id]);
        
        iids=torch.LongTensor(iids);
        
        #Get feature and question
        feature=self.data_val['table_ifvs']['features'][iids,:,:];
        feature_7x7=self.data_val['table_ifvs']['features_7x7'][iids,:,:,:].clone().permute(0,2,3,1).clone();
        
        qs=[];
        for qtext in qtexts:
            _,q=self.parse_question(qtext);
            qs.append(q.view(1,-1));
        
        qs=torch.cat(qs,dim=0);
        
        scores,attn=self.model(feature,feature_7x7,qs);
        attn=torch.stack(attn,dim=1)
        return scores,attn;

    def vqa_cam(self, coco_ids, qtexts):
        iids=[];
        for coco_id in coco_ids:
            id=self.data_val['table_iid']['coco_id'].index(coco_id);
            iids.append(self.data_val['table_iid']['iid'][id]);
        
        iids=torch.LongTensor(iids);
        
        #Get feature and question
        feature=self.data_val['table_ifvs']['features'][iids,:,:];
        feature_7x7=self.data_val['table_ifvs']['features_7x7'][iids,:,:,:].clone().permute(0,2,3,1).clone();
        
        feature = torch.tensor(feature, requires_grad=True)
        feature_7x7 = torch.tensor(feature_7x7, requires_grad=True)

        qs=[];
        for qtext in qtexts:
            _,q=self.parse_question(qtext);
            qs.append(q.view(1,-1));
        
        qs=torch.cat(qs,dim=0);
        scores,attn=self.model(feature,feature_7x7,qs);

        return scores, feature, feature_7x7



    
    def get_bbox_by_coco_id(self,coco_id):
        #Lookup image id in table_iid
        id=self.data_val['table_iid']['coco_id'].index(coco_id);
        iid=self.data_val['table_iid']['iid'][id];
        #Get bounding box information
        #BBox is a Nx4 matrix
        #   N boxes
        #   format is (x0,y0,x1,y1) in [0,1]
        bbox=self.data_val['table_ifvs']['spatials'][iid,:,0:4];
        return bbox;
    
    def question_vector_v0(self,qtext,ref_coco_ids,T=15,std=1e-3,batch=4):
        def logmeanexp(inputs,dim=None,keepdim=False):
            return (inputs-F.log_softmax(inputs,dim=dim).data).mean(dim,keepdim=keepdim)-math.log(inputs.size(dim));
        
        
        seeds=[t*1000 for t in range(T)]; #Fix seeds across runs
        
        #Preprocess question
        _,q=self.parse_question(qtext);
        q=q.view(1,-1);
        
        #Lookup image id in table_iid
        iids=[];
        for coco_id in ref_coco_ids:
            id=self.data_val['table_iid']['coco_id'].index(coco_id);
            iids.append(self.data_val['table_iid']['iid'][id]);
        
        iids=torch.LongTensor(iids);
        #Get feature and question
        feature=self.data_val['table_ifvs']['features'][iids,:,:];
        feature_7x7=self.data_val['table_ifvs']['features_7x7'][iids,:,:,:].clone().permute(0,2,3,1).clone();
        
        model2=copy.deepcopy(self.model);
        model2.train();
        s=[];
        for t in range(T):
            st=[];
            rng_state=torch.random.get_rng_state();
            torch.random.manual_seed(seeds[t]);
            #Run the model, pairing the q with each images
            with torch.no_grad():
                for j in range(0,len(ref_coco_ids),batch):
                    r=min(j+batch,len(ref_coco_ids));
                    scores,_=model2(feature[j:r],feature_7x7[j:r],q.repeat(r-j,1));
                    scores=F.log_softmax(scores,dim=1).data;
                    st.append(scores);
            
            torch.random.set_rng_state(rng_state);
            st=torch.cat(st,dim=0);
            s.append(st.data);
        
        s=torch.stack(s,dim=0); #TxKx3129
        savg=logmeanexp(s,dim=0,keepdim=True);
        sdiff=s-savg;
        s=s.permute(1,0,2);
        sdiff=sdiff.permute(1,2,0);
        v=torch.bmm(torch.exp(s),torch.exp(sdiff))/T;
        return v.view(-1);
