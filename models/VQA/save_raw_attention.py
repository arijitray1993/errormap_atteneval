############################################ NOTE ###########################
##### make sure you have envs/vqa_env environment activated for VQA to run. 
#############################################################################

import h5py as h5
import random
import json
import os
import numpy as np

from vqa_bert_interface import BertVQA
import bert_vqa_deploy_7x7_v2 as bert_vqa_deploy_7x7
from torch.utils.data import Dataset, DataLoader
import pdb
import tqdm

from multiprocessing import Pool

from vqa_gradcam import GradCam

import random

random.seed(0)

class VQA_v1_hat(Dataset):

    def __init__(self, split="train"):

        ques_data = json.load(open("/data/DataSets/VQA/OpenEnded_mscoco_"+split+"2014_questions.json"))

        ans_data = json.load(open("/data/DataSets/VQA/mscoco_"+split+"2014_annotations.json")) # load answer annotations

        hat_ims = os.listdir("/data/DataSets/VQA/HumanAttention/vqahat_"+split)

        qid2iq= dict()
        for entry in ques_data['questions']:
            qid = entry['question_id']
            im_id = entry['image_id']
            qid2iq[qid]= [im_id, entry['question']]

        
        qid2ans = dict()
        for entry in ans_data['annotations']:
            qid = entry['question_id']
            answer = entry['multiple_choice_answer']
            qid2ans[qid] = answer

        self.hatqid2iq = []
        for imf in hat_ims:
            qid = int(imf.split("_")[0])
            n = int(imf.split("_")[1].split(".")[0])
            self.hatqid2iq.append([qid,n,]+ qid2iq[qid] +[qid2ans[qid]])


    def __getitem__(self, idx):

        return self.hatqid2iq[idx] #qid, imid, question, answer

    def __len__(self):
        return len(self.hatqid2iq)


class ConVQA_data:
    def __init__(self):
        convqadata = json.load(open("../../data/consistentVQASets/updated_commonsense_conVQA_consistent.json"))

        self.convqadata = list(convqadata.items())

    def __getitem__(self, idx):

        im_file, conqas_sets = self.convqadata[idx]

        imid = int(im_file.split("_")[-1].split(".")[0])

        return imid, im_file, conqas_sets

    def __len__(self):
        return len(self.convqadata)


class vqav2_data():
    def __init__(self):
        ques_data = json.load(open("/data/DataSets/VQA/OpenEnded_mscoco_"+split+"2014_questions.json"))
        ans_data = json.load(open("/data/DataSets/VQA/mscoco_"+split+"2014_annotations.json"))
        
        qid2iq = dict()
        for entry in ques_data['questions']:
            qid = entry['question_id']
            im_id = entry['image_id']
            qid2iq[qid]= [im_id, entry['question']]

        qid2ans = dict()
        for entry in ans_data['annotations']:
            qid = entry['question_id']
            answer = entry['multiple_choice_answer']
            qid2ans[qid] = answer

        for qid in qid2iq:
            qid2iq[qid] += [qid2ans[qid],]

        self.data = list(qid2iq.values())

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



def save_output(imid, qid, n, question, answer, gt_a, max_s, attn):
    attn = attn.tolist()
    max_s = max_s.tolist()
    qid = qid.tolist()
    n = n.tolist()
    imid = imid.tolist()
    
    with open("../../data/precomputed_attention_"+model_choice+"/"+split+"/"+str(qid)+"_"+str(n)+".json", "w") as f:
        json.dump([imid, question, answer, gt_a, max_s, attn], f)

if __name__=='__main__':
    split = "val"
    model_choice = "colorcrippled"
    precompute_choice = "humanatt"
    
    if precompute_choice == "humanatt":
        vqa_model = BertVQA(model_choice, data_val="data_vqa_"+split+".pt", gpu_id=0)
        #load human attention
        vqa_hat = VQA_v1_hat(split=split)

        vqa_hat_data = DataLoader(vqa_hat, batch_size=100)

        #all_atten_data = []

        print(len(vqa_hat))
        with Pool(os.cpu_count()-2) as pool:
            for i, (qids, ns, imids, questions, gt_ans) in enumerate(tqdm.tqdm(vqa_hat_data)):
                #try:
                    #pdb.set_trace()
                    a_ids, answers, attns, max_scores, scores = vqa_model.getVQAAns_batch(imids, questions, score=True)
                    attns = attns.cpu()
                    scores = scores.cpu()
                    
                    pool.starmap(save_output, [entry for entry in zip(imids, qids, ns, questions, answers, gt_ans, scores, attns)])
                    
    elif precompute_choice == "convqa":
        vqa_model = BertVQA(model_choice, data_val="data_vqa_"+split+".pt", gpu_id=0)
        convqa = ConVQA_data()
        if not os.path.exists("../../data/consistentVQASets/val/"):
            os.mkdir("../../data/consistentVQASets/val/")
            os.mkdir("../../data/consistentVQASets/valtrain/")
            
        for i, (imid, im_file, conqas_set) in enumerate(tqdm.tqdm(convqa)):
            for conqas in conqas_set:
                questions, gt_answers = zip(*conqas)
                a_ids, answers, attns, max_scores, scores= vqa_model.getVQAAns_batch([imid]*len(conqas), questions, score=True)

                attns = attns.cpu().detach()
                scores = scores.cpu().detach()


                qformat = questions[0].replace("/", "").replace(".", "")
                fname = str(imid)+"_"+("_".join(qformat.split(" ")))+".json"
                
                p=random.random()

                if i>1000:
                    with open("../../data/consistentVQASets/val/"+fname, "w") as f:
                        json.dump((imid, questions, answers, gt_answers, scores[0].tolist(), attns[0].tolist()), f)
                else:
                    with open("../../data/consistentVQASets/valtrain/"+fname, "w") as f:
                        json.dump((imid, questions, answers, gt_answers, scores[0].tolist(), attns[0].tolist()), f)

    elif precompute_choice == 'cams':

        vqa_data = vqav2_data()

        model=bert_vqa_deploy_7x7.load(root=model_choice, data_val="data_vqa_"+split+".pt", gpuid=0, cam_mode=True)

        vqacam = GradCam(model)
        all_cams = dict()
        for i, (imid, ques, ans) in enumerate(tqdm.tqdm(vqa_data)):
            cam = vqacam([imid], [ques])
            all_cams[str(imid)+"_"+str(ques)] = cam.tolist()
        
        with open("../../data/"+model_choice+"_VQA"+split+"_cams.json", "w") as f:
            json.dump(all_cams, f)

        

