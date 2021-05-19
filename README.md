# Generating Error Maps and Evaluating Attention and Error Maps

Code for the paper: Knowing What VQA Does Not: [Pointing to Error-Inducing Regions to Improve Explanation Helpfulness](https://arxiv.org/abs/2103.14712)

## Getting and Generating Data
Make a folder: `mkdir data/precomputed_attention_colorcrippled`, in root folder. 

### Pre-compute raw BERT attention values
Go to `cd models/VQA/`

Download the VQA precomputed train and val data. If ypu are just evaluating, you just need the val data. 
The link for data_vqa_val.pt and data_vqa_train.pt will soon be posted here. 

Download the VQA model checkpoint. Link to checkpoint will be posted soon. Place it in `models/VQA/colorcrippled` folder.

Inside `save_raw_attention.py`, in `if __name__=='__main__':`, change the value of
`split` to `"train"` for first generating precomputed training data. Skip this step if you are just evaluating.

Change value of `split` to `"val"` for generating precomputed validation data. 

Run `python save_raw_attention.py`

This will precompute and save the data. 

### Download Human Attention Dataset
Download the HAT dataset from <https://computing.ece.vt.edu/~abhshkdz/vqa-hat/>. 
Extract them and place them in `data/HumanAttention/` folder. It should have two folders `vqahat_train/` and `vqahat_val/`

### Download GLOVE Features
Download glove features from <http://nlp.stanford.edu/data/glove.6B.zip>. 
We will use the `glove.6B.300d.txt` in the zip file. Place the .txt file in `data/`

### Download the error map model checkpoint

Link will be posted soon. 
Place it in `checkpoints/exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger_recheck/.` folder.


## Evaluating and reproducing paper numbers
Run 
`python main.py`

## Training
Go to `experiment_configs.py`.
Set the variable `eval_only` to `False`.
Run `python main.py`

This will run training on our best model for generating error maps. 

If you want to train your own model, add your model class in models/attention_refine/atten_refine_network.py, 
make a new experiment config in experiment_configs.py. Look at the other examples to see how to define inputs, outputs,
and eval routines. I will add more tutorial info to this later. 

## Visualizing Error/Attention Maps
Run 
`python visualize.py` 

This will save visualized error/attention maps to `vis/exp4_crippledmodel_corrpred_refinedattn_uncertainCAM_bigger_recheck/attention_refine.html` 

## Making Error Maps on your custom Image-Question pairs
Look at `interface.py` for a tutorial on how to generate error maps. 

You will need to change paths in interface.py to point to your dataset locations. The datasets are available publicly. 

