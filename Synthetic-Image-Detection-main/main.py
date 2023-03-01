import os
import pandas
import tqdm
from sys import argv
from utils.python_patch_extractor.PatchExtractor import PatchExtractor
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import datasets, models, transforms
import torch
import argparse
import os
from collections import OrderedDict


torch.manual_seed(21)
import random

random.seed(21)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')







def predict_single(input, model, device):


    

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
])
    img_net_scores=[]
    img =input
    stride_0 = ((((img.shape[0] - 32) // 20) + 7) // 8) * 8
    stride_1 = (((img.shape[1] - 32) // 10 + 7) // 8) * 8
    pe = PatchExtractor(dim=(32, 32, 3), stride=(stride_0, stride_1, 3))
    patches = pe.extract(img)
    patch_list = list(patches.reshape((patches.shape[0] * patches.shape[1], 32, 32, 3)))
    #print(len(patch_list))

    transf_patch_list = [ transform_test(Image.fromarray(patch)) for patch in patch_list]
    transf_patch_tensor = torch.stack(transf_patch_list, dim=0)

    #print(transf_patch_tensor.shape)

    input =  transf_patch_tensor.to(device)

   

    patch_scores = model(input).cpu().detach().numpy()
   # print(patch_scores)
   
    patch_predictions = np.argmax(patch_scores, axis=1)
    #print(patch_predictions)

    maj_voting = np.any(patch_predictions).astype(int)
    scores_maj_voting = patch_scores[:, maj_voting]
    img_net_scores.append(np.mean(scores_maj_voting) if maj_voting == 1 else -np.mean(scores_maj_voting))
    img_score = np.mean(img_net_scores)
    #print(img_score)
    return img_score



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to the input csv file')
    parser.add_argument('output_file', help='Path to the output csv file')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  
    else :
        device= torch.device('cpu')
    

    # img_path = args.img_path

    # detector = Detector()
    # score = detector.synth_real_detector(img_path)

    # print('Image Score: {}'.format(score))

    input_csv = args.input_file # input csv
    output_csv = args.output_file  # output csv

    root_db = os.path.dirname(input_csv) # directory of testset
    tab = pandas.read_csv(input_csv) # read input csv as table

    


    weights_path_list = [os.path.join('weigths_final', f'weight_{x}.pth') for x in 'ACF']
    model_ft_A= torch.load( weights_path_list[0],map_location='cpu').to(device)
    model_ft_C= torch.load( weights_path_list[1],map_location='cpu').to(device)
    model_ft_F= torch.load( weights_path_list[2],map_location='cpu').to(device)


    for index, dat in tqdm.tqdm(tab.iterrows(), total=len(tab)): # for on images
        filename = os.path.join(root_db, dat['filename']) # filepath of an image
        input_image =  np.asarray(Image.open(filename))
        score = (predict_single(input_image,model_ft_A,device)+predict_single(input_image,model_ft_C,device)+predict_single(input_image,model_ft_F,device))/3
        if(score>0):
            logit = int(1) # TODO compute the logit for the image
        else:
            logit = int(0)
        tab.loc[index,'logit'] = logit  # insert the logit in table

    tab.to_csv(output_csv, index=False) # save the results as csv file

    return 0

if __name__ == '__main__':
    main()
