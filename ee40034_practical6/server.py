from flask import Flask, request, render_template
import io
import base64
import glob
import pdb
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
from tqdm import tqdm

GPU_NUMBER = # (fill your assigned gpu number here) 
os.environ["CUDA_VISIBLE_DEVICES"]=f'{GPU_NUMBER}'
    
# configure app
app = Flask(__name__, template_folder='template')
app.config['SECRET_KEY'] = 'kaist123'                   # any random password
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # maximum input size in bytes

## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
## Load model - Practical 5 Section 9 (load from 'trained_softmax.model')
## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# (fill code here) - multiple lines of code

## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
## Dataset - Practical 5 Section 7 / 8 (load from 'ids' folder)
## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

class your_dataset(torch.utils.data.Dataset):
    # (fill code here) - multiple lines of code

val_transform = # (fill code here) 

# initialise the dataset for the cropped images in crop_path which is 'ids'
dataset = # (fill code here) 

# initialise the data loader with batch size of 1
loader  = # (fill code here) 


## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
## Face detection model - Practical 5 Section 4
## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# load face detection model
mtcnn = # (fill code here) 

## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
## Extract embeddings for all images in 'ids' - Practical 5 Section 10
## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

embeddings = {}

# (fill code here) - multiple lines of code - save in embeddings[FILE NAME] dict format


# check that below is about 19.4
print(f'Embedding norm check {torch.norm(embeddings['ids/AENDI.jpg'],p=2).item():.4f}')


def get_embeddings(fieldname):

    subm = request.form.get(fieldname)
    subfile = request.files[fieldname]

    # save to file
    filename = 'tmp.jpg'
    subfile.save(filename)
    image = Image.open(filename)
    image = ImageOps.exif_transpose(image)

    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    ## Crop uploaded image - Practical 5 Section 5
    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

    basewidth= 800
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize))

    bboxes = mtcnn.detect(image)

    ## this removes all images with no face detection or two or more face detections
    if bboxes[0] is None or len(bboxes[0]) != 1:
        print('No face detection on',fieldname,bboxes[0])
        return None, None

    # get bboxes confidence of face detection
    bbox = # (fill code here) 
    conf = # (fill code here) 

    # find the center and the box size of the bounding box
    sx = # (fill code here) 
    sy = # (fill code here) 
    ss = # (fill code here) 

    face = # (fill code here) 

    face_tensor = val_transform(face).unsqueeze(0).cuda()

    with torch.no_grad():
        # forward pass through the model to get prediction and probability
        embed = embednet(face_tensor).cpu()

    # get encoded image and save to memory
    data = io.BytesIO()
    face.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return embed, encoded_img_data


@app.route('/', methods=['GET', 'POST'])
def home():

    content = ''

    if request.method == 'POST':
        # if POST - compute embeddings and their distance

        embed1, enc_img1 = get_embeddings('subm1')

        if embed1 is None:
            content += 'There is a problem with your picture.<br>'
            return render_template("home.html", content=content)

        ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
        ## Compare embed1 to all 'embeddings' - Practical 5 Section 11
        ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

        ## find embedding means
        file_names  = # (fill code here) 
        embedding_matrix  = # (fill code here) 

        # find the cosine similarities between the embedding mean and all image embeddings and move to CPU
        sims = # (fill code here) 

        # sort the cosine similarities in ascending order
        sval, sidx = # (fill code here) 

        # get the image filenames in the order of ascending similarity
        sorted_files = # (fill code here) 

        ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
        ## Display the cropped image and most and leave similar images
        ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

        decoded_data = enc_img1.decode('utf-8')
        content += f'<img src="data:image/jpeg;base64,{decoded_data}" width="112"><br>\n'

        # we show the first 3 and the last 3 from the list
        files_to_show = sorted_files[:3] + sorted_files[-3:]

        content += f'<p>First 3 are the least similar, the last 3 are the most similar</p>\n'

        for file_to_show in files_to_show:

            image = Image.open(file_to_show)
            data = io.BytesIO()
            image.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())

            decoded_data = encoded_img_data.decode('utf-8')
            content += f'<img src="data:image/jpeg;base64,{decoded_data}" width="112">\n'
            content += f'{file_to_show}<br>\n'


        return render_template("home.html", content=content)

    else:
        # if GET - then just show template
        return render_template("home.html", content=content)

if __name__ == '__main__':


    app.run(host='0.0.0.0', port=8050+GPU_NUMBER)


      