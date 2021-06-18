from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
import dlib
from PIL import Image
import numpy as np
import pandas as pd
import math

# Number of style channels per StyleGAN layer
style2list_len = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 
                  512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 
                  128, 64, 64, 64, 32, 32]

# Layer indices of ToRGB modules
rgb_layer_idx = [1,4,7,10,13,16,19,22,25]

google_drive_paths = {
    "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "inversion_stats.npz": "https://drive.google.com/uc?id=1oE_mIKf-Vr7b3J04l2UjsSrxZiw-UuFg",
    "model_ir_se50.pt": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
}


def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

# given a list of filenames, load the inverted style code
@torch.no_grad()
def load_source(files, generator):
    sources = []
    
    for file in files:
        source = torch.load(f'./inversion_codes/{file}.pt')['latent']

        if source.size(0) != 1:
            source = source.unsqueeze(0)

        if source.ndim == 3:
            source = generator.get_latent(source, truncation=1, is_latent=True)
            
        sources.append(source)
        
    sources = torch.cat(sources, 0)
    if type(sources) is not list:
        sources = style2list(sources)
        
    return sources

'''
Given M, we zero out the first 2048 dimensions for non pose or hair features.
The reason is that the first 2048 mostly contain hair and pose information and rarely
anything related to other classes.

'''
def remove_2048(M, labels2idx):
    M_hair = M[:,labels2idx['hair']].clone()
    # zero out first 2048 channels (4 style layers) for non hair and pose features
    M[...,:2048] = 0
    M[:,labels2idx['hair']] = M_hair
    return M

# Compute pose M and append it as the last index of M
def add_pose(M, labels2idx):
    M = remove_2048(M, labels2idx)
    # Add pose to the very last index of M
    pose = 1-M[:,labels2idx['hair']]
    M = torch.cat([M, pose.view(-1,1,9088)], 1)
    #zero out rest of the channels after 2048 as pose should not affect other features
    M[:,-1, 2048:] = 0
    return M


# add direction specified by q from source to reference, scaled by a
def add_direction(s, r, q, a):
    if isinstance(s, list):
        s = list2style(s)
    if isinstance(r, list):
        r = list2style(r)
    if s.ndim == 1:
        s = s.unsqueeze(0)
    if r.ndim == 1:
        r = r.unsqueeze(0)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if len(s) != len(r):
        s = s.expand(r.size(0), -1)
    q = q.float()
        
    old_norm = (q*s).norm(2,dim=1, keepdim=True)+1e-8
    new_dir = q*r
    new_dir = new_dir/(new_dir.norm(2,dim=1, keepdim=True)+1e-8) * old_norm
    return s -a*q*s + a*new_dir


# convert a style vector [B, 9088] into a suitable format (list) for our generator's input
def style2list(s):
    output = []
    count = 0 
    for size in style2list_len:
        output.append(s[:, count:count+size])
        count += size
    return output

# convert the list back to a style vector
def list2style(s):
    return torch.cat(s, 1)

# flatten spatial activations to vectors
def flatten_act(x):
    b,c,h,w = x.size()
    x = x.pow(2).permute(0,2,3,1).contiguous().view(-1, c) # [b,c]
    return x.cpu().numpy()

def show(imgs, title=None):

    plt.figure(figsize=(5 * len(imgs), 5))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)

    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0.02)

def part_grid(target_image, refernce_images, part_images):
    def proc(img):
        return (img * 255).permute(1, 2, 0).squeeze().cpu().numpy().astype('uint8')

    rows, cols = len(part_images) + 1, len(refernce_images) + 1
    fig = plt.figure(figsize=(cols*4, rows*4))
    sz = target_image.shape[-1]

    i = 1
    plt.subplot(rows, cols, i)
    plt.imshow(proc(target_image[0]))
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.title('Source', fontdict={'size': 26})

    for img in refernce_images:
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(img))
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.title('Reference', fontdict={'size': 26})

    for j, label in enumerate(part_images.keys()):
        i += 1
        plt.subplot(rows, cols, i)
        plt.imshow(proc(target_image[0]) * 0 + 255)
        plt.text(sz // 2, sz // 2, label.capitalize(), fontdict={'size': 30})
        plt.axis('off')
        plt.gca().set_axis_off()

        for img in part_images[label]:
            i += 1
            plt.subplot(rows, cols, i)
            plt.imshow(proc(img))
            plt.axis('off')
            plt.gca().set_axis_off()

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.subplots_adjust(wspace=0, hspace=0)
    return fig


def display_image(image, size=256, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size,size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = ((image.clamp(-1,1)+1)/2).permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)

def get_parsing_labels():
    color = torch.FloatTensor([[0, 0, 0],
                      [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                      [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                      [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192,128,128],
                      [0, 64, 0], [0, 0, 64], [128, 0, 192], [0, 192, 128], [64,128,192], [64,64,64]])
    return (color/255 * 2)-1

def decode_segmap(seg):
    seg = seg.float()
    label_colors = get_parsing_labels()
    r = seg.clone()
    g = seg.clone()
    b = seg.clone()

    for l in range(label_colors.size(0)):
        r[seg == l] = label_colors[l, 0]
        g[seg == l] = label_colors[l, 1]
        b[seg == l] = label_colors[l, 2]

    output = torch.stack([r,g,b], 1)
    return output

def remove_idx(act, i):
    # act [N, 128]
    return torch.cat([act[:i], act[i+1:]], 0)

def interpolate_style(s, t, q):
    if isinstance(s, list):
        s = list2style(s)
    if isinstance(t, list):
        t = list2style(t)
    if s.ndim == 1:
        s = s.unsqueeze(0)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if len(s) != len(t):
        s = s.expand(t.size(0), -1)
    q = q.float()
        
    return (1 - q) * s + q * t
    
def index_layers(w, i):
    return [w[j][[i]] for j in range(len(w))]


def normalize_im(x):
    return (x.clamp(-1,1)+1)/2

def l2(a, b):
    return (a-b).pow(2).sum(1)

def cos_dist(a,b):
    return -F.cosine_similarity(a, b, 1)

def downsample(x):
    return F.interpolate(x, size=(256,256), mode='bilinear')
def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detectFace(img, face_detector, dilate=False):
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    #print("found faces: ", len(faces))

    if len(faces) > 0:
        face = faces[0]
        face_x, face_y, face_w, face_h = face

        if dilate:
            scale_v = face_h / 3
            face_y -= scale_v
            face_h += 1.5*scale_v

            scale_h = 1.5*scale_v 
            face_x -= 0.5*scale_h
            face_w += scale_h

        img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img, img_gray
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray
        #raise ValueError("No face found in the passed image ")

def alignFace(img_path, face_detector, eye_detector):
    img = cv2.imread(img_path)
    img_raw = img.copy()

    img, gray_img = detectFace(img, face_detector)
    
    eyes = eye_detector.detectMultiScale(gray_img)
    
    #print("found eyes: ",len(eyes))
    
    if len(eyes) >= 2:
        #find the largest 2 eye
        
        base_eyes = eyes[:, 2]
        #print(base_eyes)
        
        items = []
        for i in range(0, len(base_eyes)):
            item = (base_eyes[i], i)
            items.append(item)
        
        df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
        
        eyes = eyes[df.idx.values[0:2]]
        
        #--------------------
        #decide left and right eye
        
        eye_1 = eyes[0]; eye_2 = eyes[1]
        
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
        
        #--------------------
        #center of eyes
        
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        
        #----------------------
        #find rotation direction
        
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock
                
        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, point_3rd)
        c = euclidean_distance(right_eye_center, left_eye_center)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)
        
        angle = (angle * 180) / math.pi
        if direction == -1:
            angle = 90 - angle
                
        #--------------------
        #rotate image
        
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle, resample=2))
    
    return new_img
    
def align_face(filepath):
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path+"/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path+"/data/haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ",detector_path," violated.")

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path) 

    aligned_face = alignFace(filepath, face_detector, eye_detector)
    img, gray_img = detectFace(aligned_face, face_detector, True)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

