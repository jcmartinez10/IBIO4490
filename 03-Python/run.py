#!/usr/bin/python3
import urllib.request
import zipfile
import shutil
import os
from os import listdir
from random import randint

import pip

def install(package):
    pip.main(['install', package])

try:
    import PIL
except ImportError:
    print ('Pillow is not installed, installing it now!')
    install('Pillow')

try:
    import matplotlib
except ImportError:
    print ('Matplotlib is not installed, installing it now!')
    install('matplotlib')


from PIL import ImageFont, ImageDraw, Image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if not os.path.isdir('./work_folder'):
    os.mkdir('./work_folder')

if not os.path.isdir('./lfwcrop_color'):
    if not os.path.isfile('./lfwcrop_color.zip'):
        urllib.request.urlretrieve('http://conradsanderson.id.au/lfwcrop/lfwcrop_color.zip', './lfwcrop_color.zip')
    zip_ref = zipfile.ZipFile('./lfwcrop_color.zip', 'r')
    zip_ref.extractall('./')
    zip_ref.close()

images = [f for f in listdir('./lfwcrop_color/faces')]
n=int(len(images))

fig,ax = plt.subplots(2,3, gridspec_kw = {'wspace':0, 'hspace':0})


for x in range(6):
    i = randint(0, n-1)
    image=Image.open('./lfwcrop_color/faces/'+images[i])
    resized=image.resize([256,256])
    resized.save("./work_folder/"+images[i]+"_resized.png")
    draw=ImageDraw.Draw(resized)
    txt = images[i]
    rest = txt.split('_0')[0]
    rest = rest.replace('_',' ')
    try:
        font = ImageFont.truetype("arial.ttf",30)
    except:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",30)
    w, h = draw.textsize(rest,font=font)
    if w>256:
        try:
            font = ImageFont.truetype("arial.ttf",25)
        except:
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",25)
        w, h = draw.textsize(rest,font=font)
    draw.text(((256-w)/2,100), rest, font=font) 
    ax[x%2][x//2].imshow(resized)
    ax[x%2][x//2].set_xticks([])
    ax[x%2][x//2].set_yticks([])
    
    
fig.show()
plt.show()
shutil.rmtree('./work_folder')
    

