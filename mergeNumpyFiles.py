import os
import numpy as np
path = './Chirag/'
trainImages = []
for i in os.listdir(path):
  data = np.load(path+i)
  trainImages.extend(data)

np.save("Image_data/Chirag", trainImages)