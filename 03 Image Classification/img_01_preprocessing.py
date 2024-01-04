from preprocess import training_data_generator

DIRECTORY = "data/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32

#Creates a DirectoryIterator object using the above parameters:

training_iterator = training_data_generator.flow_from_directory(DIRECTORY,class_mode=CLASS_MODE,color_mode=COLOR_MODE,target_size=TARGET_SIZE,batch_size=BATCH_SIZE)

sample_batch_input,sample_batch_labels  = training_iterator.next()

print(sample_batch_input.shape,sample_batch_labels.shape)

'''
Output is: (32, 256, 256, 1) (32, 2)
This is because there are 32 images in a batch, and each is a 256x256 pixel grayscale image. The last dimension is the number of channels. 
Because these are grayscale images, there is only one channel representing light intensity.

Sample_batch_labels should be shape of (32,2), because an image can be labeled Normal ([1,0]) or Pneumonia ([0,1]).
'''