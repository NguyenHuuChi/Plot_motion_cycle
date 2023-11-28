import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense,Flatten,Activation, add
from tensorflow.keras.models import Model
tf.config.run_functions_eagerly(True)
class CNNClassifier(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.maxpool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.fc = Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output
def separate_data(data_frame):
    less=[]
    more=[]
    numpy_array = data_frame.numpy()

    # Convert the NumPy array to a Python list
    python_list = numpy_array.tolist()
    for data in python_list:
        lessimportant=data[:-1]
        moreimportant=data[-1]
        a=[[-1]*len(moreimportant[0])]*len(moreimportant)
        b=[[[-1]*len(lessimportant[0][0])]*len((lessimportant[0]))]*len(lessimportant)
        lessimportant.append(a)
        b.append(moreimportant)
        less.append(lessimportant)
        more.append(b)
    less= tf.convert_to_tensor(less, dtype=tf.float32)
    more=tf.convert_to_tensor(more, dtype =tf.float32)
    return less,more
# def separate_data(data_frame):
#     less = []
#     more = []

#     # Assuming data_frame is a TensorFlow tensor

#     # Get the number of columns (assuming data_frame shape is (batch_size, num_columns))
#     num_columns = data_frame.shape[-1]

#     for i in range(num_columns):
#         # Split each column into 'less' and 'more'
#         less_column = data_frame[:, :-1, i:i+1]
#         more_column = data_frame[:, -1:, i:i+1]

#         less.append(less_column)
#         more.append(more_column)

#     # Stack 'less' and 'more' tensors along a new axis
#     less = tf.stack(less, axis=-1)
#     more = tf.stack(more, axis=-1)

#     return less, more

class ResNet_important(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet_important, self).__init__()

        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')
    def identity_block(self, input_tensor, kernel_size, filters):
        filters1, filters2, filters3 = filters
        lesim, moreim = separate_data(input_tensor)
        def block_fn():
            x = Conv2D(filters1, strides=(1, 1),kernel_size=3, padding='same')(lesim)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters2, kernel_size=kernel_size,strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters3, kernel_size=kernel_size,strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)

            shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=(1,1))(moreim)
            shortcut = BatchNormalization()(shortcut)
            
            x_out = tf.keras.layers.add([x, shortcut])
            x_out = ReLU()(x_out)
            return x_out
        
        x = block_fn()
        return x
    def conv_block(self, input_tensor,kernel_size,filters, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        lesim, moreim = separate_data(input_tensor)
        def block_fn():
            x = Conv2D(filters1,kernel_size= (1, 1), strides=strides)(lesim)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters2, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters3, (1, 1))(x)
            x = BatchNormalization()(x)
            
            shortcut = Conv2D(filters3, (1, 1), strides=strides)(moreim)
            shortcut = BatchNormalization()(shortcut)

            x_out = tf.keras.layers.add([x, shortcut])
            x_out = ReLU()(x_out)
            return x_out
        
        x = block_fn()
        return x

    def call(self, inputs):
        
        x = self.conv1(inputs)  
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.conv_block(x,3, [64, 64, 256])
        x = self.identity_block(x, 3, [64, 64, 256])
        x = self.conv_block(x,3, [128, 128, 512])
        x = self.identity_block(x, 3, [128, 128, 512])
        x = self.conv_block(x,3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.avgpool(x)
       
        x = self.fc(x)

        return x
class ResNet50(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')

    def identity_block(self, input_tensor, kernel_size, filters):
        filters1, filters2, filters3 = filters
       
        def block_fn(input_tensor):
            x = Conv2D(filters1, strides=(1, 1),kernel_size=3, padding='same')(input_tensor)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters2, kernel_size=kernel_size,strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters3, kernel_size=kernel_size,strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)

            shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=(1,1))(input_tensor)
            shortcut = BatchNormalization()(shortcut)
            
            x_out = tf.keras.layers.add([x, shortcut])
            x_out = ReLU()(x_out)
            return x_out
        
        x = block_fn(input_tensor)
        return x

    def conv_block(self, input_tensor,kernel_size,filters, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        def block_fn(input_tensor):
            x = Conv2D(filters1,kernel_size= (1, 1), strides=strides)(input_tensor)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters2, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters3, (1, 1))(x)
            x = BatchNormalization()(x)
            
            shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
            shortcut = BatchNormalization()(shortcut)

            x_out = tf.keras.layers.add([x, shortcut])
            x_out = ReLU()(x_out)
            return x_out
        
        x = block_fn(input_tensor)
        return x

    def call(self, inputs):
        
        x = self.conv1(inputs)  
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.conv_block(x,3, [64, 64, 256])
        x = self.identity_block(x, 3, [64, 64, 256])
        x = self.conv_block(x,3, [128, 128, 512])
        x = self.identity_block(x, 3, [128, 128, 512])
        x = self.conv_block(x,3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.avgpool(x)
       
        x = self.fc(x)

        return x


