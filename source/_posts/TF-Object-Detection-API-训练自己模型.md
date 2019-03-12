---
title: TF_Object_Detection API 训练自己模型
date: 2019-03-10 14:11:46

tags:
- Tensorflow

categories:
- Deep Learning

---



安装Tensorflow_Object_detection_API 依赖库

```
Protobuf 、Python-tk、Pillow 1.0、lxml、tf Slim、Jupyter notebook、Matplotlib、Tensorflow、Cython、cocoapi
```

具体请参考：

**https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md**

安装依赖库：(具体可参考官方文档)

下载源码：

```py 

git clone https://github.com/tensorflow/models
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip3 install Cython
sudo pip3 install jupyter
sudo pip3 install matplotlib

#或者使用pip安装：
sudo pip install Cython
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

> 如果使用COCO作为评价指标的话，需要接入coco的pythonApi，

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

#### 编译项目
From tensorflow/models/research/

首先protoc编译项目，然后添加环境变量 Mac端： ~./bash_profile

```
protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# 如果protoc版本过低，请对应环境下载 https://link.zhihu.com/?target=https%3A//github.com/google/protobuf/releases

sudo cp bin/protoc /usr/bin/protoc 再次尝试编译、添加环境
```
测试安装Ok：

`python3 object_detection/builders/model_builder_test.py`

## 如果返回Ok 则安装成功，运行setup 

`python3 setup.py install`

制作自己的数据集 并使用API传输训练
利用labelImag标注数据，生成xml信息，利用Xml-to-csv.py转换成voc的格式，xml-to-csv脚本：

注意按照自己的文件结构对应修改，**我的结构**：

```py

-train_data/
  --...
-images/
  --test/
    ---testingimages.jpg
    ---image.xml
  --train/
    ---testingimages.jpg
    ---image.xml
  --..yourimages.jpg
-xml_to_csv.py

```

```
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET



def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('train_data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

main()
```

### 将Csv格式的图片信息转换为tf_record格式，提供API训练

> 首先将上述的images、data移到model/research/object_detedtion文件夹下：利用generate_tfrecord.py转换格式

> 需要修改 返回的类别和名称 以及文件路径名

`https://github.com/junqiangwu/My_Tensorflow/blob/master/object-detection/generate_tfrecord.py`

**From model/research/object_detection/**

```

from __future__ import division  
from __future__ import print_function  
from __future__ import absolute_import  
  
import os  
import io  
import pandas as pd  
import tensorflow as tf  
  
from PIL import Image  
from object_detection.utils import dataset_util  
from collections import namedtuple, OrderedDict  
  
flags = tf.app.flags  
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')  
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')  
FLAGS = flags.FLAGS  
  
  
# TO-DO replace this with label map  
def class_text_to_int(row_label):  
    if row_label == 'macncheese':  
        return 1  
    else:  
        None  
  
  
def split(df, group):  
    data = namedtuple('data', ['filename', 'object'])  
    gb = df.groupby(group)  
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]  
  
  
def create_tf_example(group, path):  
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:  
        encoded_jpg = fid.read()  

    encoded_jpg_io = io.BytesIO(encoded_jpg)  
    image = Image.open(encoded_jpg_io)  
    width, height = image.size  
  
    filename = group.filename.encode('utf8')  
    image_format = b'jpg'  
    xmins = []  
    xmaxs = []  
    ymins = []  
    ymaxs = []  
    classes_text = []  
    classes = []  
  
    for index, row in group.object.iterrows():  
        xmins.append(row['xmin'] / width)  
        xmaxs.append(row['xmax'] / width)  
        ymins.append(row['ymin'] / height)  
        ymaxs.append(row['ymax'] / height)  
        classes_text.append(row['class'].encode('utf8'))  
        classes.append(class_text_to_int(row['class']))  
  
    tf_example = tf.train.Example(features=tf.train.Features(feature={  
        'image/height': dataset_util.int64_feature(height),  
        'image/width': dataset_util.int64_feature(width),  
        'image/filename': dataset_util.bytes_feature(filename),  
        'image/source_id': dataset_util.bytes_feature(filename),  
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),  
        'image/format': dataset_util.bytes_feature(image_format),  
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),  
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),  
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),  
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),  
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),  
        'image/object/class/label': dataset_util.int64_list_feature(classes),  
    }))  
    return tf_example  
  
  
def main(_):  
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)  
    path = os.path.join(os.getcwd(), 'images')  
    
    examples = pd.read_csv(FLAGS.csv_input)  
    grouped = split(examples, 'filename')  

    num=0  
    for group in grouped:  
        num+=1  
        tf_example = create_tf_example(group, path)  
        writer.write(tf_example.SerializeToString())  
        if(num%100==0):  #每完成100个转换，打印一次  
            print(num)  
  
    writer.close()  
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)  
    print('Successfully created the TFRecords: {}'.format(output_path))  
  
  
if __name__ == '__main__':  
    tf.app.run()  
```

```

python3 generate_tfrecord.py --csv_input=train_data/train_labels.csv  --output_path=train.record 

python3 generate_tfrecord.py --csv_input=train_data/test_labels.csv  --output_path=test.record 

```
> 会在object_detection目录下生成两个.record文件，将它移到train_data目录下，train_data目录下包含：两个csv 和 两个 .record

## 在object_detection目录下:

```py

-images/
 --test/
  ---testingimages.jpg
 --train/
  ---testingimages.jpg
 --..yourimages.jpg						
-train_data	
 --train_labels.csv
 --test_labels.csv
 --train.record
 --test.record

```

### 下载预训练模型，配置网络结构信息：
```
wget http://download.tensorflow.org/models/object_detection/ ssd_mobilenet_v1_coco_11_06_2017.tar.gz

mkdir training 
在training文件夹下编写训练数据标签：object_detection.pbtxt
			item {
				  id: 1
				  name: 'macncheese' #物品类别
				 }
```

> 从`object_detection/samples/config/ssd_mobilenet_v1_pets.config`移到training文件下：并作出修改： 
> 
> `num_class: 1 batch_size: 24 fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"`

```
train_input_reader: {
	    tf_record_input_reader {
		input_path: "train_data/train.record"
		}
		label_map_path: "training/object-detection.pbtxt"
		}
```

最后在object_detection文件夹下：运行命令：

```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```
> train_dir: 训练输出文件的路径 

>pipeline_config: 网络配置文件的路径

> 测试输出模型的准确性  利用.py 转换 .pb 
> 


**From model/research/object_detection**

```
python3 export_inference_graph.py
 
--input_type image_tensor 
--pipeline_config_path training/ssd_mobilenet_v1_pets.config 
--trained_checkpoint_prefix training/model.ckpt-388 
--output_directory mac_n_cheese_inference_graph
```
- input_type : 保持一致
- pipeline: 网络结构配置图
- train_checkpoint: ckpt模型保存路径 既上面训练路径的设置位置
- out: 输出文件



####最后利用jupyter notebook加载pb模型进行测试
```
#修改object_detection_tutorial.ipynb

# What model to download.
MODEL_NAME = 'mac_n_cheese_inference_graph'

#Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1

#删除downloand程序，修改加载测试图片的路径，运行即可
```

所有的配置文件在：

https://github.com/junqiangwu/My_Tensorflow/tree/master/object-detection