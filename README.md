# TSCM
This is the code repositary for our ICRA submission: TSCM: A Teacher-Student Model for Vision Place Recognition Using Cross-Metric Knowledge Distillation
Thanks to the open source work of [baseline](https://github.com/ramdrop/stun), the code of TSCM is based on it.
## Pittsburgh Dataset
You can download the pittsburgh dataset on https://www.dropbox.com/s/ynep8wzii1z0r6h/pittsburgh.zip?dl=0
## How to use
If you want to verify the effect, you can download the [stu_30k.pickle](https://www.dropbox.com/scl/fi/2rad0vkf0fd2v9er10g2v/stu_30k.pickle?rlkey=b04iygbqlsspt1upkr9jjyjaj&dl=0) file and put it in the folder /logs/contrast/ ,run the following code
```shell
python vis.py
```

## Train the pretrained models
In training mode
```shell
self.parser.add_argument('---split', type=str, default='val', help='Split to use', choices=['val', 'test'])
```
```shell
# train the teacher net
python main.py --phase=train_tea

# train the student net supervised by the pretrained teacher net
python main.py --phase=train_stu --resume=[teacher_net_xxx/ckpt_best.pth.tar]
```
## Evaluate the pretrained models
In test mode
```shell
self.parser.add_argument('---split', type=str, default='val', help='Split to use', choices=['val', 'test'])
```
 needs to be changed to 
 ```shell
self.parser.add_argument('---split', type=str, default='test', help='Split to use', choices=['val', 'test'])
```
You can run [pre-trained models](https://www.dropbox.com/scl/fo/c7why2nf82gn1ffv6dsr7/h?rlkey=7ariswrhfecaezjh0xz40599i&dl=0).The teacher_triplet/ckpt.pth.tar in the code needs to be changed to the appropriate name.If this pre-training model doesn't work or you need more pre-training models, please contact me in Issue.
```shell
python main.py --phase=test_stu	 --resume=logs/teacher_triplet/ckpt.pth.tar
```
