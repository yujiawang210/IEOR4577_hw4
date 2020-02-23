IEOR 4577 Assignment 04
Xinyue Tian (xt2230), 
Yujia Wang (yw3400), 
Qian Wu (qw2306)

Requirement:
python3==3.7.3
tensorflow==1.14
numpy==1.16.2
boto3

Instruction:
To run the model training code in terminal, please use the following command but replace file directory
python3 sentiment_training.py --train '/.../IEOR4577_hw4/glue job outputs/train' --validation '/.../IEOR4577_hw4/glue job outputs/dev' --eval '/.../IEOR4577_hw4/glue job outputs/eval' --model_output_dir '/.../Assignment 4/IEOR4577_hw4/model output'
