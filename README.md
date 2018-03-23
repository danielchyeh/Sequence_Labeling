# Sequence-Labeling
MLDS2017 Project 1

Project Link: https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A1
## Dataset
TIMIT dataset, which has MFCC features (39 dims) and FBank features (69 dims) for each frame is used in this task. It also has 48 different kinds of phone.

TIMIT dataset can be downloaded from kaggle https://www.kaggle.com/c/hw1-timit/data (created by class MLDS2017, NTU)
## Quick start
Run the shell script
```
./hw1_best.sh [input directory] [output filename]
```
or any of the following: `hw1_cnn.sh`, `hw1_rnn.sh`

[input directory]
should be TIMIT dataset which you download from link above

[output filename] 
a .csv file which shows the result of prediction


