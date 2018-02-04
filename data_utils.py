import numpy as np
import sys
import os
import csv
from scipy import misc
import collections
import tensorflow as tf
from tensorflow.contrib import rnn

import pickle

import _pickle as cPickle

try:
	import cPickle as pickle
except ImportError:
	import pickle


train_path = "./data/mfcc/train.ark"
test_path = "./data/mfcc/test.ark"
label_path = "./label/train.lab"
test_path = "./data/mfcc/test.ark"
prepro_dir = "./prepo/"
map48_dir = "./data/phones/48phone_char.map"
map48to39_dir = "./data/phones/48_39.map"

testlabel = "./testlabel.txt"
testlabelmap = "./data/label/testlabelmap.txt"

filename = "./hw1.spydata"

model_dir = "./model/"

batch_size = 128
pointer = 0



#file = open(map48_dir, 'r')
##for r_index, row in file.readlines():
#r_array_map = []
#
#for r_index, row in enumerate(file.readlines()):
##    if (r_index < 10):
##        print(row, end='')
#    r_temp = row.split('\t')
#    r_array_map.append(r_temp)

#######################################################

def load_train_data(train_path, label_path, map48_dir, map48to39_dir, testlabelmap):

#######################################################
## open the file of training data and delete the name of ppl    
    file = open(train_path, 'r')
    #for r_index, row in file.readlines():
    t_array_rnn = []
    r_array = []
    
    for r_index, row in enumerate(file.readlines()):
        temprnn = row.split()[0]
        t_array_rnn.append(temprnn)
        
        
        r_temp = row.split()
        for k in range(1, len(r_temp),1):
            r_temp[k] = float(r_temp[k])
        r_array.append(r_temp)
        
    for i in range(0, len(r_array),1):
        del r_array[i][0]
        
    
    ## separate every ppl's array from id 1 to the end so we got 3698 ppl (3698,?,39)        
    t_array_final = [] 
    flag_final = False   
    temp_final = []
    for m in range(0,len(t_array_rnn),1):
        temprnnn = t_array_rnn[m].split('_')
        
        if (temprnnn[2] == '1'):
            flag_final = True
            
        if (flag_final == True):
            t_array_final.append(temp_final)
            temp_final = []
            flag_final = False
            
        temp_final.append(r_array[m])
        
        if (m == len(t_array_rnn)-1):
            t_array_final.append(temp_final)
    
    del t_array_final[0]
        
    ######################################################
    
    ## add the file of map 48 convert 39 and 48 phone and ENG letters     
    file = open(map48_dir, 'r')
    #for r_index, row in file.readlines():
    r_array_map = []
    
    for r_index, row in enumerate(file.readlines()):
    #    if (r_index < 10):
    #        print(row, end='')
        r_temp = row.split('\t')
        r_array_map.append(r_temp)
    
    
    file39 = open(map48to39_dir, 'r')
    #for r_index, row in file.readlines():
    r_array_map39 = []
    
    for r_index39, row39 in enumerate(file39.readlines()):
    #    if (r_index < 10):
    #        print(row, end='')
        r_temp39 = row39.split('\t')
        r_array_map39.append(r_temp39)
    
    
    
       
    
    #convert labelmap to 48 char
    file = open(testlabelmap, 'r')
    #for r_index, row in file.readlines():
    r_label_map = []
    
    for r_index, row in enumerate(file.readlines()):
    #    if (r_index < 10):
    #        print(row, end='')
        r_temp = row.split('\n')[0]
    #    r_temp[1] = int('0')
    #    r_temp = int(r_temp)
        r_label_map.append(r_temp)
        
    pre_48char = [] #convert integer 0-47 to 48 char
    for i in range(0,len(r_label_map),1):
        for j in range(0,len(r_array_map),1):
            if (r_label_map[i] == r_array_map[j][1]):
                pre_48char.append(r_array_map[j][0]) 
    ## convert 48 phones to 39 phones
    pre_48char39 = []            
    for e in range(0,len(pre_48char),1):
        for f in range(0,len(r_array_map39),1):
            if (pre_48char[e] == r_array_map39[f][0]):
                pre_48char39.append(r_array_map39[f][1])
                
    # get the 48 phones but only have 39 phones (some phones are duplicated)
    rmap39 = []
    for a in range(0,len(r_array_map39)):
        temp39 = r_array_map39[a][1]
        temp39a = temp39.split('\n')[0]
        rmap39.append(temp39a)
    
    
    ############################################
    
    
    #convert 48 char to 39 char            
    a = rmap39
    map39_afterpre = []
    temp_final1 = []
    rmap39a1 = []
    rmapc = 0
    
    ## convert 48 phone but only have 39 phones into only 39 phone without same phones
    for aa in range(0,len(rmap39),1):
        for bb in range(0,len(map39_afterpre),1):
            if (rmap39[aa] == map39_afterpre[bb]):
                rmapc = rmapc + 1
    
        if (rmapc < 1):    
            map39_afterpre.append(rmap39[aa])
            rmapc = 0
        else:
            rmapc = 0
            
    mapxx = map39_afterpre
    ##turn the 39 phones into the number of its index
    pre_39final = []            
    for cc in range(0,len(pre_48char39),1):
        for dd in range(0,len(map39_afterpre),1):
            temp39 = pre_48char39[cc]
            temp39a = temp39.split('\n')[0]
            if (temp39a == map39_afterpre[dd]):
                pre_39final.append(int(dd))
                
    
    #############################################
    #convert label to be like array formation 
    ##(alian ppl's id with different lens sentences of label)
    pmap = 0
    ch = 0
    label_array_final1 = []
    for ee in range(0,len(t_array_final),1):
        
        ch = len(t_array_final[ee])
        tempmap = pre_39final[pmap:pmap+ch]
        pmap = pmap + ch
    #    pmap = pmap + 1
        label_array_final1.append(tempmap)
        
        
    ####################################    
    #pedding
    ##calculate max len of array
    max_final = len(t_array_final[0])   
    for ff in range(0,len(t_array_final),1):
        if(max_final < len(t_array_final[ff])):
            max_final = len(t_array_final[ff])        
    
    #pedding for array to make every sentence has same length
    ##we use the max length that every sentence padds to be like max length    
    t_array_f = t_array_final
    t_array_f1 = []
    pedd_f = []
    for gg1 in range(0,len(t_array_f[0][0]),1):
        temppeddf = float(1000.0)
        pedd_f.append(temppeddf)
    
    for gg in range(0,len(t_array_final),1):
        changefinal = max_final - len(t_array_final[gg])
        for hh in range(0,changefinal,1):
            temparrayf = t_array_f[gg]
            temparrayf.append(pedd_f)
        t_array_f1.append(temparrayf)
    
    
    #pedding for label to make every sentence has same length
    ##we use the max length that every sentence padds to be like max length  
    t_label_f = label_array_final1
    label_array_f1 = []
    pedd_flabel = []
    
    temppeddf = int(100)
    pedd_flabel = temppeddf
    temparrayflabel = []
    
    for hh in range(0,len(t_label_f),1):
        changefinallabel = max_final - len(t_label_f[hh])
        for ii in range(0,changefinallabel,1):
            temparrayflabel = t_label_f[hh]
            temparrayflabel.append(pedd_flabel)
        label_array_f1.append(temparrayflabel)    
        
        
    ################################################
    #processing the array and label to be the input shape of RNN model
    
    x = t_array_f1
    xinput = np.array(x)
    
    y = label_array_f1
    #yinput = np.array(y)
    #yinput = np.array(y1)
    
    #del y1[0][48:97]
    
    tempp = []
    #for k in range(0,49,1):
    #    tempp1 = int(0)
    #    tempp.append(tempp1)
    y1 = []
    temppp = []    
    # we use one hot encoding to the labels (let it to be like (00001000000000))
    for i in range(0,len(y),1):
        for j in range(0,len(y[0]),1):
            for k in range(0,40,1):
                tempp1 = int(0)
                tempp.append(tempp1)
            if (y[i][j]<39):
                tempp[y[i][j]] = int(1)
            else:
                tempp[39] = int(1)
            temppp.append(tempp)
            tempp = []
        y1.append(temppp)
        temppp = []
    
    yinput = np.array(y1)
    
    
    return xinput, yinput

############################################



    