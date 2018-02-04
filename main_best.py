# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:49:31 2017

@author: Daniel Yeh 
"""

from keras.models import load_model
import numpy as np
import operator
import os
import sys

import csv
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.layers import Input,Dense,LSTM,TimeDistributed,Activation,Dropout,Conv1D,BatchNormalization
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import RepeatVector


#test_path = "./mfcc/test.ark"
#map48to39_dir = "./48_39.map"
#map48_dir = "./48phone_char.map"

test_path = sys.argv[1] + 'mfcc/test.ark'
map48to39_dir = sys.argv[1] + "phones/48_39.map"
map48_dir = sys.argv[1] + "phones/48phone_char.map"

result_csv = sys.argv[2]


##################################################
file = open(test_path, 'r')
#for r_index, row in file.readlines():
te_array_rnn = []
r_array = []

for r_index, row in enumerate(file.readlines()):
    temprnn = row.split()[0]
    te_array_rnn.append(temprnn)
    
    
    r_temp = row.split()
    for k in range(1, len(r_temp),1):
        r_temp[k] = float(r_temp[k])
    r_array.append(r_temp)
    
for i in range(0, len(r_array),1):
    del r_array[i][0]
       
    
te_array_final = [] 
te_id_final = [] 
flag_final = False   
temp_final = []
for m in range(0,len(te_array_rnn),1):
    temprnnn = te_array_rnn[m].split('_')
    
    if (temprnnn[2] == '1'):
        flag_final = True
        te_id_final.append((temprnnn[0]+'_'+temprnnn[1]))
        
    if (flag_final == True):
        te_array_final.append(temp_final)
        temp_final = []
        flag_final = False
        
    temp_final.append(r_array[m])
    
    if (m == len(te_array_rnn)-1):
        te_array_final.append(temp_final)

del te_array_final[0]

###########################################

# open 48 and ENG letters file
file = open(map48_dir, 'r')
#for r_index, row in file.readlines():
r_array_map = []

for r_index, row in enumerate(file.readlines()):
#    if (r_index < 10):
#        print(row, end='')
    r_temp = row.split('\t')
    r_array_map.append(r_temp)


##########################################
#Generate aaa which is 39 phone vector
file39 = open(map48to39_dir, 'r')
#for r_index, row in file.readlines():
r_array_map39 = []

for r_index39, row39 in enumerate(file39.readlines()):
#    if (r_index < 10):
#        print(row, end='')
    r_temp39 = row39.split('\t')
    r_array_map39.append(r_temp39)


rmap39 = []
for a in range(0,len(r_array_map39)):
    temp39 = r_array_map39[a][1]
    temp39a = temp39.split('\n')[0]
    rmap39.append(temp39a)
    
    
a = rmap39
aaa = []
temp_final1 = []
rmap39a1 = []
rmapc = 0

for aa in range(0,len(rmap39),1):
    for bb in range(0,len(aaa),1):
        if (rmap39[aa] == aaa[bb]):
            rmapc = rmapc + 1

    if (rmapc < 1):    
        aaa.append(rmap39[aa])
        rmapc = 0
    else:
        rmapc = 0

###########################################

#pedding
##calculate max len of array
max_final = len(te_array_final[0])   
for ff in range(0,len(te_array_final),1):
    if(max_final < len(te_array_final[ff])):
        max_final = len(te_array_final[ff])        

max_final = 777 #fit the model

#pedding for array    
t_array_f = te_array_final

te_array_f1 = []
pedd_f = []
for gg1 in range(0,len(t_array_f[0][0]),1):
    temppeddf = float(0.0)
    pedd_f.append(temppeddf)

for gg in range(0,len(te_array_final),1):
    changefinal = max_final - len(te_array_final[gg])
    for hh in range(0,changefinal,1):
        temparrayf = t_array_f[gg]
        temparrayf.append(pedd_f)
    te_array_f1.append(temparrayf)

###########################################

test_x = te_array_f1
test_xinput = np.array(test_x)


#load the model
model = load_model('mymodel_best.h5')

#predict model
predict_te = model.predict(test_xinput)

##########################################
#convert prediction to list

pre_te = predict_te.tolist()

#get rid of the pedding index
te_array_pre = [] 
flag_final = False   
temp_final = []
for m in range(0,len(te_array_rnn),1):
    temprnnn = te_array_rnn[m].split('_')
    
    if (temprnnn[2] == '1'):
        flag_final = True
        
    if (flag_final == True):
        te_array_pre.append(temp_final)
        temp_final = []
        flag_final = False
        
    temp_final.append(r_array[m])
    
    if (m == len(te_array_rnn)-1):
        te_array_pre.append(temp_final)

del te_array_pre[0]

pre_tenopad = []
temp_nopa = []

for i in range(0,len(te_array_pre),1):
    for j in range(0,len(te_array_pre[i]),1):
        temp_nopa.append(pre_te[i][j])
    pre_tenopad.append(temp_nopa)
    temp_nopa = []
#        if (j > len(te_array_pre[i])):
#            del pre_te[i][j]
#get the max of every vector
pre_teno = []
tempmax = []
for k in range(0,len(pre_tenopad),1):
    for l in range(0,len(pre_tenopad[k]),1):
        index, value = max(enumerate(pre_tenopad[k][l]), key=operator.itemgetter(1))
        tempmax.append(index)
    pre_teno.append(tempmax)
    tempmax = []
    

##change the number to be 39 character  
pre_teno39char = []
tempteno39 = []    
for m in range(0,len(pre_teno),1):
    for n in range(0,len(pre_teno[m]),1): 
        tempteno39.append(aaa[pre_teno[m][n]])
    pre_teno39char.append(tempteno39)
    tempteno39 = []


#########################################
#Trim to remove the duplicated phones like aaabbbcc to be abc    
pre_final = []
temp_final1 = []
pre_finalbi = []
te = []


bi_code = []

for x in range(0,len(pre_teno39char),1):#len(pre_teno39char)
    
    temp_final = pre_teno39char[x]
    for y in range(0,len(temp_final),1):
        temp0 = temp_final[y]
        temp1 = temp0.split('\n')[0]
        temp_final1.append(temp1)
    
#    a = ['sli','sli','sli','sli','a','a','a','b','b','a','d','d','w','sli','c','w','sli','sli','sli']
    a0 = temp_final1
    temp_final1 = []
    a1 = []
    a2 = []
    c = 0
    d = -1
    
    a0.append('xyz')

    for i in range(0,len(a0),1):
        for j in range(i+1,len(a0),1):
            if (i > d):
                if a0[i] == a0[j]:
                    c = c + 1
                else:
                    bi_code.append(c+1)
#                    for bi in range(0,len)
#                    if c > 1:
                    a1.append(a0[i]) 
                    d = i + c
                    c = 0
                    break
                
    #Trim the noise in phones in order to get better performance
    #(num of noise phone is below 3, while neighnors are higher than 3)
    pp = True            
    for bi in range(0,len(bi_code),1):
        if (bi>0 and bi<(len(bi_code)-1)):
            if ((bi_code[bi-1]>3 and bi_code[bi+1]>3 and bi_code[bi]<3)):
                pp = False
                
        if pp==True:
            a2.append(a1[bi])
        pp = True
    bi_code = []    
    
    pre_finalbi.append(a1)
    #Trim the sil at the beginning and the end            
    if a2[0] == 'sil':
        del a2[0]
        
    if a2[len(a2)-1] == 'sil':
        del a2[len(a2)-1] 
          
    pre_final.append(a2)

#Convert phones into English letters
pre_teno39le = []
temptenole = []
for o in range(0,len(pre_final),1):
    for p in range(0,len(pre_final[o]),1):
        for q in range(0,len(r_array_map),1):
            if (pre_final[o][p] == r_array_map[q][0]):
                temptenole.append(r_array_map[q][2])
                cc = ''   
                for j in range(0,len(temptenole),1):
                    tenole = temptenole[j]
                    tenole1 = tenole.split('\n')[0]
                    cc = cc + tenole1
    pre_teno39le.append(cc)
    temptenole = []

##########################################
#Get the result by combine id and predicted labels tgt
result = []
for cc in range(0,len(pre_teno39le),1):
    tem = [te_id_final[cc],pre_teno39le[cc]]
    result.append(tem)

#Save the result into a csv file which will be summitted to kaggle

with open(result_csv, mode='w',newline='', encoding='utf-8') as write_file:
    writer = csv.writer(write_file, delimiter=',')


    writer.writerow(['id','phone_sequence'])
    for i in range (0,len(result),1):
        writer.writerow(result[i])
        

##change the 39 character to be english letter
#pre_teno39let = []
#temptenolet = []
#for o in range(0,len(pre_teno39char),1):
#    for p in range(0,len(pre_teno39char[o]),1):
#        for q in range(0,len(r_array_map),1):
#            if (pre_teno39char[o][p] == r_array_map[q][0]):
#                temptenolet.append(r_array_map[q][2])
#    pre_teno39let.append(temptenolet)
#    temptenolet = []


#delete the conservative letters

