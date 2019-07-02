#!/usr/bin/python3

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
import simnetnb
import cal_pdb_test
import sys

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}


def g_data(name):
	seq,ids,mask,dis,angle,dps,hsea,hseb=cal_pdb_test.get_dis(name)
	if seq==None:
		return None,None,None,None,None,0

	idx_t=[]
	dis_t=[]
	angle_t=[]
	label=[]
	kk=0
	for i in range(len(mask)):
		if mask[i]!=0:
			idx_=[]
			dis_=[]
			angle_=[]
			nonid=dps[i]==None
			dps[i][nonid]=0.0
			idx_.append(21)
			dis_.append(dps[i][1]/10.0)
			hseaid=hsea[i]==None
			hsea[i][hseaid]=random.random()
			hsea[i][0]=hsea[i][0]/20.0
			hsea[i][1]=hsea[i][1]/20.0
			hsea[i][2]=0
			hsebid=hseb[i]==None
			hseb[i][hsebid]=random.random()
			hseb[i][0]=hseb[i][0]/20.0
			hseb[i][1]=hseb[i][1]/20.0
			hseb[i][2]=0
			hse=np.append(hsea[i],hseb[i])
			angle_.append(hse)	
			for j in range(len(ids[i])):
				if len(idx_)==10:
					break
				if abs(ids[i][j]-i)>6:
					nonid=angle[i][j]==None
					angle[i][j][nonid]=1.0
					if seq[ids[i][j]] in seqdic:
						idx_.append(seqdic[seq[ids[i][j]]])
					else:
						idx_.append(20)
					dis_.append(dis[i][j]/10.0)
					angle_.append(angle[i][j]/3.0)
			while len(idx_)<10:
				idx_.append(22)
				dis_.append(0.0)
				angle_.append(np.zeros([6]))

			nb_id=[i-6,i-5,i-4,i-3,i-2,i-1,i+1,i+2,i+3,i+4,i+5,i+6]
			for a in nb_id:
				if a in ids[i]:
					k=np.where(ids[i]==a)
					k=k[0][0]
					for k1 in range(len(angle[i][k])):
						if angle[i][k][k1]==None:
							angle[i][k][k1]=1.0
					if seq[a] in seqdic:
						idx_.append(seqdic[seq[a]])
					else:
						idx_.append(20)
					dis_.append(dis[i][k]/10.0)
					angle_.append(angle[i][k]/3.0)
				else:
					idx_.append(22)
					dis_.append(0.0)
					angle_.append(np.zeros([6]))
							
			idx_t.append(idx_)
			dis_t.append(dis_)
			angle_t.append(angle_)
			if seq[i] in seqdic:
				label.append(seqdic[seq[i]])
			else:
				label.append(20)
			kk+=1
	data_t=np.eye(23,dtype=np.float32)
	dis_t=np.array(dis_t,dtype=np.float32)
	angle_t=np.array(angle_t,dtype=np.float32)
	idx_t=np.array(idx_t,dtype=np.float32)
	label=np.array(label,dtype=np.float32)
	data_t=torch.from_numpy(data_t)
	dis_t=torch.from_numpy(dis_t)
	angle_t=torch.from_numpy(angle_t)
	idx_t=torch.from_numpy(idx_t)
	label=torch.from_numpy(label)
	idx_t=idx_t.long()
	label=label.long()
	return data_t,idx_t,dis_t,angle_t,label,kk
	

device = "cpu"
model = simnetnb.Sim()

model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

model.eval()
running_corrects = 0
k=0
kk=0

pth_file='./modelnb/best.pth'
model.load_state_dict(torch.load(pth_file))
model.eval()

rootdir = sys.argv[1]
f_name=os.listdir(rootdir)

for a in f_name:
	if a[-4:]!='.pdb':
		continue
	name=os.path.join(rootdir,a)
	x,idx_,dis_,angle_,y,kkn=g_data(name)
	if kkn==0:
		continue
	x=x.to(device)
	y=y.to(device)
	dis_=dis_.to(device)
	idx_=idx_.to(device)
	angle_=angle_.to(device)
	outputs=model(x,idx_,dis_,angle_)
	_, preds = torch.max(outputs, 1)
	running_corrects=torch.sum(preds == y.data)
	p=F.softmax(outputs,dim=1)
	p=p.cpu().data.numpy()
	out=outputs.cpu().data.numpy()
	r=F.log_softmax(outputs,dim=1)
	r=r.cpu().data.numpy()
	y=y.cpu().data.numpy()


	t=0.0
	for i in range(len(y)):
		t+=r[i][int(y[i])]

	print(a,t/len(y),running_corrects.cpu().data.numpy()/len(y))
