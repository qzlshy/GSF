import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
import simnetnb



seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}

rootdir = '../pdb_other_cb/'


def g_data(name):
	path = os.path.join(rootdir,name)
	data=np.load(path).item()
	dis=data['dis']
	angle=data['angle']
	mask=data['mask']
	ids=data['ids']
	seq=data['seq']
	dps=data['dps']
	hsea=data['hsea']
	hseb=data['hseb']
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
					for k in range(len(angle[i][j])):
						if angle[i][j][k]==None:
							angle[i][j][k]=random.random()*3.14
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
							angle[i][k][k1]=random.random()*3.14
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
	

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = simnetnb.Sim()

model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

train=np.load('train_name_cb.npy')
val=np.load('val_name_cb.npy')
test=np.load('test_name_cb.npy')

num=list(range(len(train)))

f_log=open('train_nb.log','w')

epoches=30

best_model_wts = None
best_acc = 0.0
for epoch in range(epoches):
	print('epoch:',epoch)
	model.train()
	random.shuffle(num)
	k=0
	kk=0
	running_loss = 0.0
	running_corrects = 0
	for i in num:
		x,idx_,dis_,angle_,y,kkn=g_data(train[i])
		x=x.to(device)
		y=y.to(device)
		dis_=dis_.to(device)
		idx_=idx_.to(device)
		angle_=angle_.to(device)
		optimizer.zero_grad()
		outputs=model(x,idx_,dis_,angle_)
		loss = criterion(F.log_softmax(outputs,dim=1), y)
		loss.backward()
		optimizer.step()
		_, preds = torch.max(outputs, 1)
		running_corrects += torch.sum(preds == y.data)
		running_loss += loss.item()
		k+=1
		kk+=kkn
		if (k-1)%100==99:
			print(k,running_loss/k,running_corrects.double().cpu().data.numpy()/kk)

	print('train:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)

	model.eval()
	running_loss = 0.0
	running_corrects = 0
	k=0
	kk=0
	for i in range(len(val)):
		x,idx_,dis_,angle_,y,kkn=g_data(val[i])
		x=x.to(device)
		y=y.to(device)
		dis_=dis_.to(device)
		idx_=idx_.to(device)
		angle_=angle_.to(device)
		outputs=model(x,idx_,dis_,angle_)
		loss = criterion(F.log_softmax(outputs,dim=1), y)
		_, preds = torch.max(outputs, 1)
		running_corrects += torch.sum(preds == y.data)
		running_loss += loss.item()
		k+=1
		kk+=kkn
	print('val:',running_corrects.double().cpu().data.numpy()/kk)
	test_acc = running_corrects.double().cpu().data.numpy() /kk
	if test_acc>best_acc:
		best_acc = test_acc
		best_model_wts = copy.deepcopy(model.state_dict())
	print('val:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)

model.load_state_dict(best_model_wts)
running_loss = 0.0
running_corrects = 0
k=0
kk=0
for i in range(len(test)):
	x,idx_,dis_,angle_,y,kkn=g_data(test[i])
	x=x.to(device)
	y=y.to(device)
	dis_=dis_.to(device)
	idx_=idx_.to(device)
	angle_=angle_.to(device)
	outputs=model(x,idx_,dis_,angle_)
	loss = criterion(F.log_softmax(outputs,dim=1), y)
	_, preds = torch.max(outputs, 1)
	running_corrects += torch.sum(preds == y.data)
	running_loss += loss.item()
	k+=1
	kk+=kkn
print('test:',running_corrects.double().cpu().data.numpy()/kk)
print('test:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)

model.load_state_dict(best_model_wts)
pth_file='./modelnb/best.pth'
torch.save(model.state_dict(),pth_file)
