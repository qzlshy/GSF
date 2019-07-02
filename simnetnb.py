import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class idxLayer(torch.nn.Module):
	def __init__(self):
		super(idxLayer, self).__init__()

	def forward(self, x, idx,dis,angle):
		h, w = idx.size()
		r=[]
		for i in range(h):
			t=torch.index_select(x,0,idx[i])
			t=t.view(1,-1)
			r.append(t)
		dis=dis.view(h,-1)
		angle=angle.view(h,-1)
		x=torch.cat(r,0)
		x=torch.cat((x,dis,angle),1)
		return(x)

class Sim(nn.Module):
	def __init__(self):
		super(Sim, self).__init__()
		self.idxl=idxLayer()
		self.linear1 = nn.Linear(660, 512)
		self.relu1= nn.ReLU()
		self.linear2 = nn.Linear(512, 512)
		self.relu2= nn.ReLU()
		self.linear3 = nn.Linear(512, 512)
		self.relu3= nn.ReLU()
		self.linear4 = nn.Linear(512, 21)
	

	def forward(self, x,idx,dis,angle):
		x=self.idxl(x,idx,dis,angle)
		x=self.relu1(self.linear1(x))
		x=self.relu2(self.linear2(x))
		x=self.relu3(self.linear3(x))
		x=self.linear4(x)
		
		return x

