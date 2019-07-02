"""
process structure by Long
"""
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import Bio
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO
import numpy as np


import math

def rotation(r,v,theta):
	t1=r*np.cos(theta)
	t2=np.cross(v,r)
	t2=t2*np.sin(theta)
	vr=np.dot(v,r)
	t3=vr*v*(1-np.cos(theta))
	r=t1+t2+t3
	return r


def calha1(a,b,c):
	ab=b-a
	cb=b-c
	bc=c-b
	cbmo=np.linalg.norm(cb)
	d=cb*1.0814/cbmo
	bcmo=np.linalg.norm(cb)
	bc/=bcmo
	fabc=np.cross(ab,cb)
	fmo=np.linalg.norm(fabc)
	fabc/=fmo
	d=rotation(d,fabc,math.pi*108.0300/180.0)
	d=rotation(d,bc,math.pi*117.8600/180.0)
	d+=c
	return d


p = PDBParser(PERMISSIVE=0)

seqdic={}
for seq_record in SeqIO.parse("pdbaa", "fasta"):
	seqdic[seq_record.id]=seq_record.seq

t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P','GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H','LYS':'K','ARG':'R','ASP':'D','GLU':'E'}

f=open('cullpdb_pc25_res2.0_R0.25_d181126_chains9311','r')
s=f.readline()
pdbid=[]
pdbchain=[]
pdbs=[]
while 1:
	s=f.readline().strip('\n')
	if not s:
		break
	t=s.split()[0][:4]
	t2=s.split()[0][4:]
	pdbid.append(t)
	pdbchain.append(t2)



def cal(i):
	print(pdbid[i],pdbchain[i])
	pdb_name='pdb_/pdb'+pdbid[i].lower()+'.ent'     #pdb name
	try:
		s = p.get_structure("1",pdb_name)       #read pdb struture
		s = s[0][pdbchain[i]]                   #choose chain
		res_list = PDB.Selection.unfold_entities(s, 'R')   #read aminoacid
	except:
		return 0
	
	aa_list = []
	for a in res_list:
		if PDB.is_aa(a):
			aa_list.append(a)  #get acid

	error=0
	t=aa_list[0].get_id()[1]
	aa_list_full=[]
	for a in aa_list:
		while 1:
			if a.get_id()[1]<t:
				error=1
				break
			if a.get_id()[1]==t:
				aa_list_full.append(a)
				t+=1
				break
			else:
				aa_list_full.append(None)
				t+=1
	if error==1:                 
		return 0
	
	try:
		depth=PDB.ResidueDepth(s)   #氨基酸到蛋白质表面距离
	except:
		return 0

	dep_dict=depth.property_dict
	dep_keys=depth.property_keys
	dep_list=depth.property_list
	dps=[]
	for a in aa_list_full:
		try:
			aa_id=(a.get_parent().get_id(),a.get_id())
			if dep_dict.get(aa_id):
				dps.append(dep_dict[aa_id])
			else:
				dps.append([None,None])
		except:
			dps.append([None,None])
	dps=np.array(dps)

	try:
		HSEA=PDB.HSExposureCA(s)
	except:
		return 0

	HSEA_dict=HSEA.property_dict
	HSEA_keys=HSEA.property_keys
	HSEA_list=HSEA.property_list
	hse_a=[]
	for a in aa_list_full:
		try:
			aa_id=(a.get_parent().get_id(),a.get_id())
			if HSEA_dict.get(aa_id):
				hse_a.append(HSEA_dict[aa_id])
			else:
				hse_a.append([None,None,None])
		except:
			hse_a.append([None,None,None])
	hse_a=np.array(hse_a)

	try:
		HSEB=PDB.HSExposureCB(s)
	except:
		return 0

	HSEB_dict=HSEB.property_dict
	HSEB_keys=HSEB.property_keys
	HSEB_list=HSEB.property_list

	hse_b=[]
	for a in aa_list_full:
		try:
			aa_id=(a.get_parent().get_id(),a.get_id())
			if HSEB_dict.get(aa_id):
				hse_b.append(HSEB_dict[aa_id])
			else:
				hse_b.append([None,None,None])
		except:
			hse_b.append([None,None,None])

	hse_b=np.array(hse_b)

	seq_list=''
	for a in aa_list_full:
		try:
			t=a.get_resname()
			if t in t_dic:
				seq_list+=t_dic[t]
			else:
				seq_list+='X'
		except:
			seq_list+='X'

	ca_list=[]
	for a in aa_list_full:
		try:
			t=a['CA']
			ca_list.append(t)
		except:
			t=None
			ca_list.append(t)

	cb_list=[]
	for a in aa_list_full:
		try:
			t=a['CB']
			cb_list.append(t)
		except:
			t=None
			cb_list.append(t)

	n_list=[]
	for a in aa_list_full:
		try:
			t=a['N']
			n_list.append(t)
		except:
			t=None
			n_list.append(t)
	c_list=[]
	for a in aa_list_full:
		try:
			t=a['C']
			c_list.append(t)
		except:
			t=None
			c_list.append(t)



	angle=[]                             #三个角两个氨基酸相对位置
	for j in range(len(ca_list)):
		angle_t=[]
		for k in range(len(ca_list)):
			if ca_list[j]!=None and ca_list[k]!=None:
				ca1=ca_list[j].get_vector()
				ca2=ca_list[k].get_vector()
				if cb_list[j]!=None:
					cb=cb_list[j].get_vector()
					t1=PDB.vectors.calc_angle(cb,ca1,ca2)
				else:
					if c_list[j]!=None and n_list[j]!=None and ca_list[j]!=None:
						ca_v=ca_list[j].get_vector().get_array()
						c_v=c_list[j].get_vector().get_array()
						n_v=n_list[j].get_vector().get_array()
						cb=calha1(n_v,c_v,ca_v)
						cb=PDB.vectors.Vector(cb)
						t1=PDB.vectors.calc_angle(cb,ca1,ca2)
					else:
						t1=None
				if n_list[j]!=None:
					n_=n_list[j].get_vector()
					t2=PDB.vectors.calc_angle(n_,ca1,ca2)
				else:
					t2=None
				if c_list[j]!=None:
					c_=c_list[j].get_vector()
					t3=PDB.vectors.calc_angle(c_,ca1,ca2)
				else:
					t3=None
				angle_t.append([t1,t2,t3])
			else:
				angle_t.append([None,None,None])
		angle.append(angle_t)

	angle_d=[]              #六个角
	for j in range(len(angle)):
		angle_dt=[]
		for k in range(len(angle[j])):
			angle_dt.append(angle[j][k]+angle[k][j])
		angle_d.append(angle_dt)
	angle_d=np.array(angle_d)

	ca_num=len(ca_list)
	ca_dist=[]             #CA距离
	for j in range(len(ca_list)):
		for k in range(len(ca_list)):
			if ca_list[j]!=None and ca_list[k]!=None:
				ca_dist.append(ca_list[j]-ca_list[k])
			else:
				ca_dist.append(None)
	
	ca_dist=np.array(ca_dist)
	ca_dist=ca_dist.reshape(ca_num,ca_num)

	mask=[]    #是否有CA
	for j in range(len(ca_list)):
		if ca_list[j]!=None:
			mask.append(1)
		else:
			mask.append(0)
	
	ids=ca_dist==None
	ca_dist[ids]=100   #算不出来距离的设置为100
	ca_dist_cs=[]
	angle_cs=[]
	num_cs=[]
	for j in range(len(ca_dist)):
		t=ca_dist[j]
		s=t.argsort()
		ca_dist_cs.append(t[s[1:17]])
		angle_cs.append(angle_d[j][s[1:17]])
		num_cs.append(s[1:17])

	dic_r={}
	dic_r['dis']=ca_dist_cs
	dic_r['angle']=angle_cs
	dic_r['mask']=mask
	dic_r['ids']=num_cs
	dic_r['seq']=seq_list
	dic_r['dps']=dps
	dic_r['hsea']=hse_a
	dic_r['hseb']=hse_b

	out_name='pdb_other_cb/'+pdbid[i].lower()+pdbchain[i]+'_all.npy'
	np.save(out_name,dic_r)

for i in range(len(pdbid)):
	cal(i)
	
