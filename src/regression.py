#!/usr/bin/env python3
import numpy as np
import read_inp
import calc_utils
import sys

##############################################################################
# read necessary parameters

args = read_inp.parse_hyperparam("Hyperparameters for tensor fitting")

[nsample,all_eles,full_ele_list,unique_ele,nspecies,natom,coords,bc,Nj,bispec_list,tens,l,rcut,reg,rank] = read_inp.read_input(args)

max_natom=max(natom)

##############################################################################
# convert Cartesian tensor to spherical tensor if necessary 

if args.tensconv:
    
    if rank==0 or rank==1:
        Ctensor=tens
        
    elif rank==2:
        dim=int(np.sqrt(tens.shape[-1]))
        if (dim**2) == tens.shape[-1]:
            Ctensor=np.array([np.reshape(tens[i,:], (dim,dim)) for i in range(nsample)])            
            stensor=np.zeros((nsample,(2*2+1)*2))
            for i in range(nsample):
                count=0
                for m in range(-2,2+1):
                    T=calc_utils.Cartesian_to_spherical_tens(2,2,m,Ctensor[i,:,:])
                    stensor[i,count]=T.real
                    count+=1
                    stensor[i,count]=T.imag
                    count+=1
                    
        else:
            print("Cartesian tensor dimension error! Exit.")
            sys.exit()
        
    elif rank>2:
        print("Currently unable to convert tensors of rank higher than 2 with this python script! Exit.")
        sys.exit()
    
else:
    stensor=tens
    
##############################################################################
# calculate equivariant representation

representation=np.zeros((nsample,max_natom,(2*l+1),Nj),dtype='complex_')
for i in range(nsample):
    for a in range(natom[i]):
        count=0
        for m in range(-l,l+1):
            Y=0
            for b in range(natom[i]):
                if b!=a:
                    r=np.sqrt((coords[i][a,0]-coords[i][b,0])**2+(coords[i][a,1]-coords[i][b,1])**2+(coords[i][a,2]-coords[i][b,2])**2)
                    if r<=rcut:
                        Y+=calc_utils.spherical_harmonics(l, m, coords[i][b,:]-coords[i][a,:], r)*calc_utils.cutoff_func(r, rcut, 0.0)
            representation[i,a,count,:]=bispec_list[i][a]*Y
            count+=1
    
##############################################################################
# 

ele_rep=np.zeros((nsample,(2*l+1),nspecies,Nj),dtype='complex_')
for species in range(nspecies):
    ele=unique_ele[species]
    for i in range(nsample):
        for a in range(natom[i]):
            if all_eles[i][a] == ele :
                ele_rep[i,:,species,:]+=representation[i,a,:,:]
                
descriptor=np.reshape(ele_rep,(nsample,(2*l+1),nspecies*Nj))  
des1=np.zeros((nsample*2,(2*l+1),nspecies*Nj)) 
des2=np.zeros((nsample*2,(2*l+1),nspecies*Nj)) 
des3=np.zeros((nsample*2,(2*l+1),nspecies*Nj*2)) 
for i in range(2*l+1):
    des1[:,i,:]=np.vstack([descriptor[:,i,:].real,descriptor[:,i,:].imag]) 
    des2[:,i,:]=np.vstack([descriptor[:,i,:].imag,descriptor[:,i,:].real]) 
    des3[:,i,:]=np.hstack([des1[:,i,:],des2[:,i,:]])
    
feats=np.concatenate(([des3[:,i,:] for i in range(2*l+1)]))

stensor=np.reshape(np.transpose(stensor), (nsample*10))

if args.skipfit:
    A=feats
    B=stensor
    coefficient=np.load("coefficients.npy")
else:
    regularizer=np.zeros((nspecies*Nj*2,nspecies*Nj*2))
    np.fill_diagonal(regularizer, reg)
    A=np.vstack([feats,regularizer])

    reg_out=np.zeros((Nj*nspecies*2))
    B=np.hstack([stensor,reg_out])

    coeffs=np.linalg.lstsq(A, B)
    coefficient=coeffs[0]
    np.save("coefficients.npy",coefficient)

out=np.matmul(A,coefficient)
    
rmse=np.sqrt(np.mean((out-B)**2))
print("fit RMSE: {}".format(rmse))