#!/usr/bin/env python3

###        FUNCTIONS FOR PREPARING INPUT AND OUTPUT AND BUILD MODELS       ###

import numpy as np
import calc_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

##############################################################################
# build and compile NN model

def convert_Cart_to_spher(tens,rank,l,nsample):
    if rank==0 or rank==1:
        Ctensor=tens
        
    elif rank==2:
        dim=int(np.sqrt(tens.shape[-1]))
        if (dim**2) == tens.shape[-1]:
            Ctensor=np.array([np.reshape(tens[i,:], (dim,dim)) for i in range(nsample)])     
        else:
            print("Cartesian tensor dimension error! Exit.")
            sys.exit()
        
    elif rank>2:
        print("Currently unable to convert tensors of rank higher than 2 with this python script! Exit.")
        sys.exit()
    
    stensor=np.zeros((nsample,(2*l+1)*2))
    for i in range(nsample):
        count=0
        for m in range(-l,l+1):
            T=calc_utils.Cartesian_to_spherical_tens(rank,l,m,Ctensor[i,:,:])
            stensor[i,count]=T.real
            count+=1
            stensor[i,count]=T.imag
            count+=1    
    
    return stensor

##############################################################################
# calculate equivariant representation for linear ridge regression
def get_rep_lrr(nsample,max_natom,natom,l,Nj,bispec_list,coords,rcut):
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
    
    return representation
    
##############################################################################
# building input for linear ridge regression

def build_inp_out_lrr(representation,stensor,nsample,l,nspecies,Nj,unique_ele,natom,all_eles):
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
    stensor=np.reshape(np.transpose(stensor), (nsample*(2*l+1)*2))
    
    return [feats,stensor]

##############################################################################
# get spherical harmonics

def get_Ylm(nsample,max_natom,natom,l,coords,rcut):
    Y=np.zeros((nsample,max_natom,2*l+1),dtype='complex')
    for i in range(nsample):
        for a in range(natom[i]):
            count=0
            for m in range(-l,l+1):
                for b in range(natom[i]):
                    if b!=a:
                        r=np.sqrt((coords[i][a,0]-coords[i][b,0])**2+(coords[i][a,1]-coords[i][b,1])**2+(coords[i][a,2]-coords[i][b,2])**2)
                        if r<=rcut:
                            Y[i,a,count]+=calc_utils.spherical_harmonics(l, m, coords[i][b,:]-coords[i][a,:], r)*calc_utils.cutoff_func(r, rcut, 0.0)                        
                count+=1
    return Y

##############################################################################
# normalize input for neural network

def normalization(nspecies,unique_names,nsample,natom,all_names,bispec_list,Nj):
    bispec_list=np.array(bispec_list)
    ele_rep=[]
    for species in range(nspecies):
        ele=unique_names[species]
        ele_bc=[]
        for i in range(nsample):
            for a in range(natom[i]):
                if all_names[i][a] == ele :
                    ele_bc.append(bispec_list[i,a,:])
        ele_rep.append(np.array(ele_bc))
 
    max_ele_bc=[]
    min_ele_bc=[]
    mean_ele_bc=[]
    for s in range(nspecies):
        max_bc=[]
        min_bc=[]
        mean_bc=[]
        for j in range(Nj):
            max_bc.append(np.max(ele_rep[s][:,j]))
            min_bc.append(np.min(ele_rep[s][:,j]))
            mean_bc.append(np.mean(ele_rep[s][:,j]))
        max_ele_bc.append(max_bc)
        min_ele_bc.append(min_bc)
        mean_ele_bc.append(mean_bc)

    norm_bispec_list=np.zeros((bispec_list.shape))  
    for s in range(nspecies):
        ele=unique_names[s]          
        for a in range(natom[0]):
            for j in range(Nj):
                if all_names[0][a] == ele:
                    norm_bispec_list[:,a,j]=(bispec_list[:,a,j]-mean_ele_bc[s][j])/(max_ele_bc[s][j]-min_ele_bc[s][j])

    return norm_bispec_list

##############################################################################
# building input and output for neural network

def build_inp_out_nn(unique_names,natom,l,Y,all_names,nsample,norm_bispec_list,stensor):
    Y2=[]
    for m in range(2*l+1):
        if Y[:,:,m].imag.all()!=0:
            Y2.append(Y[:,:,m].real)
            Y2.append(Y[:,:,m].imag)
        else:
            Y2.append(Y[:,:,m].real)
    Y2=np.transpose(Y2)

    inp=[]
    for ele in unique_names:
        for a in range(natom[0]):
            if all_names[0][a]==ele:
                inp_bc=[]
                for m in range((2*l+1)*2-1):
                    for j in range(nsample):
                        inp_bc.append(norm_bispec_list[j,a,:])
                inp.append(np.array(inp_bc))

    Ylm=[]
    for m in range((2*l+1)*2-1):
        for j in range(nsample):
            sh=[]
            for ele in unique_names:
                for a in range(natom[0]):
                    if all_names[0][a]==ele:
                        sh.append(Y2[a,j,m])
            Ylm.append(sh)        
    inp.append(np.array(Ylm))
    
    stensor2=[]
    for m in range((2*l+1)*2):
        if stensor[:,m].all()!=0:
            stensor2.append(stensor[:,m])
    stensor2=np.transpose(np.array(stensor2))
    
    B=[]
    for m in range((2*l+1)*2-1):
        for j in range(nsample):
            B.append(stensor2[j,m])
    B=np.array(B)
    
    return [inp,B]

##############################################################################
# build and compile NN model

def create_NNYlm(unique_names,natom,nspecies,names,Nj,reg,nlayers,nnodes,learn_rate,activ):
    
    layers_collect=[]
    input_collect=[]
        
    for species in range(nspecies):
        ele=unique_names[species]
        hidden=[]
        for i in range(nlayers):
            hidden.append(layers.Dense(nnodes[i], activation=activ, kernel_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg)))

        dense=layers.Dense(1, name='{}_contri'.format(ele), use_bias=False,kernel_regularizer=keras.regularizers.l2(reg), bias_regularizer=keras.regularizers.l2(reg))
        
        for a in range(natom):
            if names[a]==ele:
                inputs=layers.Input(shape=(Nj))
                if nlayers==0:
                    layer_dense = dense(inputs)
                    
                else:
                    for i in range(nlayers):
                        layer=hidden[i]
                        if i==0:
                            hidden_layer=layer(inputs)
                        else:
                            hidden_layer=layer(hidden_layer)
               
                    layer_dense = dense(hidden_layer)
                input_collect.append(inputs)
                layers_collect.append(layer_dense)
    
    inpY=layers.Input(shape=(natom))
    input_collect.append(inpY)
    concat = layers.Concatenate(axis=-1)(layers_collect)
    mult=tf.keras.layers.Multiply()([concat,inpY])
    out=tf.math.reduce_sum(mult,axis=-1)
#    out=keras.layers.Lambda( lambda x: K.sum(x, axis=1), input_shape=(None,53))(concat)
    model= keras.models.Model(input_collect,out)
        
    opt=keras.optimizers.Adam(learning_rate=learn_rate)
    
    model.compile(loss=calc_utils.loss_fn, optimizer=opt)

    return model
