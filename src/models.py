#!/usr/bin/env python3

###                 MAIN SCRIPT FOR RUNNING APPROPRIATE MODEL              ###

import numpy as np
import read_inp
import calc_utils
import sys
import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##############################################################################
# run linear ridge regression

##############################################################################
# read necessary parameters

args = read_inp.parse_hyperparam("Hyperparameters for tensor fitting")

nsample,all_eles,full_ele_list,unique_ele,nspecies,natom,coords,bc,Nj,bispec_list,tens,l,rcut,reg,rank,nlayers,activ,nnodes,epoch,batch,learn_rate = read_inp.read_input(args)

max_natom=max(natom)
#check for correct input
if args.neurnets and len(nnodes)!=nlayers:
    print("Missing or extra number of nodes for the specified number of hidden layers! Exit.")
    sys.exit()

##############################################################################
# convert Cartesian tensor to spherical tensor if necessary 

if args.tensconv:    
    stensor=modules.convert_Cart_to_spher(tens,rank,l,nsample)
    
else:
    stensor=tens

##############################################################################

if args.neurnets:
    # running NN
    Ylm=modules.get_Ylm(nsample,max_natom,natom,l,coords,rcut)
    norm_bispec_list=modules.normalization(nspecies,unique_ele,nsample,natom,all_eles,bispec_list,Nj)
    [inp,B]=modules.build_inp_out_nn(unique_ele,natom,l,Ylm,all_eles,nsample,norm_bispec_list,stensor)
    
    if args.skipfit:
        model=keras.models.load_model("NNYlm_model",compile=False)
        model.compile(loss=calc_utils.loss_fn)
    else:
        model=modules.create_NNYlm(unique_ele,natom[0],nspecies,all_eles[0],Nj,reg,nlayers,nnodes,learn_rate,activ)
        history=model.fit(inp,B,epochs=epoch,verbose=0,batch_size=batch)
        model.save("NNYlm_model")
    
    out=model.predict(inp)   
    
else:
    # running LRR
    
    representation=modules.get_rep_lrr(nsample,max_natom,natom,l,Nj,bispec_list,coords,rcut)
    [feats,stensor]=modules.build_inp_out_lrr(representation,stensor,nsample,l,nspecies,Nj,unique_ele,natom,all_eles)
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

f=open("tensors.out","w")
f.write("Prediction\tReference\n")
if args.neurnets:
    for i in range(nsample*((2*l+1)*2-1)):
        f.write("{}\t{}\n".format('%.12f' % out[i], '%.12f' % B[i]))
else:
    for i in range(nsample*((2*l+1)*2)):
        f.write("{}\t{}\n".format('%.12f' % out[i], '%.12f' % B[i]))
    
f.close()
