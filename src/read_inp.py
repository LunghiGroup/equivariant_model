#!/usr/bin/env python3

###                      READING INPUTS AND ARGUMENTS                      ###

import argparse
from ase.io import read
import numpy as np

##############################################################################
# parse hyperparameters argument

def parse_hyperparam(descr):
    parser = argparse.ArgumentParser(description=descr)
    
    parser.add_argument("-l", "--lorder", type=int, required=True, help="Order of the spherical tensor and the spherical harmonics")
    parser.add_argument("-in","--input", type=str, required=True, help="Input file containing file names for 3 files containing coordinates, bispectrum components, and tensors")
    parser.add_argument("-tc","--tensconv", action='store_true', help="Option to convert Cartesian tensor to spherical tensor")
    parser.add_argument("-rc","--rcut", type=float, default=4.0, help="Cutoff radius")
    parser.add_argument("-reg","--regularize", type=float, required=True, help="Regularization value")
    parser.add_argument("-r","--rank", type=int, help="Order of the Cartesian tensor, required if -tc option is turn on")
    parser.add_argument("-sf","--skipfit", action='store_true', help="Skip training when turn on, used for test and validation")
    parser.add_argument("-nn","--neurnets", action='store_true', help="Use Neural Networks to fit")
    parser.add_argument("-nlay","--nlayers", type=int, default=2, help="Number of hidden layers in Neural Networks")
    parser.add_argument("-nnod","--nnodes", nargs='+', help="List of number of nodes per hidden layer in Neural Networks")
    parser.add_argument("-af","--activfunc", type=str, default="sigmoid", help="Activation function for the hidden layers in Neural Networks")
    parser.add_argument("-lrate","--learnrate", type=int, default=0.001, help="Learning rate of the optimizer in Neural Networks")
    parser.add_argument("-bs","--batchsize", type=int, default=100, help="Batch size when learning with Neural Networks")
    parser.add_argument("-ep","--epoch", type=int, default=1000, help="Number of epochs to train the Neural Networks")

    args = parser.parse_args()
    return args

##############################################################################
# read input file

def read_input(args):
    with open(args.input) as f:
        for line in f:
            xyzfile,bcfile,tensfile=line.split(" ")
    tensfile=tensfile.strip()
    
    #read xyz
    xyz = read(xyzfile,':')
    nsample = len(xyz)
    
    all_eles = [xyz[i].get_chemical_symbols() for i in range(nsample)]
    full_ele_list=[ele for sublist in all_eles for ele in sublist]
    unique_ele = list(sorted(set(full_ele_list)))
    nspecies = len(unique_ele)
    
    natom=[xyz[i].get_number_of_atoms() for i in range(nsample)]
    coords=[xyz[i].get_positions() for i in range(nsample)]
    
    #read bispec
    bc=np.genfromtxt(fname=bcfile, delimiter=" ")
    Nj=bc.shape[-1]
    count=0
    bispec_list=[]
    for i in range(nsample):
        bispec=[]
        for a in range(natom[i]):
            bispec.append(bc[count+a])
        bispec_list.append(bispec)
        count+=natom[i]
        
    #read tensor
    tens=np.genfromtxt(fname=tensfile, delimiter=" ")
    
    l=args.lorder
    rcut=args.rcut
    reg=args.regularize
    rank=args.rank
    
    #NN hyperparameters
    nlayers=args.nlayers
    activ=args.activfunc
    if args.nnodes is None:
        nnodes=[16]*nlayers
    else:
        nnodes=args.nnodes
    epoch=args.epoch
    batch=args.batchsize
    learn_rate=args.learnrate

    return nsample,all_eles,full_ele_list,unique_ele,nspecies,natom,coords,bc,Nj,bispec_list,tens,l,rcut,reg,rank,nlayers,activ,nnodes,epoch,batch,learn_rate
