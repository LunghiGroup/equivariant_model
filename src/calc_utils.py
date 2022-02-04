#!/usr/bin/env python3

###                          UTILITIES FUNCTIONS                           ###

import numpy as np
import random as rd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##############################################################################
# spherical harmonics from cartesian coordinates

def spherical_harmonics(l, m, R, r):
    X=R[0]
    Y=R[1]
    Z=R[2]
    
    if l==0:
        Y=1/2*np.sqrt(1/np.pi)
        
    elif l==1:
        if m==-1:
            Y=1/2*np.sqrt(3/(2*np.pi))*(X-Y*1j)/r
        elif m==0:
            Y=1/2*np.sqrt(3/np.pi)*Z/r
        elif m==1:
            Y=-1/2*np.sqrt(3/(2*np.pi))*(X+Y*1j)/r
    
    elif l==2:
        if m==-2:
            Y=1/4*np.sqrt(15/(2*np.pi))*((X-Y*1j)**2/r**2)
        elif m==-1:
            Y=1/2*np.sqrt(15/(2*np.pi))*(((X-Y*1j)*Z)/r**2)
        elif m==0:
            Y=1/4*np.sqrt(5/np.pi)*((3*(Z**2)-r**2)/r**2)
        elif m==1:
            Y=-1/2*np.sqrt(15/(2*np.pi))*(((X+Y*1j)*Z)/r**2)
        elif m==2:
            Y=1/4*np.sqrt(15/(2*np.pi))*((X+Y*1j)**2/r**2)
            
    elif l==4:
        if m==-4:
            Y=(3/16)*np.sqrt(35/(2*np.pi))*((X-Y*1j)**4/r**4)
        elif m==-3:
            Y=(3/8)*np.sqrt(35/np.pi)*(((X-Y*1j)**3*Z)/r**4)
        elif m==-2:
            Y=(3/8)*np.sqrt(5/(2*np.pi))*(((X-Y*1j)**2*(7*Z**2-r**2))/r**4)
        elif m==-1:
            Y=(3/8)*np.sqrt(5/np.pi)*(((X-Y*1j)*Z*(7*Z**2-3*r**2))/r**4)
        elif m==0:
            Y=(3/16)*np.sqrt(1/np.pi)*((35*Z**4-30*Z**2*r**2+3*r**4)/r**4)
        elif m==1:
            Y=(-3/8)*np.sqrt(5/np.pi)*(((X+Y*1j)*Z*(7*Z**2-3*r**2))/r**4)
        elif m==2:
            Y=(3/8)*np.sqrt(5/(2*np.pi))*(((X+Y*1j)**2*(7*Z**2-r**2))/r**4)
        elif m==3:
            Y=(-3/8)*np.sqrt(35/np.pi)*(((X+Y*1j)**3*Z)/r**4)
        elif m==4:
            Y=(3/16)*np.sqrt(35/(2*np.pi))*((X+Y*1j)**4/r**4)
    
    elif l==6:
        if m==-6:
            Y=(1/64)*np.sqrt(3003/np.pi)*((X-Y*1j)**6/r**6)
        elif m==-5:
            Y=(3/32)*np.sqrt(1001/np.pi)*(((X-Y*1j)**5*Z)/r**6)
        elif m==-4:
            Y=(3/32)*np.sqrt(91/(2*np.pi))*(((X-Y*1j)**4*(11*Z**2-r**2))/r**6)
        elif m==-3:
            Y=(1/32)*np.sqrt(1365/np.pi)*(((X-Y*1j)**3*Z*(11*Z**2-3*r**2))/r**6)
        elif m==-2:
            Y=(1/64)*np.sqrt(1365/np.pi)*(((X-Y*1j)**2*(33*Z**4-18*Z**2*r**2+r**4))/r**6)
        elif m==-1:
            Y=(1/16)*np.sqrt(273/(2*np.pi))*(((X-Y*1j)*Z*(33*Z**4-30*Z**2*r**2+5*r**4))/r**6)
        elif m==0:
            Y=(1/32)*np.sqrt(13/np.pi)*((231*Z**6-315*Z**4*r**2+105*Z**2*r**4-5*r**6)/r**6)
        elif m==1:
            Y=(-1/16)*np.sqrt(273/(2*np.pi))*(((X+Y*1j)*Z*(33*Z**4-30*Z**2*r**2+5*r**4))/r**6)
        elif m==2:
            Y=(1/64)*np.sqrt(1365/np.pi)*(((X+Y*1j)**2*(33*Z**4-18*Z**2*r**2+r**4))/r**6)
        elif m==3:
            Y=(-1/32)*np.sqrt(1365/np.pi)*(((X+Y*1j)**3*Z*(11*Z**2-3*r**2))/r**6)
        elif m==4:
            Y=(3/32)*np.sqrt(91/(2*np.pi))*(((X+Y*1j)**4*(11*Z**2-r**2))/r**6)
        elif m==5:
            Y=(-3/32)*np.sqrt(1001/np.pi)*(((X+Y*1j)**5*Z)/r**6)
        elif m==6:
            Y=(1/64)*np.sqrt(3003/np.pi)*((X+Y*1j)**6/r**6)

    return Y

##############################################################################
# convert cartesian tensor to spherical tensor

def Cartesian_to_spherical_tens(r,l,m,D):
    if r==0:
        T=D
    elif r==1:
        if l==1:
            if m==0:
                T=D[2]
            elif m==-1:
                T=(1/np.sqrt(2))*(D[0]-D[1]*1j)
            elif m==1:
                T=-(1/np.sqrt(2))*(D[0]+D[1]*1j)
    elif r==2:
        if l==1:
            if m==0:
                T=(1/np.sqrt(2))*(D[0,1]-D[1,0])
            elif m==-1:
                T=(1/(2*1j))*(D[0,2]-D[2,0]+(D[1,2]-D[2,1])*1j)
            elif m==1:
                T=(1/(2*1j))*(D[0,2]-D[2,0]+(D[2,1]-D[1,2])*1j)
        elif l==2:
            if m==0:
                T=(1/np.sqrt(6))*(2*D[2,2]-(D[0,0]+D[1,1]))
            elif m==-1:
                T=(1/2)*(D[0,2]+D[2,0]-(D[1,2]+D[2,1])*1j)
            elif m==1:
                T=(-1/2)*(D[0,2]+D[2,0]+(D[1,2]+D[2,1])*1j)
            elif m==-2:
                T=(1/2)*(D[0,0]-D[1,1]-(D[0,1]+D[1,0])*1j)
            elif m==2:
                T=(1/2)*(D[0,0]-D[1,1]+(D[0,1]+D[1,0])*1j)
    return T

##############################################################################
# cutoff function for smooth cutoff

def cutoff_func(r, rcut, rmin0):
    if r<=rcut:
        f_c = (1/2)*(np.cos(np.pi*(r-rmin0)/(rcut-rmin0))+1)
    else:
        f_c=0.0
    return f_c

##############################################################################
# generate random rotation 

def rand_rot():
    theta=rd.uniform(-np.pi,np.pi)
    axis=rd.uniform(0,4)

    Rx=np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
    Ry=np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-np.sin(theta),0.0,np.cos(theta)]])
    Rz=np.array([[np.cos(theta),-np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])
    
    if axis==1.0:
        R=Rx
    elif axis==2.0:
        R=Ry
    elif axis==3.0:
        R=Rz
    elif 1.0<axis<2.0:
        R=Rx.dot(Ry)
    elif 2.0<axis<3.0:
        R=Ry.dot(Rz)
    elif axis>3.0 or axis<1.0:
        R=Rz.dot(Rx)
    
    return R

##############################################################################
# generate random rotation around a single axis

def rand_rot_single_axis(axis):
    theta=rd.uniform(-np.pi,np.pi)

    Rx=np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
    Ry=np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-np.sin(theta),0.0,np.cos(theta)]])
    Rz=np.array([[np.cos(theta),-np.sin(theta),0.0],[np.sin(theta),np.cos(theta),0.0],[0.0,0.0,1.0]])
    
    if axis==1.0:
        R=Rx
    elif axis==2.0:
        R=Ry
    elif axis==3.0:
        R=Rz
    elif 1.0<axis<2.0:
        R=Rx.dot(Ry)
    elif 2.0<axis<3.0:
        R=Ry.dot(Rz)
    elif axis>3.0 or axis<1.0:
        R=Rz.dot(Rx)
    
    return theta,R

##############################################################################
# get Wigner D matrix from euler angles

def Wigner_D(l,angle):
    alpha=angle[0]
    beta=angle[1]
    gamma=angle[2]
    D=np.zeros((2*l+1,2*l+1), dtype='complex_')
    
    #construct small d matrix
    d=np.zeros((2*l+1,2*l+1))
    d[2,2]=d[-2,-2]=(1/4)*(1+np.cos(beta))**2
    d[2,1]=d[-1,-2]=-(1/2)*(np.sin(beta))*(1+np.cos(beta))
    d[2,0]=d[0,-2]=np.sqrt(3/8)*(np.sin(beta))**2
    d[2,-1]=d[1,-2]=-(1/2)*(np.sin(beta))*(1-np.cos(beta))
    d[2,-2]=(1/4)*(1-np.cos(beta))**2
    d[1,1]=d[-1,-1]=(1/2)*(2*(np.cos(beta))**2+np.cos(beta)-1)
    d[1,0]=d[0,-1]=-np.sqrt(3/8)*np.sin(2*beta)
    d[1,-1]=(1/2)*(-2*(np.cos(beta))**2+np.cos(beta)+1)
    d[0,0]=(1/2)*(3*(np.cos(beta))**2-1)
    d[0,1]=(-1)**(1-0)*d[1,0]
    d[0,2]=(-1)**(2-0)*d[2,0]
    d[-2,0]=(-1)**(0+2)*d[0,-2]
    d[-2,1]=d[-1,2]=(-1)**(1+2)*d[1,-2]
    d[-2,2]=(-1)**(2+2)*d[2,-2]
    d[-2,-1]=(-1)**(-1+2)*d[-1,-2]
    d[-1,0]=(-1)**(0+1)*d[0,-1]
    d[-1,1]=(-1)**(1+1)*d[1,-1]    
    d[1,2]=(-1)**(2-1)*d[2,1]
    
    for m1 in range(-l,l+1):
        for m2 in range(-l,l+1):
            #Dm'm
            D[m1,m2]=np.exp(-m1*alpha*1j)*d[m1,m2]*np.exp(-m2*gamma*1j)
    
    D=np.transpose(D)           
    return D

##############################################################################
# get euler angle from rotation matrix

def rotmat2eul(R):
    euler=np.zeros((3))
    euler[0]=np.arctan2(R[1,2],R[0,2])
    euler[1]=np.arccos(R[2,2])
    euler[2]=np.arctan2(R[2,1],-R[2,0])
    
    return euler

##############################################################################
# RMSE loss function for NN

def loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mean_rmse=tf.reduce_mean(squared_difference, axis=-1)
    return tf.sqrt(mean_rmse)

##############################################################################
