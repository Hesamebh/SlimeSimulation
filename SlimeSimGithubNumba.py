# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:59:59 2021

@author: Hesam
Slime simulation 
"""

import numpy as np
from scipy import ndimage
import cv2
from numba import prange ,jit
import os

png_dir = 'anim/'  # Folder for saving PNG files, Make this empty folder befor first run
for file_name in os.listdir(png_dir):
    file_path = os.path.join(png_dir, file_name)
    os.remove(file_path)
iterat=500 # Number of images to be generated 

movespeed=10.0  # Agent speed

deltatimes=1
sensorSize=3
senseKernel=np.ones([sensorSize*2+1,sensorSize*2+1], dtype=float)
sensorAngleRad =25. * (3.1415 / 180)

diffusePercent=5 #diffuse rate percent 
decayRate=0.95
decaySize=9 # decay kernel Size, choose odd numberfor symetry
dP=decayRate*diffusePercent/100.
dP_=decayRate*(1-dP)/(decaySize*decaySize-1)

turnSpeed = 25.0 * 3.1415/180


kernel=np.ones([decaySize,decaySize])*dP_

kernel[int(decaySize/2),int(decaySize/2)]=dP

widthorig =704
heightorig =704  
scale_percent = 300 # percent of original size for higher simulation resolusion
width = int(widthorig * scale_percent / 100)
height = int(heightorig* scale_percent / 100)
dim = (width, height)



grid = np.zeros((width,height), dtype=float)
nextgrid = np.zeros((width,height), dtype=float)
imgout=np.zeros((width,height,3), dtype=np.uint8)
settingz=np.zeros(6)
agent=np.zeros(6)
retwecor=np.zeros(6)
##########################################
# Setting values for simulation
settingz=np.array([movespeed,deltatimes,sensorAngleRad,width,height,turnSpeed], dtype=float)


##########################################
# Sense in different directions
@jit(nopython=True) 
def sense(agents,settingzs):
    
    width =settingzs[3]
    height =settingzs[4] 
    movespeed=settingzs[0]
    deltatimes=settingzs[1]
    sensorAngleRad=settingzs[2]
    
    sensorAngle = agents[2] 
    sensorDirc = np.cos(sensorAngle) 
    sensorDirs = np.sin(sensorAngle)
   
    sensorCentreX = agents[0]+sensorDirc*movespeed*deltatimes
    sensorCentreY = agents[1]+sensorDirs*movespeed*deltatimes
    xf= min(width - 1, max(0, sensorCentreX))+.5
    yf= min(height - 1, max(0, sensorCentreY))+.5
    
      
    
    sensorAngle = agents[2] + sensorAngleRad
    sensorDirc = np.cos(sensorAngle) 
    sensorDirs = np.sin(sensorAngle)
   
    sensorCentreX = agents[0]+sensorDirc*movespeed*deltatimes
    sensorCentreY = agents[1]+sensorDirs*movespeed*deltatimes
    xl= min(width - 1, max(0, sensorCentreX))+.5
    yl= min(height - 1, max(0, sensorCentreY))+.5
    
      
    
    sensorAngle = agents[2] -sensorAngleRad
    sensorDirc = np.cos(sensorAngle) 
    sensorDirs = np.sin(sensorAngle)
  
    sensorCentreX = agents[0]+sensorDirc*movespeed*deltatimes
    sensorCentreY = agents[1]+sensorDirs*movespeed*deltatimes
    xr= min(width - 1, max(0, sensorCentreX))+.5
    yr= min(height - 1, max(0, sensorCentreY))+.5
   
    
    retwecoors=[xf,yf,xl,yl,xr,yr]
    
    return retwecoors
	
############################################
# Update agents position and angles  
@jit(nopython=True)
def updateantz(agentu,settingzu,sensegridu):
    
    deltatimes=settingzu[1]
    widthf=settingzu[3]
    heightf=settingzu[4]    
    turnSpeed=settingzu[5]
    
    retwecor=sense(agentu,settingzu)
    xf,yf,xl,yl,xr,yr=np.int16(retwecor)
    weightForward = sensegridu[xf,yf]
    weightLeft = sensegridu[xl,yl]
    weightRight = sensegridu[xr,yr]
   
   
    randomSteerStrength =np.random.rand() 
  
    
    if (weightForward > weightLeft and weightForward > weightRight):
        agentu[2] += 0.0
    
    elif (weightForward < weightLeft and weightForward < weightRight):
        agentu[2] +=(randomSteerStrength - 0.5)*2 * turnSpeed * deltatimes
    
	
	# Turn right
    # Turn left
    elif (weightLeft > weightRight):
        agentu[2]+=randomSteerStrength * turnSpeed * deltatimes;
     
    elif (weightRight > weightLeft):
        agentu[2]-=randomSteerStrength * turnSpeed * deltatimes;
        


    direction=np.array([np.cos(agentu[2]),np.sin(agentu[2])])
    
    newPos=[agentu[0]+direction[0]*movespeed*deltatimes, agentu[1]+direction[1]*movespeed*deltatimes]
  
    if (newPos[0]<0 or newPos[0]>(width-1) or newPos[1]<0 or newPos[1]>(height-1)):
        
        newPos[0] = min(widthf-1.,max(0., newPos[0]))
        newPos[1] = min(heightf-1.,max(0., newPos[1]))
        agentu[2] = randomSteerStrength*2*np.pi
	
	
    agentu[0] = newPos[0]
    agentu[1] = newPos[1]
    agentu[2]=agentu[2]%(2*np.pi)
    agentr=agentu[0:3]
    return agentr
 


kk=0
agentNo=width*height

###########################################
# Generating agents
antz = np.zeros((agentNo,3), dtype=float)
for ii in range(0,width,3):
    for jj in range(0,height,3):
        
        antz[kk,0]=ii
        antz[kk,1]=jj
        antz[kk,2]=np.random.rand()*2*np.pi
        grid[ii,jj]=1.
        kk+=1

agentNo=kk 
antz=antz[0:kk,:]  
antz2=antz[0:kk,:]      

nextgrid = ndimage.convolve(grid, kernel, mode="wrap", cval=0)
print("Agent no ",agentNo)
############################################
# Main loop for generating PNG images
for jj in range(iterat):
    
    sensegrid =ndimage.convolve(grid, senseKernel, mode="constant", cval=0)
    
    for ii in prange(agentNo):
        agentc=antz[ii,:] 
        antz2[ii,:]=updateantz(agentc,settingz,sensegrid)
    grid=nextgrid.copy()
    antz=antz2.copy()
   
    x=np.int16(np.round(antz[:,0]))
    y=np.int16(np.round(antz[:,1]))
       
    grid[x,y]=1.
    
    nextgrid =ndimage.convolve(grid, kernel, mode="constant", cval=0)
    
    
    print(f'{jj:06}/{iterat:06} ',np.max(sensegrid),np.mean(antz[:,2]),np.min(antz[:,2]),np.max(antz[:,2]))
    imgout[:,:,0]=np.uint8(nextgrid*255)
    imgout[:,:,1]=np.uint8(nextgrid*255)
    imgout[:,:,2]=np.uint8(nextgrid*255)
    imgoutp = cv2.resize(imgout, (widthorig,heightorig), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite(f'anim/IMG1{(jj):06}.png',imgoutp)
     
    
#################################
# Make animation from PNG files
import imageio

png_dir = 'anim/'
images = []
for file_name in os.listdir(png_dir):
    
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
   
imageio.mimsave('AnimNumba.mp4',images, fps=20)    


