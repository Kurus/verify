import numpy as np
from scipy import signal as sg
dim = 56
dim_p=dim + 2
dep = 16
ker = 64

in_l = np.zeros(dim_p*dim_p*dep, dtype='uint8').reshape((dim_p,dim_p,dep))
in_ori = np.arange(dim*dim*dep, dtype='uint8').reshape((dim,dim,dep))
in_l[1:-1,1:-1,:] = in_ori
# print(in_l[:,:,0]); print("_____________________-")
f_in = open("input_layer","w")
for z in range(0,dim):
    for y in range(0,dep):
        for x in range(0,dim):
            lis = str(in_l[z:z+3,x:x+3,y].flatten().tolist())[1:-1] # repeat it ker /4
            f_in.write(lis+'\n')
            f_in.write(lis+'\n')
            f_in.write(lis+'\n')
            f_in.write(lis+'\n')



ker_l_1 = np.arange(ker*dep, dtype='uint8').reshape((ker,dep))
# print(ker_l_1);print("________")
f_k_1 = open("ker_1x1","w")
for z in range(0,dep):
    lis = str(ker_l_1[:,z])[1:-1]
    f_k_1.write(lis+'\n')

ker_l_3 = np.arange(ker*dep*9, dtype='uint8').reshape((ker,dep,9))
# print(ker_l_3[:,0,:]);print("________")
f_k_3 = open("ker_3x3","w")
for m in range(0,dim): # repet 3x3 kernel
    for z in range(0,dep):
        lis = ker_l_3[:,z,:]
        for x in range(0,ker,8):
            for a in range(0,8):
                eig = lis[x+a,0:8]
                f_k_3.write(str(eig)[1:-1]+'\n')
            nin = lis[x:x+8,-1].flatten()[::-1] #reversed
            f_k_3.write(str(nin)[1:-1]+'\n')



out = np.zeros(ker*dep*dim*dim, dtype='uint8').reshape((ker,dep,dim,dim))
for k in range(0,ker):
    for l in range(0,dep):
        res = sg.convolve(in_l[:,:,l],[[ker_l_1[k,l]]] , "valid").astype(int)
        res = np.bitwise_and(res, 0xff)
        out[k,l,:,:]=res[1:-1,1:-1]
# print(out[1,1,:,:]);print('______')
f_out = open("out_1x1","w")
# out = np.arange(ker*dep*dim*dim, dtype='uint32').reshape((ker,dep,dim,dim))
for r in range(0,dim):
    for d in range(0,dep):
        for c in range(0,dim):
            f_out.write(str(out[:,d,r,c])[1:-1]+'\n')






out = np.zeros(ker*dep*dim*dim, dtype='uint8').reshape((ker,dep,dim,dim))
for k in range(0,ker):
    for l in range(0,dep):
        res = sg.convolve(in_l[:,:,l],ker_l_3[k,l].reshape((3,3)) , "valid").astype(int)
        res = np.bitwise_and(res, 0xff)
        out[k,l,:,:]=res
# print(out[1,1,:,:]);print('______')
# out = np.arange(ker*dep*dim*dim, dtype='uint32').reshape((ker,dep,dim,dim))

f_out = open("out_3x3","w")
for r in range(0,dim):
    for d in range(0,dep):
        for c in range(0,dim):
            f_out.write(str(out[:,d,r,c])[1:-1]+'\n')