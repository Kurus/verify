import numpy as np
from scipy import signal as sg
dim = 6
dim_p=dim + 2
dep = 4
ker = 32

in_l = np.zeros(dim_p*dim_p*dep, dtype='uint8').reshape((dim_p,dim_p,dep))
in_ori = np.arange(dim*dim*dep, dtype='uint8').reshape((dim,dim,dep))
in_l[1:-1,1:-1,:] = in_ori
# print(in_l[:,:,0]); print("_____________________-")
f_in = open("input_layer.txt","w")
f_in_b = open("input_layer.bin","wb")
for z in range(0,dim):
    for y in range(0,dep):
        for x in range(0,dim):
            lis = in_l[z:z+3,x:x+3,y].flatten().tolist()
            for rep in range(0,ker,4):
                f_in.write(str(lis)[1:-1]+'\n')
                f_in_b.write(bytearray(lis))




ker_l_1 = np.arange(ker*dep, dtype='uint8').reshape((ker,dep))
# print(ker_l_1);print("________")
f_k_1 = open("ker_1x1.txt","w")
f_k_1_b = open("ker_1x1.bin","wb")
for z in range(0,dep):
    lis = ker_l_1[:,z]
    f_k_1_b.write(bytearray(lis))
    f_k_1.write(str(lis)[1:-1]+'\n')

ker_l_3 = np.arange(ker*dep*9, dtype='uint8').reshape((ker,dep,9))
# print(ker_l_3[0,0,:]);print("________")
f_k_3 = open("ker_3x3.txt","w")
f_k_3_b = open("ker_3x3.bin","wb")
for m in range(0,dim): # repet 3x3 kernel
    for z in range(0,dep):
        lis = ker_l_3[:,z,:]
        for x in range(0,ker,8):
            for a in range(0,8):
                eig = lis[x+a,0:8]
                f_k_3_b.write(bytearray(eig))
                f_k_3.write(str(eig)[1:-1]+'\n')
            nin = lis[x:x+8,-1].flatten()[::-1] #reversed
            f_k_3_b.write(bytearray(nin))
            f_k_3.write(str(nin)[1:-1]+'\n')



out = np.zeros(ker*dep*dim*dim, dtype='uint8').reshape((ker,dep,dim,dim))
for k in range(0,ker):
    for l in range(0,dep):
        res = sg.convolve(in_l[:,:,l],[[ker_l_1[k,l]]] , "valid").astype(int)
        res = np.bitwise_and(res, 0xff)
        out[k,l,:,:]=res[1:-1,1:-1]
# print(out[1,1,:,:]);print('______')
f_out = open("out_1x1.txt","w")
f_out_b = open("out_1x1.bin","wb")
# out = np.arange(ker*dep*dim*dim, dtype='uint32').reshape((ker,dep,dim,dim))
for r in range(0,dim):
    for d in range(0,dep):
        for c in range(0,dim):
            lis = out[:,d,r,c]
            f_out_b.write(bytearray(lis))
            f_out.write(str(lis)[1:-1]+'\n')






out = np.zeros(ker*dep*dim*dim, dtype='uint32').reshape((ker,dep,dim,dim))
for k in range(0,ker):
    for l in range(0,dep):
        kk = np.rot90(ker_l_3[k,l].reshape((3,3)),2)
        res = sg.convolve(in_l[:,:,l],kk , "valid").astype(int)
        res = np.bitwise_and(res, 0xff)
        out[k,l,:,:]=res
# print(out[1,1,:,:]);print('______')
# out = np.arange(ker*dep*dim*dim, dtype='uint32').reshape((ker,dep,dim,dim))

f_out = open("out_3x3.txt","w")
f_out_b = open("out_3x3.bin","wb")
for r in range(0,dim):
    for d in range(0,dep):
        for c in range(0,dim):
            lis = out[:,d,r,c]
            f_out_b.write(bytearray(lis))
            f_out.write(str(lis)[1:-1]+'\n')