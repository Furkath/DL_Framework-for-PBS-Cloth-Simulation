import taichi as ti 
import numpy as np

ti.init(arch=ti.cuda,default_fp=ti.f64)

n=4
bob=np.ones((n,n,3))
jack = []
for i in range(10):
        v = ti.Vector.field(3, dtype=float, shape=(n, n))
        #v.fill(i)
#        print(v.type)
        v.from_numpy(bob*i*1.1)
        jack.append(v)

#print(jack[5].type)
print(jack)


dick=np.ones((7,n,n,3))
print(dick[6])
#@ti.kernel
#def fill():
#    for i,j in v:
#        #v[1,1]=50
#        print(i.dtype)
#        print(j.dtype)
#        v[i,j]=50

#fill()

#vv = ti.Vector.field(3, dtype=float, shape=(2,n, n))
#print(vv)

#vv[(1,2,1,1)]=100

#vs=vv[1]
#print(vs)
