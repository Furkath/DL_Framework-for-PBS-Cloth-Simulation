import taichi as ti
import numpy as np
#ti.init(arch=ti.vulkan)

ti.init(arch=ti.cuda)#,default_fp=ti.f64)
#ti.init(arch=ti.vulkan)#,default_fp=ti.f64)

pressure = 3000000/2/2/4 #50000/2/2/4

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0]) #ti.Vector([0, 0, 0])
spring_Y = 1e4#3e4
dashpot_damping = 3e4#1e4
drag_damping = 3#1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]


#read data
data = np.load("./data/simuPressData.npz")
xdata = data['dataX']
#vdata = data['dataV']
xdata =  np.transpose(xdata, (0, 2, 3, 1))
xdata=np.float32(xdata)
print(xdata.shape)
#frames = []
length = 100
#length = xdata.shape[0]
for tim in range(length):
    xi = ti.Vector.field(3, dtype=ti.f32, shape=(xdata.shape[1],xdata.shape[2]) )
    xi.from_numpy(xdata[tim])
    frames.append(xi)
    print(tim)
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))
print(frames[10])

if_bc = ti.field(int, shape=(n, n)) # you can assign various values to it, for different kinds of bc.

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

@ti.kernel
def select_bc():
    for num in ti.grouped(x):
        #if num[0] == 0 and num[1]==0:
        if num[0] == 0 or num[0] == n-1 or num[1] == 0 or num[1] == n-1 : 
            if_bc[num] = 1
        else:
            if_bc[num] = 0

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]

    #print(x)


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if ( (j) // 4) % 3 == 0:
            colors[i * n + j] = (126/255, 47/255, 142/255)
        elif ( (j) // 4) % 3 == 1:
            colors[i * n + j] = (119/255, 172/255, 48/255)
        else:
            colors[i * n + j] = (162/255, 20/255, 47/255)

initialize_mesh_indices()

spring_offsets = []
spring_neighbors=[]
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

else:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))

spring_neighbors.append( ti.Vector([-1, 0]) )
spring_neighbors.append( ti.Vector([ 0, 1]) )
spring_neighbors.append( ti.Vector([ 1, 0]) )
spring_neighbors.append( ti.Vector([ 0,-1]) )
#spring_neighbors.append( ti.Vector([-1, 0]) )

@ti.kernel
def readFrame(framei: ti.template() ): 
    x = framei

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        j2 = ti.Vector([0, 0]) 
        for spring_offset in ti.static(spring_offsets): 
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n: # you may use a bool list to store inner points if not a rectangle shape!
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        # Pressure
        #for pair_ in range(4):
        for j1 in ti.static(spring_neighbors):
            #j2 = ti.Vector([j1[1],-j1[0]])  #which is faster?
            j2[0]=j1[1]
            j2[1]=-j1[0]
            ij1=i+j1
            ij2=i+j2
            #print(j2)
            #j1 = spring_neighbors[pair_]
            #j2 = spring_neighbors[pair_+1]
            if 0 <= ij1[0] < n and 0 <= ij1[1] < n and 0 <= ij2[0] < n and 0 <= ij2[1] < n : #as above
                x_ij1= x[i]-x[ij1]
                x_ij2= x[i]-x[ij2]
                force += pressure*ti.math.cross(x_ij1, x_ij2)
        
        v[i] += force * dt

    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        #### Collison handling
        '''
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        '''
        ####  
        #### BC
        if if_bc[i]  == 1:
            v[i] = ti.Vector([0.0, 0.0, 0.0]) 
        ####
        x[i] += dt * v[i]

#def up_vert(fi):


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", 
                        (2000,2000),
                        #(1000, 1000), 
                        #show_window = False,
                        vsync=True)
window2 = ti.ui.Window("Taichi Cloth Simulation on GGUI 2", 
                        (1000,1000),
                        #(720, 720), 
                        show_window = False,
                        vsync=True)
canvas = window.get_canvas()
canvas2 = window2.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
f_i = 0
#initialize_mass_points()
#print(x)
#select_bc()

#if(True):
while window.running:
    #if current_t > 1.5: #1.5 
    if f_i > xdata.shape[0] - 1 : #1.5 
        # Reset
        #initialize_mass_points()
        current_t = 0
        f_i = 0 

    for i in range(10):#range(substeps):
        #substep()
        x.from_numpy(xdata[f_i])
        #readFrame(frames[f_i])
        current_t += dt
        #f_i += 1
    update_vertices()

    camera.position(0, 0.3, 2)
    camera.lookat(0, 0.4, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    #scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    
    #canvas2.set_image(window.get_image_buffer_as_numpy())
    window.show()
    #window2.show()
