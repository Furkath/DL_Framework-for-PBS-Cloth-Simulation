import taichi as ti
#ti.init(arch=ti.vulkan)  # Alternatively, 
ti.init(arch=ti.cpu)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2  +2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n +4)
colors = ti.Vector.field(3, dtype=float, shape=n * n +4)

bending_springs = False

@ti.kernel
def initialize_mass_points():
    #random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    random_offset = ti.Vector([-0.409743+0.2, 0.3715345]) * 0.1 #read compare data

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


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

    indices[num_triangles*3-6] = n*n 
    indices[num_triangles*3-5] = n*n+1
    indices[num_triangles*3-4] = n*n+2
    indices[num_triangles*3-3] = n*n+3
    indices[num_triangles*3-2] = n*n+2
    indices[num_triangles*3-1] = n*n+1
    colors[ n*n  ] = (114/255*1.5,100/255*1.5,91/255*1.5)
    colors[ n*n+1] = (114/255*1.5,100/255*1.5,91/255*1.5)
    colors[ n*n+2] = (114/255*1.5,100/255*1.5,91/255*1.5)
    colors[ n*n+3] = (114/255*1.5,100/255*1.5,91/255*1.5)

    for i, j in ti.ndrange(n, n):
        if ( ( j) // 4) % 3 == 0:
            colors[i * n + j] = (126/255, 47/255, 142/255)
        elif ( ( j) // 4) % 3 == 1:
            colors[i * n + j] = (119/255, 172/255, 48/255)
        else:
            colors[i * n + j] = (162/255, 20/255, 47/255)

initialize_mesh_indices()

spring_offsets = []
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

@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):
        #drag
        v[i] *= ti.exp(-drag_damping * dt)
        #collision
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            #velocity projection
            normal = offset_to_center.normalized()
            vnrel = min(v[i].dot(normal), 0) 
            #bouncing absorption
            v[i] -= vnrel * normal
            #friction
            if vnrel < 0 :
                v[i] *= 0.95 #v_rel actually, but v_ball  is 0
        x[i] += dt * v[i]

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]
    vertices[ n*n  ] = (-10., -0.5, -70.)
    vertices[ n*n+1] = (-10., -0.5,  10.)
    vertices[ n*n+2] = ( 10., -0.5, -70.)
    vertices[ n*n+3] = ( 10., -0.5,  10.) 

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", 
                                (1000,1000),
                                #(1440,1440),
                                #(2880,2880),
                                #show_window =False, 
                                vsync=True)
window2 = ti.ui.Window("Taichi Cloth Simulation on GGUI 2",
                        (1440,1440),
                        show_window =False,
                        vsync=True)                     
canvas = window.get_canvas()
canvas2 = window2.get_canvas()
canvas.set_background_color((1.5*59/255, 1.5*55/255, 1.5*50/255))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

current_t = 0.0
initialize_mass_points()
gif_f = 0
while window.running:
    if current_t > 1.55:#1.5:
        # Reset
        #exit(0)
        initialize_mass_points()
        current_t = 0

    for i in range(10): #(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 2)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(-1, 1, 1), color=(0.7, 0.7, 0.7))
    scene.point_light(pos=( 1, 1, 1), color=(0.7, 0.7, 0.7))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(92/255, 94/255, 163/255))
    canvas.scene(scene)
    gif_fra= f"{gif_f:0>{6}}"
    #canvas2.set_image(window.get_image_buffer_as_numpy())
#    if gif_f > 290:
#        window2.save_image('GIF_Ball_simu/'+str(gif_f)+'.png')
#        print("done!") 
    #window.save_image('GIF_Ball_simu/frame'+gif_fra+'.png')
    #print(gif_f)
    gif_f += 1
    window.show()
    #window2.show()
