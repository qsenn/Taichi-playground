import taichi as ti
import taichi.math as tm
import os

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

output_dir = "./output"
video_manager = ti.tools.VideoManager(output_dir=output_dir, framerate=30, automatic_build=True)

@ti.func
def complex_sqr(z):
    return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = tm.vec2(-0.8, tm.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02

print("Rendering started...")
for i in range(100):
    paint(i * 0.03)
    
    video_manager.write_frame(pixels.to_numpy())
    
    if i % 10 == 0:
        print(f"Frame {i}/100 rendered")

video_manager.make_video(gif=True, mp4=True)
old_mp4 = os.path.join(output_dir, "video.mp4")
new_mp4 = os.path.join(output_dir, "hello_world.mp4")
old_gif = os.path.join(output_dir, "video.gif")
new_gif = os.path.join(output_dir, "hello_world.gif")

if os.path.exists(old_mp4):
    os.rename(old_mp4, new_mp4)
if os.path.exists(old_gif):
    os.rename(old_gif, new_gif)
print(f"Rendering finished! Check the './output' directory.")