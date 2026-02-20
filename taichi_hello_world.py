import taichi as ti
import taichi.math as tm
import os

ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

# 비디오 저장을 위한 매니저 설정
# 'output' 폴더에 영상이 저장됩니다.
video_manager = ti.tools.VideoManager(output_dir="./output", framerate=30, automatic_build=True)

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

# 총 100프레임(약 3초) 동안 렌더링 후 저장
print("Rendering started...")
for i in range(100):
    paint(i * 0.03)
    
    # 픽셀 데이터를 비디오 매니저에 넘김
    video_manager.write_frame(pixels.to_numpy())
    
    if i % 10 == 0:
        print(f"Frame {i}/100 rendered")

# 비디오 파일 생성 및 GIF 변환 (선택 사항)
video_manager.make_video(gif=True, mp4=True)
print(f"Rendering finished! Check the './output' directory.")