from moviepy import VideoFileClip  # 修改导入方式

# 读取视频
clip = VideoFileClip("C:/Users/Lenovo/Desktop/雪豹 万圣节喵.mp4")

# 调整分辨率：方法名从 resize 改为 resized
clip = clip.resized(width=240)

# 设置帧率并保存为GIF：建议将 fps 调整为 10-15，避免文件体积过大
clip.write_gif("雪豹 万圣节喵.gif", fps=12, logger=None) 

print("转换完成！")