import numpy as np
from PIL import Image, ImageOps
import io
import math
import os
import cv2
from io import BytesIO
import random
from skimage.draw import random_shapes
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


def get_edge_color(image, side='left'):
    """获取图片边缘的颜色"""
    width, height = image.size
    if side == 'left':
        edge_pixels = [image.getpixel((0, y)) for y in range(height)]
    elif side == 'right':
        edge_pixels = [image.getpixel((width-1, y)) for y in range(height)]
    elif side == 'top':
        edge_pixels = [image.getpixel((x, 0)) for x in range(width)]
    elif side == 'bottom':
        edge_pixels = [image.getpixel((x, height-1)) for x in range(width)]
    else:
        edge_pixels = []
    
    # 计算边缘颜色的平均值
    if edge_pixels:
        avg_color = tuple(
            int(sum(channel)/len(edge_pixels)) 
            for channel in zip(*edge_pixels)
        )
        return avg_color
    return (255, 255, 255)  # 默认白色

def rotate_image(input_data, angle, expand=True, fill_color=None):
    """
    旋转图像并补充为矩形图片
    参数:
        input_data: 可以是以下两种形式之一:
                   - 本地图像路径 (str)
                   - 二进制图像数据 (bytes)
        angle: 旋转角度 (度)
        expand: 是否扩展图像大小以适应旋转后的内容
        fill_color: 填充颜色 (RGB元组)，如果为None则自动使用边缘颜色
    
    返回:
        旋转后的二进制图像数据 (bytes)
    """
    # 将二进制数据转换为PIL图像
    try:
        image = Image.open(io.BytesIO(input_data))
    except Exception as e:
        raise ValueError("无法解析图像数据: " + str(e))
    
    # 转换为RGB模式（如果原始图像是RGBA或其他模式）
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 计算旋转后的新尺寸
    if expand:
        # 计算旋转后的新尺寸
        w, h = image.size
        radians = math.radians(angle)
        new_w = int(abs(w * math.cos(radians)) + abs(h * math.sin(radians)))
        new_h = int(abs(w * math.sin(radians)) + abs(h * math.cos(radians)))
    else:
        new_w, new_h = image.size
    
    # 如果没有指定填充颜色，则使用边缘颜色
    if fill_color is None:
        # 根据旋转角度决定使用哪一侧的边缘颜色
        normalized_angle = angle % 360
        if 45 <= normalized_angle < 135:
            fill_color = get_edge_color(image, 'top')
        elif 135 <= normalized_angle < 225:
            fill_color = get_edge_color(image, 'right')
        elif 225 <= normalized_angle < 315:
            fill_color = get_edge_color(image, 'bottom')
        else:
            fill_color = get_edge_color(image, 'left')
    
    # 旋转图像
    rotated = image.rotate(angle, expand=expand, fillcolor=fill_color)
    
    # 如果需要，确保图像尺寸正确（对于某些角度，expand可能不完全准确）
    if expand:
        rotated = ImageOps.pad(rotated, (new_w, new_h), color=fill_color)

    # output_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/difficulty_variation/test_output.jpg"
    # rotated.save(output_path, format='JPEG')  # 或 'PNG'

    # 将旋转后的图像保存为二进制数据
    output = io.BytesIO()
    rotated.save(output, format='PNG')
    
    return output.getvalue()

def add_gaussian_noise(image_data, noise_intensity=0.1):
    """
    向图片添加高斯噪声
    
    参数:
    - image_data: 二进制图片数据(bytes)或NumPy数组
    - noise_intensity: 噪声强度(0-1之间)
    - is_binary_data: 是否为二进制数据(True)或NumPy数组(False)
    
    返回:
    - 添加噪声后的二进制图片数据
    """
    # 1. 将二进制数据转换为NumPy数组(如果是二进制输入)
    image = Image.open(io.BytesIO(image_data))

    if image is None:
        raise ValueError("无法解码图片数据")
    
    # 3. 将图片转换为NumPy数组并转换为浮点型(0-1范围)以便处理
    image = np.array(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # 4. 生成高斯噪声(均值为0，标准差为noise_intensity)
    noise = np.random.normal(0, noise_intensity, image.shape)
    
    # 5. 添加噪声并确保值在0-1范围内
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # 6. 转换回0-255范围并转为uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    # 7. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', noisy_image)
    
    return encoded_image.tobytes()

import numpy as np

def add_salt_pepper_noise(image_data, noise_strength=0.1):
    """
    在图像上添加椒盐噪声
    
    参数:
    image_data: 二进制图片数据(bytes)或NumPy数组
    noise_strength: 0-1之间的浮点数，表示噪声像素比例
 g   
    返回:
    添加椒盐噪声后的二进制图片数据
    """
    # 1. 处理输入数据
    if isinstance(image_data, bytes):
        # 如果是二进制数据，转换为numpy数组
        image = np.array(Image.open(io.BytesIO(image_data)))
    else:
        # 如果是numpy数组，直接使用
        image = image_data
    
    # 2. 确保图像是二值图像（如果不是，转换为0-1范围）
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
        image = (image > 0.5).astype(np.uint8)  # 二值化
    
    # 3. 创建图像副本
    noisy_image = image.copy()
    # 获取图像尺寸
    h, w = image.shape[:2]  # 处理彩色或灰度图像
    total_pixels = h * w
    
    # 4. 计算噪声像素总数
    n_noise = int(noise_strength * total_pixels)
    
    # 5. 如果噪声强度为0则直接返回
    if n_noise == 0:
        return noisy_image
    
    # 6. 生成随机坐标 (展平索引)
    indices = np.random.choice(total_pixels, size=n_noise, replace=False)
    
    # 7. 计算白噪声和黑噪声的数量 (各占一半)
    salt_pixels = n_noise // 2
    pepper_pixels = salt_pixels
    
    # 8. 额外的噪声点随机分配给其中一类
    if n_noise % 2 != 0:
        if np.random.rand() > 0.5:
            salt_pixels += 1
        else:
            pepper_pixels += 1
    
    # 9. 添加白噪声 (盐)
    salt_indices = indices[:salt_pixels]
    noisy_image.flat[salt_indices] = 1
    
    # 10. 添加黑噪声 (椒)
    pepper_indices = indices[salt_pixels:salt_pixels+pepper_pixels]
    noisy_image.flat[pepper_indices] = 0
    
    # 11. 转换回0-255范围并转为uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    # 12. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', noisy_image)
    
    return encoded_image.tobytes()

def add_speckle_noise(image_data, noise_intensity=0.1):
    """
    向图片添加散斑噪声
    
    参数:
    - image_data: 二进制图片数据(bytes)或NumPy数组
    - noise_intensity: 噪声强度(0-1之间)
    
    返回:
    - 添加噪声后的二进制图片数据
    """
    # 1. 将二进制数据转换为NumPy数组(如果是二进制输入)
    image = Image.open(io.BytesIO(image_data))

    if image is None:
        raise ValueError("无法解码图片数据")
    
    # 2. 将图片转换为NumPy数组并转换为浮点型(0-1范围)以便处理
    image = np.array(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # 3. 生成散斑噪声(乘性噪声)
    noise = np.random.normal(0, noise_intensity, image.shape)
    noisy_image = image + image * noise
    
    # 4. 确保值在0-1范围内
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # 5. 转换回0-255范围并转为uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    # 6. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', noisy_image)
    
    return encoded_image.tobytes()


def add_random_occlusion(image_data, percent=20, blocks=5):
    """
    向图片添加随机遮挡
    
    参数:
    - image_data: 二进制图片数据(bytes)
    - percent: 遮挡面积百分比(0-100)
    - blocks: 遮挡块数量
    
    返回:
    - 添加遮挡后的二进制图片数据
    """
    # 1. 解码图片
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("无法解码图片数据")
    
    # 2. 转换为NumPy数组
    image = np.array(image)
    h, w = image.shape[:2]
    
    # 3. 生成随机遮挡形状
    max_size = int((percent/100) * (h*w) / blocks)
    shapes, _ = random_shapes(
        image_shape=(h, w),
        min_shapes=blocks,
        max_shapes=blocks,
        min_size=int(min(h,w)*0.1),
        max_size=int(min(h,w)*0.2),
        # multichannel=len(image.shape)>2,
        # shape='random',
        random_seed=random.randint(0,1000)
    )
    
    # 4. 创建遮挡掩码
    mask = shapes == 1  # 形状区域为1，背景为0
    if len(image.shape) == 3:  # 彩色图像
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    
    # 5. 生成随机颜色遮挡块
    occlusion_colors = np.random.randint(
        0, 256, 
        size=(blocks, 1, 1, 3 if len(image.shape)==3 else 1),
        dtype=np.uint8
    )
    
    ## 6. 应用遮挡
    for i in range(1, blocks+1):
        # 获取当前形状的2D布尔掩码
        shape_mask = (shapes == i)
        
        # 确保掩码是2D的
        if shape_mask.ndim > 2:
            shape_mask = shape_mask.squeeze()
        
        # 获取颜色值
        color = occlusion_colors[i-1]
        
        # 对每个通道分别处理
        for c in range(3):
            # 直接操作图像数组
            image[shape_mask, c] = color[c]

    # for i in range(1, blocks+1):
    #     # 为每个形状分配随机颜色
    #     color = occlusion_colors[i-1]
    #     # 找到当前形状的区域
    #     shape_mask = (shapes == i)
    #     if len(image.shape) == 3:
    #         shape_mask = np.repeat(shape_mask[:,:,np.newaxis], 3, axis=2)
    #     # 应用遮挡
    #     if image.ndim == 2:
    #         image = np.stack([image]*3, axis=-1)  # 转换为 RGB
    #         # 然后对所有通道应用掩码
    #         image[shape_mask, :] = color
    #     else:
    #         image[shape_mask] = color
    
    # 7. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', image)
    
    return encoded_image.tobytes()

import cv2
import numpy as np
from PIL import Image
import io

def add_blur(image_data, blur_intensity=0.1):
    """
    向图片添加模糊效果（模拟失焦或运动模糊）
    
    参数:
    - image_data: 二进制图片数据(bytes)或NumPy数组
    - blur_intensity: 模糊强度(0-1之间)，值越大越模糊
    
    返回:
    - 添加模糊后的二进制图片数据
    """
    # 1. 将二进制数据转换为PIL Image
    image = Image.open(io.BytesIO(image_data))

    if image is None:
        raise ValueError("无法解码图片数据")
    
    # 2. 将图片转换为NumPy数组，并确保是uint8类型
    image = np.array(image)
    
    # 检查数据类型，如果不是uint8，则转换
    if image.dtype != np.uint8:
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255  # 布尔值转0或255
        else:
            image = image.astype(np.uint8)  # 其他类型转uint8
    
    # 3. 根据模糊强度计算内核大小（奇数）
    kernel_size = int(blur_intensity * 50)  # 最大模糊内核约为50
    kernel_size = max(3, kernel_size)  # 最小内核为3
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 确保奇数
    
    # 4. 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # 5. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', blurred_image)
    
    return encoded_image.tobytes()



def add_low_resolution(image_data, reduction_factor=0.1):
    """
    模拟低分辨率效果：先降采样到极低分辨率，再插值回原尺寸
    
    参数:
    - image_data: 二进制图片数据(bytes)
    - reduction_factor: 降采样因子(0-1之间)，值越小分辨率越低
    
    返回:
    - 处理后的二进制图片数据
    """
    # 1. 将二进制数据转换为PIL Image
    image = Image.open(io.BytesIO(image_data))

    if image is None:
        raise ValueError("无法解码图片数据")
    
    # 2. 获取原始尺寸
    original_size = image.size
    original_mode = image.mode
    
    # 3. 计算降采样后的尺寸（至少保留1像素）
    new_width = max(1, int(original_size[0] * reduction_factor))
    new_height = max(1, int(original_size[1] * reduction_factor))
    
    # 4. 降采样到极低分辨率
    small_image = image.resize((new_width, new_height), Image.NEAREST)
    
    # 5. 插值回原始尺寸（使用线性插值）
    low_res_image = small_image.resize(original_size, Image.BILINEAR)
    
    # 6. 转换为numpy数组并确保类型为uint8
    low_res_array = np.array(low_res_image, dtype=np.uint8)  # 显式指定类型
    
    # 7. 处理颜色模式转换
    if original_mode == 'RGB':
        low_res_array = cv2.cvtColor(low_res_array, cv2.COLOR_RGB2BGR)
    
    # 8. 再次检查数据类型（调试用）
    if low_res_array.dtype != np.uint8:
        low_res_array = low_res_array.astype(np.uint8)
    
    # 9. 编码回二进制数据
    _, encoded_image = cv2.imencode('.png', low_res_array)
    
    return encoded_image.tobytes()

def save_binary_images_to_jpg(image_list, variable_names, output_dir='output'):
    """
    将二进制图像数据列表保存为以变量名命名的JPG文件
    
    参数:
    - image_list: 包含二进制图像数据的列表
    - variable_names: 对应的变量名列表
    - output_dir: 输出目录(默认为'output')
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入长度是否匹配
    if len(image_list) != len(variable_names):
        raise ValueError("图像列表和变量名列表长度不匹配")
    
    # 遍历并保存每个图像
    for i, (image_data, var_name) in enumerate(zip(image_list, variable_names)):
        # 构造文件名(替换不合法的文件名字符)
        safe_name = "".join([c if c.isalnum() else "_" for c in var_name])
        filename = f"{safe_name}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # 写入文件
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            print(f"已保存: {filepath}")
        except Exception as e:
            print(f"保存 {filepath} 失败: {str(e)}")

def combine_4_images_square_resized(image_binaries, output_size=(1024,1024)):
    """
    将4张图像拼接成2x2正方形布局，自动调整图片大小
    
    参数:
        image_binaries: 包含4个图像二进制数据的列表
        output_size: 输出图像的总大小 (width, height)
        
    返回:
        拼接后图像的二进制数据
    """
    if len(image_binaries) != 4:
        raise ValueError("需要提供4张图片的二进制数据")
    
    images = [Image.open(io.BytesIO(binary)) for binary in image_binaries]
    
    # 计算每个单元格的大小
    cell_width = output_size[0] // 2
    cell_height = output_size[1] // 2
    
    # 创建新图像
    new_image = Image.new('RGB', output_size)
    
    # 调整并放置每张图片
    positions = [(0, 0), (cell_width, 0), 
                (0, cell_height), (cell_width, cell_height)]
    
    for i, (img, pos) in enumerate(zip(images, positions)):
        # 调整图片大小以适应单元格
        img = img.resize((cell_width, cell_height), Image.LANCZOS)
        new_image.paste(img, pos)
    
    img_byte_arr = io.BytesIO()
    new_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def _load_font(font_path, size):
    """加载字体，优先使用用户提供的字体路径，若无则使用系统默认字体"""
    fallback_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    ]
    
    def try_load(fp):
        try:
            return ImageFont.truetype(fp, size)
        except Exception as e:
            return None

    font = try_load(font_path) or next((try_load(fp) for fp in fallback_fonts), None)
    return font or ImageFont.load_default()

def _measure_text_width(draw, text, font):
    """测量文本的宽度"""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]

def _line_height(font):
    """计算行高"""
    try:
        return sum(font.getmetrics())
    except Exception:
        img = Image.new("L", (10, 10), 0)
        dr = ImageDraw.Draw(img)
        bbox = dr.textbbox((0, 0), "Hg", font=font)
        return bbox[3] - bbox[1]

def _wrap_text_cjk_safe(draw, text, font, max_width):
    """优化的换行函数，支持英文/数字/标点和CJK字符的分词和换行"""
    def wrap_paragraph(paragraph):
        tokens = re.findall(r'\s+|[^\s\u4e00-\u9fff]+|[\u4e00-\u9fff]', paragraph)
        lines = []
        line = ""
        for token in tokens:
            candidate = line + token
            if _measure_text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line.rstrip())
                if _measure_text_width(draw, token, font) > max_width:
                    sub_line = ""
                    for ch in token:
                        if _measure_text_width(draw, sub_line + ch, font) <= max_width:
                            sub_line += ch
                        else:
                            if sub_line:
                                lines.append(sub_line)
                            sub_line = ch
                    if sub_line:
                        line = sub_line
                    else:
                        line = ""
                else:
                    line = token
        if line:
            lines.append(line.rstrip())
        return lines

    return [line for paragraph in str(text).split("\n") for line in wrap_paragraph(paragraph)]

def _layout_lines(draw, text, font, max_width, space_height, padding, line_spacing_ratio):
    """计算并返回文本的布局信息"""
    lines = _wrap_text_cjk_safe(draw, text, font, max_width)
    line_height = _line_height(font)
    line_spacing = max(1, int(line_height * line_spacing_ratio))
    total_height = line_height * len(lines) + line_spacing * (len(lines) - 1)
    fits = total_height <= max(0, space_height - 2 * padding)
    return fits, lines, line_height, line_spacing, total_height

def add_text_to_image_with_space(image_byte, text, text_height_ratio=0.2, position=None, 
                                  font_path="/mmu_cd_ssd/zhangzhenyu06/workspace/fonts/arimo.ttf", 
                                  max_font_size=100, min_font_size=12, padding=12, 
                                  line_spacing_ratio=0.2, allow_expand=True, horizontal_margin=12):
    """将文本添加到图片中，自动调整字体大小和行间距"""
    try:
        image = Image.open(io.BytesIO(image_byte))
    except Exception as e:
        raise ValueError(f"无法解析图像数据: {e}")

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_width, img_height = image.size
    space_height = max(1, int(img_height * text_height_ratio))
    
    position = position if position in ("top", "bottom") else random.choice(["top", "bottom"])
    total_height = img_height + space_height
    canvas = Image.new("RGB", (img_width, total_height), (255, 255, 255))
    
    text_y_offset = 0 if position == "top" else img_height
    canvas.paste(image, (0, space_height) if position == "top" else (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    lo, hi = min_font_size, max_font_size
    best = None
    max_line_width = max(1, img_width - 2 * (padding + horizontal_margin))
    
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        fits, lines, line_height, line_spacing, total_text_height = _layout_lines(
            draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
        )
        
        if fits:
            best = (mid, font, lines, line_height, line_spacing, total_text_height)
            lo = mid + 1
        else:
            hi = mid - 1
    
    if not best:
        font = _load_font(font_path, min_font_size)
        fits, lines, line_height, line_spacing, total_text_height = _layout_lines(
            draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
        )
        if not fits and allow_expand:
            needed_space = total_text_height + 2 * padding
            new_total_height = img_height + needed_space
            new_canvas = Image.new("RGB", (img_width, new_total_height), (255, 255, 255))
            new_canvas.paste(image, (0, needed_space) if position == "top" else (0, 0))
            canvas = new_canvas
            draw = ImageDraw.Draw(canvas)
            space_height = needed_space
        best = (min_font_size, font, lines, line_height, line_spacing, total_text_height)
    
    font_size, font, lines, line_height, line_spacing, total_text_height = best
    fits, lines, line_height, line_spacing, total_text_height = _layout_lines(
        draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
    )

    start_y = text_y_offset + max(padding, (space_height - total_text_height) // 2)
    y = start_y
    for line in lines:
        line_width = _measure_text_width(draw, line, font)
        x = max(padding, (img_width - line_width) // 2)
        draw.text((x, y), line, font=font, fill="black")
        y += line_height + line_spacing

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()


# 均接收二维输入
# rotate_image(input_data, angle, expand=True, fill_color=None):
# add_gaussian_noise(image_data, noise_intensity=0.1)
# add_salt_pepper_noise(image_data, noise_strength)
# add_speckle_noise(image_data, noise_intensity=0.1)
# add_random_occlusion(image_data, percent=20, blocks=5)
# add_blur(image_data, blur_intensity=0.1)
# add_low_resolution(image_data, reduction_factor=0.1)




# input_path = "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/difficulty_variation/image.jpg"

# if isinstance(input_path, str) and os.path.isfile(input_path):
#     # 如果是本地文件路径，读取为二进制数据
#     with open(input_path, 'rb') as f:
#         image_data = f.read()
# elif isinstance(input_path, bytes):
#     # 如果已经是二进制数据，直接使用
#     image_data = input_path
# else:
#     raise ValueError("输入必须是有效的图像文件路径或二进制图像数据")

# out_rotate = rotate_image(image_data,283)
# out_gaus = add_gaussian_noise(image_data,0.5)
# out_salt_pepper = add_salt_pepper_noise(image_data,0.5)
# out_speckle_noise = add_speckle_noise(image_data,0.5)
# # out_random_occlusion = add_random_occlusion(image_data,0.5)
# out_blur = add_blur(image_data,0.5)
# out_low_resolution = add_low_resolution(image_data,0.5)
# save_binary_images_to_jpg(
#     [out_rotate, out_gaus, out_salt_pepper,out_speckle_noise,out_blur,out_low_resolution],
#     ['out_rotate', 'out_gaus', 'out_salt_pepper','out_speckle_noise','out_blur','out_low_resolution'],
#     "/mmu_cd_ssd/zhangzhenyu06/workspace/EasyR1/difficulty_variation/test"
#     )