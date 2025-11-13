import librosa
import numpy as np
import io
import base64
import soundfile as sf

def librosa_audio_to_base64(audio_numpy: np.ndarray, sr: int = 16000, format: str = 'wav') -> str:
    """
    将librosa加载的音频数据转换为base64编码字符串
    Args:
        audio_numpy (np.ndarray): librosa加载的音频数据 (shape: (n_samples,))
        sr (int): 采样率
        format (str): 输出音频格式 (支持: 'wav', 'flac', 'ogg', etc.)

    Returns:
        str: base64编码的音频字符串

    Raises:
        ValueError: 如果音频数据或采样率无效
    """
    if not isinstance(audio_numpy, np.ndarray) or audio_numpy.ndim != 1:
        raise ValueError("音频数据必须是1维numpy数组")
    if sr <= 0:
        raise ValueError("采样率必须为正整数")

    # 使用soundfile将numpy数组写入字节流
    byte_io = io.BytesIO()
    sf.write(byte_io, audio_numpy, sr, format=format)
    byte_io.seek(0)  # 重置指针位置

    # 编码为base64
    return base64.b64encode(byte_io.read()).decode('utf-8')


def audio_file_to_base64(audio_path):
    """
    将音频文件转换为base64编码字符串

    Args:
        audio_path (str): 音频文件路径

    Returns:
        str: base64编码的音频字符串

    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件不是支持的音频格式
    """
    # 支持的音频格式
    supported_formats = ('.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a')

    # 检查文件格式
    if not audio_path.lower().endswith(supported_formats):
        raise ValueError(f"不支持的音频格式，支持: {', '.join(supported_formats)}")

    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            return base64.b64encode(audio_data).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    except Exception as e:
        raise Exception(f"处理音频文件时出错: {str(e)}")


def base64_to_numpy_audio(audio_base64: str, sr: int = 16000) -> np.ndarray:
    """
    将base64编码的音频数据转换为librosa可用的NumPy数组
    
    Args:
        audio_base64: base64编码的音频字符串
        sr: 目标采样率（可选，若为None则保留原始采样率）

    Returns:
        audio_np: 音频数据的NumPy数组
        sr: 实际采样率

    Raises:
        ValueError: 如果base64解码失败或音频数据无效
    """
    try:
        # 1. Base64解码为字节流
        audio_bytes = base64.b64decode(audio_base64)

        # 2. 将字节流转换为文件类对象
        audio_io = io.BytesIO(audio_bytes)

        # 3. 使用soundfile读取音频数据
        audio_np, sample_rate = sf.read(audio_io)

        # 4. 转换为单声道（如有需要）
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # 5. 重采样（如果指定了目标sr且与原始sr不同）
        if sr is not None and sr != sample_rate:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr

        return audio_np, sample_rate
    
    except Exception as e:
        raise ValueError(f"音频转换失败: {str(e)}")