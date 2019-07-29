# -*- coding: utf-8 -*-
import math
import wave
from math import exp, pi, cos

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

# ------------------------<start:对外函数----------------------
# 使用采样下标,不用对应的时刻。时刻的方式等到要画图显示再使用
# -----<start:组合功能------
# 对整段处理的对外函数都是以文件路径作为传入参数的,包括
# 获取原始数据,获取滤波后的数据,获取欠采样后的数据
# 1.获取原始数据


def get_wav_data(file_path):
    """
    1.获取整段原始数据

    读取文件,取单通道并归一化的数据

    Parameters
    ----------
    file_path : string
        wav文件路径

    Returns
    -------
    data: array-like
        单通道归一化的数据
    fs: int
        采样频率
    """

    data, fs = wav_read(file_path)
    return data, fs


# 2.获取滤波后数据
def get_filted_data(data, fs, method, params=None):
    """
    2.获取整段滤波后数据

    根据传入的滤波方法和滤波参数对信号进行滤波

    Parameters
    ----------
    data : array-like
        需要滤波的信号
    fs : int
        信号的采样频率
    method : string
        滤波方法,可选{'low', 'high', 'band'}
    params : tuple, optional
        滤波器的参数,
        对于'low'类型,参数分别是(阶数, 截止频率)
        对于'high'类型,参数分别是(阶数, 截止频率)
        对于'band'类型,参数分别是(高通阶数, 低通阶数, 高通截止频率, 低通截止频率)

    Returns
    -------
    filted_data: array-like
        滤波后的信号
    fs: int
        滤波后的信号的采样频率

    """
    filted_data = []
    if method == 'low':
        if params is None:
            N, fc = 4, 150
        else:
            N, fc = params
        filted_data = low_pass(data, fs, N=N, fc=fc)
    elif method == 'high':
        if params is None:
            N, fc = 4, 25
        else:
            N, fc = params
        filted_data = high_pass(data, fs, N=N, fc=fc)
    elif method == 'band':
        if params is None:
            N1, N2, f1, f2 = (2, 2, 25, 150)
        else:
            N1, N2, f1, f2 = params
        filted_data = band_pass(data, fs, N1=N1, N2=N2, f1=f1, f2=f2)
    elif method == 'gammatone':
        filted_data = g_filter(data, fs)
    else:
        pass
    return filted_data, fs


# 3.获取欠采样后数据
def get_downsample_data(data, fs, down_fs):
    """
    3.获取整段欠采样数据

    Parameters
    ----------
    data : array-like
        需要欠采样的信号
    fs : int
        原信号的采样频率
    down_fs : int
        欠采样后的频率

    Returns
    -------
    down_data: array-like
        欠采样后的信号
    fs: int
        欠采样后的信号采样频率
    """

    down_data, down_fs = downsample(data, fs, down_fs)
    return down_data, down_fs


# 获取预处理后的数据
# get_pre_process_data = get_downsample_data
def get_pre_process_data(file_path, filter='low', params=None, down_fs=1000):
    """
    组合起来,获取预处理后的数据

    Parameters
    ----------
    file_path : string
        文件路径
    filter : str, optional
        滤波方式 (默认低通)
    params : tuple, optional
        滤波器参数
    down_fs : int, optional
        欠采样频率 (默认1000)

    Returns
    -------
    proc_data: array-like
        预处理后的信号
    proc_fs: int
        预处理后信号的采样频率
    """
    data, fs = get_wav_data(file_path)
    down_data, down_fs = get_downsample_data(data, fs, down_fs)
    filted_data, filted_fs = get_filted_data(
        down_data, down_fs, filter, params)
    proc_data = filted_data
    proc_fs = filted_fs

    return proc_data, proc_fs


# 对一小段信号进行处理时都是以data作为传入参数的
# 4.获取一段预处理后的数据
def get_seg_data(data, fs, start_N, win_N):
    """
    获取所给信号的一个片段

    Parameters
    ----------
    data : array-like
        源信号
    fs : int
        采样频率
    start_N : int
        截取的起点
    win_N : int
        需要截取的长度

    Returns
    -------
    seg: array-like
        截取出来的信号
    fs: int
        截取的信号的采样频率

    """

    seg = data[start_N:start_N + win_N]
    return seg, fs


def get_envelope_data(data, fs, method='square', bunch_size=10, lpf_fc=8):
    """
    5.获取某一段信号的包络

    Parameters
    ----------
    data : array-like
        信号
    fs : int
        信号的采样频率`
    method : str, optional
        求包络的方法,{'square', 'homomorphic'} (分别是平方包络和同态包络)

    Returns
    -------
    envelope: array-like
        包络
    fs: int
        包络的采样频率(同态包络不会改变采样频率,但是平方包络会使采样频率下降)
    """

    envelope = []
    if method == 'square':
        envelope, fs = square_envelope(data, fs, sum_N=bunch_size, lpf_fc=lpf_fc)
    elif method == 'homomorphic':
        envelope, fs = homomorphic_envelope(data, fs, lpf_fc=lpf_fc)

    return envelope, fs


def get_corrs_data(data, fs):
    """
    6.获取某一段信号的自相关

    Parameters
    ----------
    data : array-like
        信号
    fs : int
        采样频率

    Returns
    -------
    corrs: array-like
        自相关函数
    fs: int
        自相关函数的采样频率
    """

    corrs = autocorr(data)
    return corrs, fs


def get_peaks_ind(data, scope=None, height=None):
    """
    7.获取某一段信号的峰值组

    Parameters
    ----------
    data : array-like
        信号
    scope : tuple, optional
        峰值的所搜范围 (默认是整段信号)
    height : float, optional
        峰值的最小阈值 (默认没有设置最小阈值)

    Returns
    -------
    peaks_ind: array
        所有峰值索引的集合
    peaks: array
        所有峰值的集合
    """

    peaks_ind, peaks = find_peaks_(data, scope, height)
    return peaks_ind, peaks


def get_peak_ind(data, peaks_ind):
    """
    8.获取某一段信号最后选择的峰值

    Parameters
    ----------
    data : array-like
        信号
    peaks_ind : array-like
        已经获得的信号的峰值组

    Returns
    -------
    peak_ind: int
        最后选择的峰值的索引
    peak: float
        最后选择的峰值

    """

    peak_ind = select_peak(data, peaks_ind)
    peak = data[peak_ind]
    return peak_ind, peak


def get_seg_all(data, fs, start_time, method='square', bunch_size=10, lpf_fc=8,
                scope_t=None, height=None):
    """
    9.获取某一段信号的全部有用信息,包络和自相关,峰值

    Parameters
    ----------
    data : 信号
        信号
    fs : int
        采样频率
    start_time : int
        本信号在整段信号中的位置
    method : str, optional
        求包络的方法 {'square', 'homomorphic'} (平方包络和同态包络,默认是平方包络)
    scope_t : tuple, optional
        峰值的搜索范围 (默认是本信号的整段)
    height : int, optional
        峰值的最小阈值 (默认没有最小阈值)

    Returns
    -------
    seg: tuple
        所截取的信号,(time, data)
    env: tuple
        所截取信号的包络(time_env, envelope)
    corrs: tuple
        包络的自相关(time_corrs, corrs)
    peaks: tuple
        自相关的峰值组(time_peaks, peaks)
    peak: tuple
        最后选择的自相关的峰值(time_peak, peak)
    scope: tuple
        峰值搜索范围(scope_l, scope_r)

    """

    time = start_time + np.arange(0, len(data)) * 1.0 / fs
    envelope, fs = get_envelope_data(data, fs, method=method, bunch_size=bunch_size, lpf_fc=lpf_fc)
    time_env = start_time + np.arange(0, len(envelope)) * 1.0 / fs

    corrs, fs = get_corrs_data(envelope, fs)
    time_corrs = start_time + np.arange(0, len(corrs)) * 1.0 / fs

    scope_N = [int(N * fs) for N in scope_t]
    peaks_ind, peaks = get_peaks_ind(corrs, scope_N, height)
    if len(peaks_ind) == 0:
        peaks_ind = np.append(peaks_ind, scope_N[1] - 1)
        peaks = corrs[peaks_ind]
        print("%s window: %s No peaks!!" % (str(start_time), method))
        # try:
        #     raise ValueError("empty peaks!")
        # except ValueError as exp:
        #     print("Error", exp)

    time_peaks = start_time + peaks_ind * 1.0 / fs

    peak_ind, peak = get_peak_ind(corrs, peaks_ind)
    time_peak = start_time + peak_ind * 1.0 / fs

    time_scope_l = start_time + scope_t[0]
    time_scope_r = start_time + scope_t[1]

    seg = (time, data)
    env = (time_env, envelope)
    corrs = (time_corrs, corrs)
    peaks = (time_peaks, peaks)
    peak = (time_peak, peak)
    scope = (time_scope_l, time_scope_r)

    return seg, env, corrs, peaks, peak, scope


def get_hr(data, fs, start_time, method='square', bunch_size=10, lpf_fc=8, scope_t=None, height=None):
    """
    10.获取某一段信号的心率

    Parameters
    ----------
    data : array-like
        信号片段
    fs : int
        采样频率
    start_time : int
        本信号片段在整段信号中的位置
    method : str, optional
        求包络的方法{'square', 'homomorphic'} (默认是平方包络)
    scope_t : tuple, optional
        峰值的搜索范围(默认是本信号片段的整段)
    height : int, optional
        峰值的最小阈值(默认没有最小阈值)

    Returns
    -------
    heart_rate: float
        本信号片段的心率

    """

    _, _, _, _, (time_peak, _), _ = get_seg_all(
        data, fs, start_time, method, bunch_size, lpf_fc, scope_t, height)
    if start_time >= 35 and start_time <=45:
        print("start_time is %f, peak_time is %f" % (start_time, time_peak))
    heart_rate = 60 * 1.0 / (time_peak - start_time)
    return heart_rate


def get_hr_series(data, fs, win_N, slide_N, method='square', bunch_size=10, lpf_fc=8,
                  scope_t=None, height=None):
    """
    11.获取整段心率

    Parameters
    ----------
    data : array-like
        整段信号
    fs : int
        采样频率
    win_N : int
        计算心率所需窗口长度
    slide_N : int
        计算心率的滑动步长
    method : str, optional
        求包络的方法{'square', 'homomorphic'} (默认是平方包络)
    scope_t : tuple, optional
        峰值的搜索范围 (默认是每个信号片段的整个窗长)
    height : float, optional
        峰值的最小阈值 (默认没有最小阈值)

    Returns
    -------
    hr_series: array-like
        整段信号的心率
    """

    N = len(data)
    cnt = segment_cnt(N, win_N, slide_N)
    hr_series = list(range(0, cnt))
    for i in range(0, cnt):
        start_N = i * slide_N
        start_time = start_N / fs
        seg = data[start_N: start_N + win_N]
        hr_series[i] = get_hr(seg, fs, start_time, method, bunch_size, lpf_fc, scope_t, height)
    return hr_series


def get_spectrum(data, fs):
    """
    获取信号的频谱

    Parameters
    ----------
    data : array-like
        信号
    fs : int
        采样频率

    Returns
    -------
    freqs: array-like
        频谱上的频率点
    amps: array-like
        频谱上的频谱幅值
    """

    freqs, amps = spectrum(data, fs)
    return freqs, amps
# ----->end:组合功能------

# ------------------------<start:对外函数----------------------

# ------------------------<start:底层函数----------------------
# 这些都是根据数据,采样率等参数来计算的,使用的全都是采样下标,而不是时刻点

# -----<start:组合功能------
# 预处理

# ----->end:组合功能------

# -----<start:单个功能------
# 读取文件


# def wav_read(filename):
#     """
#     读取wav文件,返回单通道且归一化的数据

#     Parameters
#     ----------
#     filename:
#         wav文件路径

#     Returns
#     -------
#     fs: int
#         wav文件的采样频率
#     data: numpy array
#         从wav文件中读取的数据,只取单通道,并归一化
#     """
#     fs, data = wavfile.read(filename)

#     # 取单通道数据
#     if data.ndim == 1:
#         pass
#     else:
#         data = data.T[0]
#     # 归一化
#     data = data * 1.0 / (max(abs(data)))
#     return data, fs

def wav_read(filename, way='rb'):
    """
    读取wav文件，取两个通道的数据，并归一化
    :param filename: wav文件名
    :param way: 文件打开方式，默认二进制打开
    :return: wave_data, time, framerate
    """
    # 以二进制的方式打开
    f = wave.open(filename, way)
    # 获取wav文件的参数
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 从wave文件中读取所有的数据(二进制形式)
    bin_data = f.readframes(nframes)
    f.close()
    wave_data = np.ndarray(shape=(nframes, ))
    if sampwidth == 2:
        # 将数据转成short类型,存储在ndarray(多维数组结构中)
        wave_data = np.frombuffer(bin_data, dtype=np.short)
    elif sampwidth == 1:
        wave_data = np.frombuffer(bin_data, dtype=np.uint8)
        wave_data = wave_data.astype(np.short)
        wave_data -= 128

    # 幅值归一化
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    # 将数据reshape为2列，至于多少行，numpy自己算
    # （本来wav文件里的数据就是左右两个通道交替存放的，变成2列，那么每一列就是一个通道）
    if nchannels == 2:
        wave_data.shape = -1, 2
        # 转置，将原本2列的通道数据转成2行
        wave_data = wave_data.T
        wave_data = wave_data[0, :]
    # 生成时间变量列表
    time = np.arange(0, nframes) * (1.0 / framerate)
    return wave_data, framerate


# 滤波
def low_pass(x, fs, N=4, fc=150):
    """
    低通滤波器

    Parameters
    ----------
    x : array_like
        一维的array
    fs : int
        数据的采样频率
    N : int, optional
        低通滤波器的阶数 (the default is 4)
    fc : int, optional
        低通滤波器的截止频率 (the default is 150Hz)

    Returns
    -------
    y: array
        滤波后的数据
    """

    b, a = signal.butter(N, fc * 2 / fs, 'low')
    y = signal.lfilter(b, a, x)
    return y


def high_pass(x, fs, N=4, fc=25):
    """
    高通滤波器

    Parameters
    ----------
    x : array_like
        一维的array
    fs : int
        数据的采样频率
    N : int, optional
        高通滤波器的阶数 (the default is 4)
    fc : int, optional
        高通滤波器的截止频率 (the default is 25Hz)

    Returns
    -------
    y : array
        滤波后的数据
    """

    b, a = signal.butter(N, fc * 2 / fs, 'high')
    y = signal.lfilter(b, a, x)

    return y


def band_pass(x, fs, N1=2, N2=2, f1=25, f2=400):
    """
    带通滤波器

    Parameters
    ----------
    x : array_like
        一维的array
    fs : int
        数据的采样频率
    N1 : int, optional
        高通的阶数 (the default is 2)
    N2 : int, optional
        低通的阶数 (the default is 2)
    f1 : int, optional
        高通的截止频率 (the default is 25Hz)
    f2 : int, optional
        低通的截止频率 (the default is 400Hz)

    Returns
    -------
    y : array
        滤波后的数据
    """

    b, a = signal.butter(N1, f1 * 2 / fs, 'high')
    y = signal.lfilter(b, a, x)
    b, a = signal.butter(N2, f2 * 2 / fs, 'low')
    y = signal.lfilter(b, a, y)
    # 归一化
    y = y * 1.0 / (np.max(np.abs(y)))

    return y

def g_filter(data, fs):
    FCore1 = 20
    FCore2 = 45
    FCore3 = 100
    kexi1 = 0.23
    kexi2 = 0.3
    kexi3 = 0.5
    FBand1 = FCore1*kexi1
    FBand2 = FCore2*kexi2
    FBand3 = FCore3*kexi3
    Order1 = 2
    Order2 = 2
    Order3 = 2
    t_S1 = np.arange(0, 0.1, 1/fs)
    S1_dura = len(t_S1)
    G1 = [0]*S1_dura
    G2 = [0]*S1_dura
    G3 = [0]*S1_dura
    for i in range(0, S1_dura):
        G1[i] = pow(t_S1[i], Order1)*exp(-2*pi*FBand1*t_S1[i])*cos(2*pi*FCore1*t_S1[i])
        G2[i] = pow(t_S1[i], Order2)*exp(-2*pi*FBand2*t_S1[i])*cos(2*pi*FCore2*t_S1[i])
        G3[i] = pow(t_S1[i], Order3)*exp(-2*pi*FBand3*t_S1[i])*cos(2*pi*FCore3*t_S1[i])

    G1 = [0.8*i/(max(G1)*S1_dura) for i in G1]
    G2 = [0.8*i/(max(G2)*S1_dura) for i in G2]
    G3 = [0.008*i/(max(G3)*S1_dura) for i in G3]

    dataFiltered = np.convolve(data, G1, 'same') + \
        np.convolve(data, G2, 'same')+np.convolve(data, G3, 'same')
    dataFiltered /= max(dataFiltered)
    return dataFiltered

# 重采样
def downsample(x, fs, down_fs):
    """
    欠采样,从fs到down_fs。

    使用间隔的方法欠采样,保证采样前后数据的实际时间长度基本一致

    Parameters
    ----------
    x : array-like
        数据
    fs : int
        原始采样频率
    down_fs : int
        欠采样后的频率

    Returns
    -------
    y : array-like
        欠采样后的数据
    down_fs : int
        欠采样后的频率
    """
    N = x.size  # 原始数据的长度
    down_N = math.floor(N * down_fs / fs)   # 欠采样后数据的长度
    step = math.floor(fs / down_fs)     # 欠采样取点的间隔
    x = x[::step]   # 间隔取点
    y = x[:down_N]  # 保证欠采样后实际时间长度与原始一致
    return y, down_fs


# 计算分窗数量
def segment_cnt(x_N, win_N, slide_N):
    """
    根据数据长度, 窗口长度, 滑动长度确定可以分成多少个窗

    当x_N < win_N,显然一个窗口都没有
    当x_N > win_N,那么可以滑动的次数+1,就等于分窗数量

    Parameters
    ----------
    x_N : int
        数据长度
    win_N : int
        窗口长度
    slide_N : int
        滑动长度

    Returns
    -------
    cnt : int
        可以分窗的数量
    """
    cnt = 0
    if x_N < win_N:
        cnt = 0

    else:
        cnt = math.ceil((x_N - win_N + 1) / slide_N)
    return cnt

# 截取窗口


def segment(x, start_N, win_N):
    """
    根据起点和所需长度截取窗口数据

    [description]

    Parameters
    ----------
    x : array-like
        一维数据
    start_N : int
        截取的起点
    win_N : int
        窗口的长度

    Returns
    -------
    y : array-like
        截取出来的窗口数据
    """
    y = x[start_N: start_N + win_N]
    return y


# 计算平方包络
def square_envelope(x, fs, sum_N, lpf_fc):
    """
    求给定信号的平方包络

    平方包络其实是每sum_N个点求一次平方和

    Parameters
    ----------
    x : array-like
        一维数组
    fs : int
        采样频率
    sum_N : int
        每sum_N个点求一次

    Returns
    -------
    y : array-like
        一维数组
    fs : int
        求包络后的采样频率
    """

    x_N = len(x)
    y_N = math.floor(x_N / sum_N)   # 可以计算的包络点数

    y = np.zeros(y_N)
    for i in range(1, y_N):
        seg_x = x[(i - 1) * sum_N:i * sum_N]  # 用sum_N个原始数据来算一个包络点
        y[i] = np.sum(np.power(seg_x, 2)) # 平方包络其实就是平方和
        # y[i] = np.sqrt(np.mean(seg_x**2))
    if len(y) == 0:
        print("here")
    fs /= sum_N
    # if lpf_fc:
    #     y = low_pass(y, fs, N=4, fc=lpf_fc)
    y = y / np.max(np.abs(y))   # 归一化

    
    return y, fs

# 计算同态包络


def homomorphic_envelope(x, fs, lpf_fc=8):
    """
    求数据的同态包络

    Parameters
    ----------
    x : array-like
        一维数组
    fs : int
        采样频率
    lpf_fc : int
        低通滤波器的截止频率

    Returns
    -------
    y : array-like
        一维数组
    fs : int
        求包络后的采样频率
    """
    b, a = signal.butter(1, 2 * lpf_fc / fs, 'low')
    x_hilbert = signal.hilbert(x)
    x_filtfilt = signal.filtfilt(
        b, a, np.log10(np.abs(x_hilbert)))

    y = np.exp(x_filtfilt)
    # Remove spurious spikes in first sample:
    y[0] = y[1]
    fs = fs
    # 去偏
    y -= np.mean(y)
    return y, fs


def autocorr(x):
    """
    返回序列x的自相关

    Parameters
    ----------
    x : array-like
        一维数组

    Returns
    -------
    y : array-like
        自相关序列
    """
    y = np.correlate(x, x, 'full')
    y = y[len(x) - 1:]
    y /= max(y)
    return y


# 计算自相关(按照公式实现的自相关,时间复杂度高,不要使用)
def autocor2(x):
    x_N = len(x)
    y = np.zeros(x_N)
    for i in range(0, x_N):
        for j in range(0, x_N - i):
            y[i] += x[j] * x[i + j]
        y[i] /= x_N

    y /= y[0]
    return y


# 查找峰值
def find_peaks_(x, scope=None, height=None, delta_N=None, delta_H=None):
    """
    查找给定序列的峰值点

    可以设定搜索范围,峰值点的特征。核心是利用scipy.signal.find_peaks

    Parameters
    ----------
    x : array-like
        一维数组
    scope : tuple, optional
        搜索范围 ((index_l, index_r))
    height : number or ndarray or sequence, optional
        峰值的高度特征 (对应find_peaks的height)
    delta_N : number, optional
        两个peaks之间的最小距离 (对应find_peaks的distance)
    delta_H : number or ndarray or sequence, optional
        两个peaks之间高度差 (对应find_peaks的threshold)

    Returns
    -------
    peaks_ind : 峰值下标
    peaks : 峰值
    """

    peaks_ind, _ = signal.find_peaks(
        x, height=height, threshold=delta_H, distance=delta_N)

    if scope is not None:
        peaks_ind = peaks_ind[np.where(
            (peaks_ind >= scope[0]) & (peaks_ind < scope[1]))]

    peaks = x[peaks_ind]

    return peaks_ind, peaks

# 确认选取哪个峰值


def select_peak(x, peaks_ind):
    """
    确认最后选择哪个峰值

    Parameters
    ----------
    x : array-like
        一维数组
    peaks_ind : arrary
        一组峰值的下标

    Returns
    -------
    peak_ind: int
        最后确认选择的峰值的下标
    """

    # TODO:这里是假的,后面修改
    tmp_arr = [x[ind] for ind in peaks_ind]

    peak_ind = peaks_ind[np.argmax(tmp_arr)]
    return peak_ind

# ----->end:单个功能------


# ----->start:辅助的功能------
def spectrum(x, fs):
    """计算信号的频谱
    :param x: 信号数据
    :param fs: 采样频率
    :return: freqs, amps(频率点及对应的幅值)
    """
    N = x.size
    amps = abs(np.fft.rfft(x))
    freqs = np.linspace(0, fs / 2, N / 2 + 1)
    return freqs, amps

# # 将采样下标转成时刻点
# def index2time(index, start_time, fs):
#     """
#     将索引转换成时间

#     Parameters
#     ----------
#     index : number,list,tuple,array-like
#         需要转换的索引
#     start_time : float
#         索引对应的时间
#     fs : int
#         采样频率

#     Returns
#     -------
#     float
#         索引对应的时间
#     """

#     time = start_time + index * 1.0 / fs
#     return time

# # 将时刻点转成采样下标


# def time2index(time, start_ind, fs):
#     """
#     将时间转换成索引

#     Parameters
#     ----------
#     time : number,list,tuple,array-like
#         需要转换的时间
#     start_ind : int
#         第一个时刻对应的索引
#     fs : int
#         采样频率

#     Returns
#     -------
#     index: array-like
#         转换后的索引
#     """

#     index = int(start_ind + time * fs)
#     return index
# -----<end:辅助的功能------

# ------------------------>end:底层函数------------------------


# ------------------------<start:测试本文件------------------------
if __name__ == '__main__':
    """
    直接运行核心任务,而不经过界面。方便调试
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    json_file = 'params.json'
    with open(json_file, encoding='UTF-8') as f_obj:
        params = json.load(f_obj)

    method = params['method']
    filter_ = params['filter']
    filter_params = params['filter_params']
    down_fs = params['down_fs']
    win_t = params['win_t']
    slide_t = params['slide_t']
    scope_t = params['scope_t']
    bunch_size = params['bunch_size']
    lpf_fc = params['lpf_fc']
    height = params['height']

    win_N = int(win_t * down_fs)
    slide_N = int(slide_t * down_fs)
    scope_N = [int(num * down_fs) for num in scope_t]

    file_path = "E:\\我的电脑收纳\\001 - Inbox - 中转站，先到碗里来\\实验测试心音\\房性奔马率.wav"

    # 0.确定需要用到什么方法,然后使用不同的方法计算
    for method_ in method:
        # 1.根据文件名获取数据并预处理
        data, fs = get_pre_process_data(
            file_path, filter_, filter_params, down_fs)

        # 2.计算预处理后的整段信号的心率
        hr_series = get_hr_series(data, fs, win_N, slide_N,
                                  method_, bunch_size, lpf_fc, scope_t, height)
        time = win_t + np.arange(0, len(hr_series)) * slide_t

        # 3.绘制心率曲线图
        plt.figure()
        plt.plot(time, hr_series)

        # 4.声明绘制哪个时刻的所以数据
        pick_times = ()

        for pick_time in pick_times:
            # 5.截取计算该时刻心率需要用到的信号窗口
            start_time = pick_time - slide_t
            start_N = int(start_time * fs)
            seg = data[start_N: start_N + win_N]
            print(pick_time)
            time_seg = start_time + np.arange(0, len(seg)) * fs

            # 6.获取需要的所有数据
            seg_with_t, env_with_t, corrs_with_t,
            peaks_with_t, peak_with_t, scope_with_t = get_seg_all(
                seg, fs, start_time, method_, bunch_size, lpf_fc, scope_t, height)

            (time_env, envelope) = env_with_t
            (time_corrs, corrs) = corrs_with_t
            (time_peaks, peaks) = peaks_with_t
            (time_peak, peak) = peak_with_t

            # 7.绘制
            fig, axarr = plt.subplots(3, 1)
            axarr[0].plot(time_seg, seg)
            axarr[1].plot(time_env, envelope)
            axarr[2].plot(time_corrs, corrs)
            axarr[2].plot(time_peaks, peaks, 'rx')
            axarr[2].plot(time_peak, peak, 'bx')

    plt.show()

    # ------------------------>end:测试本文件------------------------
