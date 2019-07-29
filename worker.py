# -*- coding: utf-8 -*-
import json
import sounddevice as sd
from scipy.io import wavfile
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

import heart_rate_core as hrcore


class WorkerSignals(QObject):
    """
    定义了传递给主程序的信号
    1.result:计算得到的结果
    2.finished:线程可以结束
    """

    result = pyqtSignal(tuple)
    finished = pyqtSignal()


class WorkerCalc(QObject):
    """
    任务,主要提供了几个slot,作为任务的入口点。
    1.计算整段的PCG的心率
    2.计算所选时刻的心率
    3.播放信号片段
    4.获取整段PCG的时域和频域信息
    """

    def __init__(self, json_file):
        super(WorkerCalc, self).__init__()
        self.json_file = json_file
        self.signal = WorkerSignals()
        self.file_path = ""
        self.pick_time = 0

        # params.json中相关参数的声明
        self.method1 = ""
        self.method2 = ""
        self.filter = ""
        self.filter_params = tuple(range(0, 4))
        self.down_fs = 0
        self.win_t = 0
        self.slide_t = 0
        self.win_N = 0
        self.slide_N = 0
        self.bunch_size = 0
        self.lpf_fc = 0
        self.scope_t = []
        self.scope_N = []
        self.height = 0

        self.get_params(self.json_file)

    @pyqtSlot()
    def play_audio(self):
        """
        播放当前的信号片段
        """

        fs, data = wavfile.read(self.file_path)
        win_N = 3 * fs
        start_N = int(self.pick_time - self.win_t) * fs
        seg = data[start_N:start_N + win_N]
        sd.default.samplerate = fs
        sound_seg = np.array(seg, dtype=np.int16)

        sd.play(sound_seg, blocking=True)
        self.signal.result.emit(('^v^', ))  # 只是不想破坏队形,保持跟其他线程一样
        self.signal.finished.emit()

    @pyqtSlot()
    def get_heart_rate_overall(self):  # A slot takes no params
        """
        获取整段PCG的心率。步骤如下:
        1.对信号进行预处理,包括:滤波,欠采样
        2.计算预处理后的整段信号的心率
        计算心率的方法有多种,根据params.json中设置的method使用不同的方法计算
        """

        self.get_params(self.json_file)  # 每次执行算法之前都重新读取参数文件，这样修改参数后不必重启工程
        print(self.scope_t)
        data, fs = hrcore.get_pre_process_data(
            self.file_path, self.filter, self.filter_params, self.down_fs)

        hr_series1 = hrcore.get_hr_series(
            data, fs, self.win_N, self.slide_N, self.method1, self.bunch_size, self.lpf_fc, self.scope_t)

        hr_series2 = hrcore.get_hr_series(
            data, fs, self.win_N, self.slide_N, self.method2, self.bunch_size, self.lpf_fc, self.scope_t)

        hr_fs = 1.0 / self.slide_t
        time = self.win_t + np.arange(0, len(hr_series1)) * 1.0 / hr_fs
        hrs1 = (time, hr_series1)
        hrs2 = (time, hr_series2)

        self.signal.result.emit((hrs1, hrs2))
        self.signal.finished.emit()

        if __name__ == '__main__':
            """直接运行时,方便调试"""
            return hrs1, hrs2

    @pyqtSlot()
    def get_heart_rate_pick(self):
        """
        计算所选时刻的心率。步骤如下:
        1.对整段信号进行预处理,包括滤波,欠采样
        2.截取所选时刻的信号窗口
        3.根据params.json中设置的方法计算心率
        """
        self.get_params(self.json_file)  # 每次执行算法之前都重新读取参数文件，这样修改参数后不必重启工程
        print(self.win_t)
        data, fs = hrcore.get_pre_process_data(
            self.file_path, self.filter, self.filter_params, self.down_fs)

        print("pick_time is %f" % self.pick_time)
        start_time = self.pick_time - self.win_t
        pick_N = int(start_time * fs)

        seg, fs = hrcore.get_seg_data(data, fs, pick_N, self.win_N)
        time = start_time + np.arange(0, len(seg)) * 1.0 / fs

        seg_all1 = hrcore.get_seg_all(
            seg, fs, start_time, self.method1, self.bunch_size, self.lpf_fc, self.scope_t, self.height)
        _, _, _, peaks, peak, _ = seg_all1
        print("here is get_heart_rate_pick:")
        print(peaks)
        print("peak_time is %f; peaks is %f" % (peak[0], peak[1]))
        print()
        seg_all2 = hrcore.get_seg_all(
            seg, fs, start_time, self.method2, self.bunch_size, self.lpf_fc, self.scope_t, self.height)
        self.signal.result.emit((seg_all1, seg_all2))
        self.signal.finished.emit()

        if __name__ == '__main__':
            """直接运行时,方便调试"""
            return seg_all1, seg_all2

    @pyqtSlot()
    def get_whole_pcg(self):
        """
        获取整段PCG的时域和频域信息

        """
        self.get_params(self.json_file)  # 每次执行算法之前都重新读取参数文件，这样修改参数后不必重启工程
        wave, fs = hrcore.get_wav_data(self.file_path)
        time_wave = np.arange(0, len(wave)) * 1.0 / fs
        freqs, amps = hrcore.get_spectrum(wave, fs)

        down_data, down_fs = hrcore.get_downsample_data(
            wave, fs, self.down_fs)
        wave_proc, fs = hrcore.get_filted_data(
            down_data, self.down_fs, self.filter, self.filter_params)

        time_proc = np.arange(0, len(wave_proc)) * 1.0 / fs
        freqs_proc, amps_proc = hrcore.get_spectrum(wave_proc, fs)
        pcg_t = (time_wave, wave)
        pcg_f = (freqs, amps)
        pcg_proc_t = (time_proc, wave_proc)
        pcg_proc_f = (freqs_proc, amps_proc)
        whole_pcg = (pcg_t, pcg_f, pcg_proc_t, pcg_proc_f)

        self.signal.result.emit(whole_pcg)
        self.signal.finished.emit()

    def get_params(self, json_file):
        """
        辅助函数
        解析params.json文件中设置的参数

        Parameters
        ----------
        json_file : str
            json文件的路径
        """

        with open(json_file, encoding='UTF-8') as f_obj:
            params = json.load(f_obj)

        if isinstance(params['method'], list):
            self.method1 = params['method'][0]
            self.method2 = params['method'][1]
        self.filter = params['filter']
        self.filter_params = params['filter_params']
        self.down_fs = params['down_fs']
        self.win_t = params['win_t']
        self.slide_t = params['slide_t']
        self.bunch_size = params['bunch_size']
        self.lpf_fc = params['lpf_fc']
        self.scope_t = params['scope_t']
        self.height = params['height']

        self.win_N = int(self.win_t * self.down_fs)
        self.slide_N = int(self.slide_t * self.down_fs)
        self.scope_N = [int(num * self.down_fs) for num in self.scope_t]


if __name__ == '__main__':
    """
    直接运行核心任务,而不经过界面。方便调试
    """
    import matplotlib.pyplot as plt

    json_file = "params.json"
    worker = WorkerCalc(json_file)
    worker.file_path = \
        "E:\\我的电脑收纳\\001 - Inbox - 中转站，先到碗里来\\实验测试心音\\fanxiao.wav"

    hrs1, hrs2 = worker.get_heart_rate_overall()
    plt.plot(hrs1[0], hrs1[1])

    worker.pick_time = 11
    seg_all1, seg_all2 = worker.get_heart_rate_pick()

    seg_with_t, env_with_t, corrs_with_t,
    peaks_ind_with_t, peak_ind_with_t = seg_all1

    fig, axarr = plt.subplots(3, 1)
    axarr[0].plot(seg_with_t[0], seg_with_t[1])
    axarr[1].plot(env_with_t[0], env_with_t[1])
    axarr[2].plot(corrs_with_t[0], corrs_with_t[1])
    axarr[2].plot(peaks_ind_with_t[0], peaks_ind_with_t[1], 'rx')
    axarr[2].plot(peak_ind_with_t[0], peak_ind_with_t[1], 'bx')
    plt.show()
