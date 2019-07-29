# 运行该文件用于打包成exe

import os
import PyInstaller.__main__

PyInstaller.__main__.run([
    '--name=%s' % 'HeartRate',
    # '--onefile',
    # '--debug',
    # '--windowed',
    '--add-binary=%s' % 'D:\Anaconda3\envs\dsp\Lib\site-packages\_sounddevice_data\portaudio-binaries\libportaudio64bit.dll;.\_sounddevice_data\portaudio-binaries',
    # '--paths=%s' % 'D:\Anaconda3\envs\dsp\Lib\site-packages\_sounddevice_data\portaudio-binaries',
    '--add-data=%s' % os.path.join('params.json;.'),
    '--icon=%s' % os.path.join('Icon.ico'),
    os.path.join('main.py'),
])