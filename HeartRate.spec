# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['F:\\HRUI'],
             binaries=[('D:\\Anaconda3\\envs\\dsp\\Lib\\site-packages\\_sounddevice_data\\portaudio-binaries\\libportaudio64bit.dll', '.\\_sounddevice_data\\portaudio-binaries')],
             datas=[('params.json', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='HeartRate',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True , icon='Icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='HeartRate')
