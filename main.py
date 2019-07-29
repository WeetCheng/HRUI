# -*- coding: utf-8 -*-
import json

from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import QDir, QFileInfo, Qt, QThread
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *

import worker
from Ui_main_window import Ui_MainWindow

"""
主应用,负责掌控全局,绘图。


包含两个窗口类,主窗口和matplotlib窗口。
主窗口的布局如下:
1.上边是菜单栏和工具栏
2.左边是工作空间,在菜单栏或者工具栏中点击打开文件夹可以选择工作空间
3.中间四块核心的画布,用于绘制心音分析相关图形
4.下边是状态栏,用于显示鼠标在画布中的坐标

主窗口类中可以开启以下线程:
1.获取整段的心率
2.获取所选时刻对应窗口计算心率时的所有信息
3.播放心音
4.获取整段的时域和频域信息

各个线程的处理流程如下:
线程1: 双击工作空间的文件 -> on_double_clicked -> 启动线程1 -> 线程1运行
    -> 返回result和finished -> 线程1结束 -> on_result_all -> plot_to_canvas

线程2: 画布1或3中选点 -> on_pick -> 启动线程2 -> 线程2运行 -> 返回result和finished
    -> 线程2结束 -> on_result_pick -> plot_to_canvas

线程3: 点击播放按钮 -> on_play -> 启动线程3 -> 线程3运行 -> 返回result和finished
    -> 线程3结束 -> on_result_play(设置播放按钮图标为unchecked)

线程4: 画布中右键菜单弹出整段 -> on_popup_pcg -> 启动线程4 -> 线程4运行(获取时频信息)
    -> 返回result和finished ->线程4结束 -> on_result_pcg(弹出matplotlib窗口,绘制)

工作空间支持双击运行,四块画布支持鼠标右键菜单弹出

matplotlib窗口:
用于绘制波形,包含工具栏,画布,状态栏
"""


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, json_file, *args, **kargs):
        """
        UI界面相关内容，初始化主要包括以下内容：
        1. 界面的初始化，包括直接使用QtDesigner设计的界面，matplotlib的toolbar，以及默认工作空间
        2. 将所有的属性列在一块，方便查看有哪些属性
        3. 核心工作需要另外开线程来实现，线程相关内容放在init_thread中初始化
        4. 鼠标和键盘事件的响应放在init_event中初始化

        Parameters
        ----------
        QMainWindow :

        """
        super(MainWindow, self).__init__(*args, **kargs)
        # 初始化布局
        self.setupUi(self)

        # -----<start:类成员------
        # 界面相关的类成员(都在Ui_MainWindow中,这里省略不写)

        # 文件相关的类成员
        self.dir = ""
        self.select_file = ""
        self.file_path = ""
        self.ext = ["*.wav", "*.mp3"]
        self.json_file = json_file

        # matplotlib相关的类成员
        self.mpl_canvas1 = MplCanvas(self.mpl_widget1)
        self.mpl_canvas2 = MplCanvas(self.mpl_widget2)
        self.mpl_canvas3 = MplCanvas(self.mpl_widget3)
        self.mpl_canvas4 = MplCanvas(self.mpl_widget4)
        self.current_canvas = self.mpl_canvas1

        self.mpl_canvases = [self.mpl_canvas1,
                             self.mpl_canvas2,
                             self.mpl_canvas3,
                             self.mpl_canvas4]
        self.mpl_toolbars = [NavigationToolbar(mpl_canvas.figure.canvas,
                                               self, coordinates=False)
                             for mpl_canvas in self.mpl_canvases]
        # self.figures = [mpl_canvas.figure
        #                 for mpl_canvas in self.mpl_canvases]

        # popup窗口,四个画布都可以弹出。增加第五个,用于绘制整段PCG
        self.mpl_window1 = MatplotlibWindow(1)
        self.mpl_window2 = MatplotlibWindow(2)
        self.mpl_window3 = MatplotlibWindow(3)
        self.mpl_window4 = MatplotlibWindow(4)
        self.mpl_window5 = MatplotlibWindow(5)  # 用于绘制整段PCG的时域和频域

        # 保存四个画布的内容
        self.content1 = []
        self.content2 = []
        self.content3 = []
        self.content4 = []
        self.content5 = []  # 这个是单独用来画预处理前后整段PCG的时域和频域

        # 菜单类成员
        self.context = QMenu(self)
        self.popup = QAction('弹出窗口', self)
        self.popup_all = QAction('弹出所有', self)
        self.popup_pcg = QAction('弹出整段', self)
        self.context.addAction(self.popup)
        self.context.addAction(self.popup_all)
        self.context.addAction(self.popup_pcg)

        # 子线程相关的类成员: worker类成员及对应的线程
        # 1.计算整段PCG的心率
        # 2.计算所选时刻的心率
        # 3.播放所选时刻对应窗口的心音(原始信号)
        # 4.获取整段PCG的时域和频域
        self.worker_calc_overall = worker.WorkerCalc(self.json_file)  # 计算整段PCG的心率
        self.worker_calc_pick = worker.WorkerCalc(self.json_file)  # 计算一点的心率
        self.worker_play = worker.WorkerCalc(self.json_file)  # 播放心音
        self.worker_whole_pcg = worker.WorkerCalc(self.json_file)  # 获取整段PCG的时域和频域
        self.thread_calc_overall = QThread()  #
        self.thread_calc_pick = QThread()
        self.thread_play = QThread()
        self.thread_whole_pcg = QThread()

        # ----->end:类成员------

        self.init_size()
        self.init_mpl()
        self.init_workspace()
        self.init_threads()
        self.init_events()

        self.show()
    # -------------<start:初始化使用的函数---------------
    # -----<start:初始化UI界面-----

    def init_size(self):
        """初始化widget的各种尺寸"""
        self.tree_view.resize(271, 694)
        self.tree_view.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mpl_widget1.resize(400, 300)
        self.mpl_widget2.resize(400, 300)
        self.mpl_widget3.resize(400, 300)
        self.mpl_widget4.resize(400, 300)

    def init_mpl(self):
        """
        1.将matplotlib的画布添加到界面中的占位Widget中
        2.初始化matplotlib工具栏
        """
        # 添加matplotlib画布
        vbl = QVBoxLayout()
        vbl.addWidget(self.mpl_canvas1)
        self.mpl_widget1.setLayout(vbl)

        vb2 = QVBoxLayout()
        vb2.addWidget(self.mpl_canvas2)
        self.mpl_widget2.setLayout(vb2)

        vb3 = QVBoxLayout()
        vb3.addWidget(self.mpl_canvas3)
        self.mpl_widget3.setLayout(vb3)

        vb4 = QVBoxLayout()
        vb4.addWidget(self.mpl_canvas4)
        self.mpl_widget4.setLayout(vb4)

        # 初始化当前的matplotlib的工具栏和画布
        # self.switch_mpl(self.mpl_canvas1)
        for toolbar in self.mpl_toolbars:
            self.addToolBar(toolbar)
            toolbar.setVisible(False)
        self.mpl_toolbars[0].setVisible(True)

    def init_workspace(self):
        """
        初始化默认工作空间,并显示其中的文件
        """
        with open(self.json_file, encoding='UTF-8') as f_obj:
            params = json.load(f_obj)

        self.dir = params["dir"]
        self.workspace(self.dir)
    # ----->end:初始化UI界面-----

    # -----<start:connect signals to slots-----
    def init_threads(self):
        """
        初始化线程, 目前有两个线程
        1.计算整段PCG的所有心率
        2.计算所选时刻的心率
        线程的使用有几种方法,这里使用的是moveToThread的方法,其步骤如下:
        1.创建worker对象和QThread对象(前面在类成员定义那里已经做了)
        2.将worker对象moveToThread添加都QThread对象中,也就是说后面该线程启动后,执行的worker对象里的任务
        3.设置线程启动后具体执行的worker对象中的哪个函数(线程启动会释放started信号)
        4.线程在执行worker对象中的任务时可以根据需要释放某些信号,与主程序进行通信
        """

        # 将所有的worker移动到线程中
        self.worker_calc_overall.moveToThread(self.thread_calc_overall)
        self.worker_calc_pick.moveToThread(self.thread_calc_pick)
        self.worker_play.moveToThread(self.thread_play)
        self.worker_whole_pcg.moveToThread(self.thread_whole_pcg)

        # 设置线程启动时执行的程序
        self.thread_calc_overall.started.connect(
            self.worker_calc_overall.get_heart_rate_overall)
        self.thread_calc_pick.started.connect(
            self.worker_calc_pick.get_heart_rate_pick)
        self.thread_play.started.connect(self.worker_play.play_audio)
        self.thread_whole_pcg.started.connect(
            self.worker_whole_pcg.get_whole_pcg)

        # 设置主应用接收到result信号时的处理函数
        self.worker_calc_overall.signal.result.connect(self.on_result_all)
        self.worker_calc_pick.signal.result.connect(self.on_result_pick)
        self.worker_play.signal.result.connect(self.on_result_play)
        self.worker_whole_pcg.signal.result.connect(self.on_result_pcg)

        # 设置主应用接收到finished信号时退出线程
        self.worker_calc_overall.signal.finished.connect(
            self.thread_calc_overall.quit)
        self.worker_calc_pick.signal.finished.connect(
            self.thread_calc_pick.quit)
        self.worker_play.signal.finished.connect(self.thread_play.quit)
        self.worker_whole_pcg.signal.finished.connect(
            self.thread_whole_pcg.quit)

    # ----->end:connect signals to slots-----

    # -----<start:connect events-----

    def init_events(self):
        """
        设置与鼠标相关的event的handler
        1.点击打开文件
        2.双击文件列表中的文件名
        3.在matplotlib画布上鼠标移动
        4.第一个和第三个画布的鼠标选点
        """
        # 设置菜单栏和工作空间中的event响应函数
        self.action_Open_Folder.triggered.connect(self.on_open_folder)
        self.action_Play.triggered.connect(self.on_play)
        self.tree_view.doubleClicked.connect(self.on_double_clicked)

        # 设置右键显示菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_context_menu)

        # 设置右键菜单项的响应
        self.popup.triggered.connect(self.on_popup)
        self.popup_all.triggered.connect(self.on_popup_all)
        self.popup_pcg.triggered.connect(self.on_popup_pcg)

        # 设置主面板中四个画布的鼠标响应:
        # 1.进入figure
        # 2.退出figure
        # 3.进入axes
        # 4.退出axes
        # 5.在axes中鼠标移动
        for mpl_canvas in self.mpl_canvases:
            mpl_canvas.mpl_connect('figure_enter_event', self.on_enter_fig)
            mpl_canvas.mpl_connect('figure_leave_event', self.on_leave_fig)
            mpl_canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
            mpl_canvas.mpl_connect('axes_leave_event', self.on_leave_axes)
            mpl_canvas.mpl_connect('motion_notify_event', self.on_motion)
            mpl_canvas.mpl_connect('button_press_event', self.on_click_fig)

        # 第1个和第3个画布的选点响应
        self.mpl_canvas1.mpl_connect('pick_event', self.on_pick)
        self.mpl_canvas3.mpl_connect('pick_event', self.on_pick)

    # ----->end:connect events-----

    # ------------->end:初始化使用的函数----------------

    def workspace(self, dir):
        """
        处理tree_view显示的工作空间样式
        Parameters
        ----------
        dir : string
            要显示的工作空间的根路径

        """

        if dir:
            # self.tree_view.clear()
            model = QFileSystemModel()
            model.setRootPath(QDir.rootPath())

            model.setNameFilters(self.ext)
            model.setNameFilterDisables(False)

            model.setIconProvider(IconProvider())

            self.tree_view.setModel(model)
            self.tree_view.setRootIndex(model.index(self.dir))

            # 除了文件名那一栏,其他栏全部隐藏
            self.tree_view.setColumnHidden(1, True)
            self.tree_view.setColumnHidden(2, True)
            self.tree_view.setColumnHidden(3, True)

    # -------------<start:events对应的handlers-------------

    def on_open_folder(self):
        """
        event handler:
        打开目录菜单的handler
        """

        # 弹出浏览对话框
        self.dir = QFileDialog.getExistingDirectory(
            self, "Select a folder:", "", QFileDialog.ShowDirsOnly)

        # 筛选出.wav和mp3格式
        if self.dir:
            self.workspace(self.dir)

    def on_play(self):
        """
        event handler:

        点击工具栏上的播放按键的handler
        开启播放线程

        """

        self.worker_play.file_path = self.worker_calc_pick.file_path
        self.worker_play.pick_time = self.worker_calc_pick.pick_time
        self.thread_play.start()

    def on_double_clicked(self, index):
        """
        event handler:
        双击工作空间中文件的handler

        Parameters
        ----------
        index : 双击的item在tree_widget中的索引

        """

        # 取得双击的文件路径
        file_info = QFileInfo(self.tree_view.model().fileInfo(index))
        # 告诉线程文件路径,并启动线程
        if file_info.isFile():
            self.file_path = file_info.absoluteFilePath()
            self.worker_calc_overall.file_path = self.file_path
            self.thread_calc_overall.start()

    def on_context_menu(self, pos):
        """
        event handler:
        右键的handler(显示右键菜单)

        ----------
        pos : 右键的鼠标位置

        """

        # customContextMenuRequested signal会传递位置参数给slots
        hovered_widget = QApplication.widgetAt(self.mapToGlobal(pos))

        if hovered_widget in [self.mpl_canvas2, self.mpl_canvas4]:
            self.popup_pcg.setVisible(False)
            self.context.exec_(self.mapToGlobal(pos))
        elif hovered_widget in [self.mpl_canvas1, self.mpl_canvas3]:
            self.popup_pcg.setVisible(True)
            self.context.exec_(self.mapToGlobal(pos))

    # -----<start:响应鼠标进入figure-----
    def on_enter_fig(self, event):
        """
        event handler:
        鼠标进入figure的handler
        原本设想鼠标进入哪个canvas就将工具栏切换成谁的，但是在实践的时候发现，
        如果使用垂直分布，鼠标会误触切换成其他的画布。所以改成鼠标单击确认显示谁的工具栏
        """
        pass
        # which_canvas = event.canvas
        # self.switch_mpl(which_canvas)

    # ----->end:响应鼠标进入figure-----

    def on_leave_fig(self, event):
        """
        event handler:
        鼠标离开figure的handler
        """

        self.statusBar().showMessage("")

    def on_enter_axes(self, event):
        """
        event handler:
        鼠标进入axes的handler
        """

        pass

    def on_leave_axes(self, event):
        """
        event handler:
        鼠标离开axes的handler
        """

        pass

    # -----<start:响应鼠标在画布中移动-----
    def on_motion(self, event):
        """
        event handler:
        鼠标在figure中移动的handler
        """
        inaxes = event.inaxes
        xdata = event.xdata
        ydata = event.ydata

        if inaxes is None:
            self.statusBar().showMessage("")
        else:
            self.statusBar().showMessage('x={:6.3f}, y={:6.3f}'.format(
                xdata, ydata))

    def switch_mpl(self, which_canvas):
        """
        辅助函数,切换到不同figure时,在界面上作出反应
        原本设想鼠标进入哪个canvas就将工具栏切换成谁的，但是在实践的时候发现，
        如果使用垂直分布，鼠标会误触切换成其他的画布。所以改成鼠标单击确认显示谁的工具栏

        Parameters
        ----------
        which_canvas : FigureCanvas的派生类
            哪个画布

        """
        for toolbar in self.mpl_toolbars:
            toolbar.setVisible(False)
        self.mpl_toolbars[self.mpl_canvases.index(
            which_canvas)].setVisible(True)

        shadow = QGraphicsDropShadowEffect()
        for mpl_canvas in self.mpl_canvases:
            shadow.setBlurRadius(0)
            mpl_canvas.setGraphicsEffect(shadow)

        shadow.setBlurRadius(1)
        which_canvas.setGraphicsEffect(shadow)
        self.current_canvas = which_canvas

    def on_click_fig(self, event):
        """
        event handler:
        鼠标点击figure的handler
        """
        which_canvas = event.canvas
        self.switch_mpl(which_canvas)
    # ----->end:响应鼠标在画布中移动-----

    def on_pick(self, event):
        """
        event handler:
        鼠标在canvas上选点的handler
        """
        line = event.artist
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        ind = event.ind

        pick_time = round(xdata[ind][0])

        # 向线程传递文件路径和所选时刻,启动线程
        self.worker_calc_pick.file_path = self.file_path
        self.worker_calc_pick.pick_time = pick_time
        self.thread_calc_pick.start()

    def on_popup(self):
        """
        event handler:
        右键菜单弹出当前所在的画布的响应函数。
        需要判断当前所在的画布是哪一个,然后才能准确弹出。
        """
        # 画布1
        if self.current_canvas == self.mpl_canvas1:
            plot_to_canvas(
                self.content1, self.mpl_window1.canvas, show_txt=True)
            self.mpl_window1.show()
        # 画布2
        elif self.current_canvas == self.mpl_canvas2:
            plot_to_canvas_pick(
                self.content2, self.mpl_window2.canvas, show_txt=True)
            self.mpl_window2.show()
        # 画布3
        elif self.current_canvas == self.mpl_canvas3:
            plot_to_canvas(
                self.content3, self.mpl_window3.canvas, show_txt=True)
            self.mpl_window3.show()
        # 画布4
        elif self.current_canvas == self.mpl_canvas4:
            plot_to_canvas_pick(
                self.content4, self.mpl_window4.canvas, show_txt=True)
            self.mpl_window4.show()
        else:
            print("No such canvas!")

    def on_popup_all(self):
        """
        event handler:
        右键菜单弹出所有画布内容的响应函数
        """

        plot_to_canvas(self.content1, self.mpl_window1.canvas)
        self.mpl_window1.show()

        plot_to_canvas_pick(self.content2, self.mpl_window2.canvas)
        self.mpl_window2.show()

        plot_to_canvas(self.content3, self.mpl_window3.canvas)
        self.mpl_window3.show()

        plot_to_canvas_pick(self.content4, self.mpl_window4.canvas)
        self.mpl_window4.show()

    def on_popup_pcg(self):
        """
        event handler:
        右键菜单弹出整段PCG时域和频域的响应函数。
        由于之前并没有计算整段PCG的时域和频域信息,
        因此这里需要开另一个线程来计算时域和频域,
        然后线程返回主应用程序,才能画图

        """

        self.worker_whole_pcg.file_path = self.worker_calc_overall.file_path
        self.thread_whole_pcg.start()
    # ------------->end:events对应的handlers---------------

    # -------------<start:signals对应的slots---------------
    def on_result_all(self, data):
        """
        线程信号的slot函数:
        计算整段心率的线程返回时的slot函数

        Parameters
        ----------
        data : tuple
            使用两种不同的方法计算出来的心率随时间变化的序列

        """
        # 两种方法计算出来的结果需要绘制到不同的画布中,拆解
        self.content1, self.content3 = data

        # 清空画布,因为第2,4画布的内容归属于第1,3画布,因此也要跟着清空
        self.mpl_canvas1.figure.clear()
        self.mpl_canvas3.figure.clear()

        self.mpl_canvas2.figure.clear()
        self.mpl_canvas4.figure.clear()
        self.mpl_canvas2.draw()
        self.mpl_canvas4.draw()

        # 绘制
        plot_to_canvas(self.content1, self.mpl_canvas1)
        plot_to_canvas(self.content3, self.mpl_canvas3)

    def on_result_pick(self, data):
        """
        线程信号的slot函数:
        获取所选时刻计算过程的信息的slot函数

        Parameters
        ----------
        data : tuple
            data包含两种不同的计算方法获取的包络,自相关,峰值等信息

        """
        self.content2, self.content4 = data
        # seg, env, corrs, peaks, peak, scope = self.content2
        # print("here is on_result_pick:")
        # print(peaks)

        plot_to_canvas_pick(self.content2, self.mpl_canvas2)
        plot_to_canvas_pick(self.content4, self.mpl_canvas4)

    # ------------->end:signals对应的slots---------------

    # -------------<start:辅助函数---------------
    def on_result_play(self, data):
        """
        线程信号的slot函数:
        工具栏上播放按钮点击之后保持按住的状态,
        直到音频播放结束,这里解除按住的状态

        Parameters
        ----------
        data : 这里没什么作用

        """
        # print(data[0])  # just joking
        self.action_Play.setChecked(False)

    def on_result_pcg(self, data):
        """
        线程信号的slot函数:
        线程计算完PCG的时域和频域信息之后,绘图

        Parameters
        ----------
        data : tuple
            包含整段PCG的时域和频域信息
        """

        self.content5 = data
        plot_to_canvas_pcg(self.content5, self.mpl_window5.canvas, True)
        self.mpl_window5.show()
    # ------------->end:辅助函数---------------


# -------------<start:外部辅助函数---------------
def plot_to_canvas(plot_data, canvas, show_txt=False):
    """
    绘制到指定画布中

    绘制的整段心率变化时使用的

    Parameters
    ----------
    plot_data : tuple (time, data)
        时间信息和幅值信息
    canvas : FigureCanvas
        指定绘制到哪块画布中
    show_txt : bool, optional
        是否添加文字,比如标题,坐标轴内容等

    """

    time, heart_rate = plot_data
    figure = canvas.figure
    figure.clear()

    axis = figure.add_subplot(111)
    axis.plot(time, heart_rate, picker=5)
    if show_txt:
        axis.set_title("heart_rate")
        axis.set_xlabel("time (seconds)")
        axis.set_ylabel("heart rate (times/min)")
    canvas.draw()


def plot_to_canvas_pick(plot_data, canvas, show_txt=False):
    """
    绘制到指定画布中

    绘制的所选时对应窗口的所有信息时使用的

    Parameters
    ----------
    plot_data : tuple (seg, env, corrs, peaks, peak, scope)
        窗口信号,其包络,包络自相关,峰值组,最后的峰值,搜索范围等
    canvas : FigureCanvas
        指定绘制到哪块画布中
    show_txt : bool, optional
        是否添加文字,比如标题,坐标轴内容等
    """
    seg_with_t = plot_data[0]
    env_with_t = plot_data[1]
    corrs_with_t = plot_data[2]
    peaks_with_t = plot_data[3]
    peak_with_t = plot_data[4]
    scope_with_t = plot_data[5]

    figure = canvas.figure
    figure.clear()

    axarr = figure.subplots(3, 1, sharex=True)
    figure.subplots_adjust(hspace=0.4)
    axarr[0].plot(seg_with_t[0], seg_with_t[1])
    axarr[1].plot(env_with_t[0], env_with_t[1])
    axarr[2].plot(corrs_with_t[0], corrs_with_t[1])
    axarr[2].plot(peaks_with_t[0], peaks_with_t[1], 'rx')
    axarr[2].plot(peak_with_t[0], peak_with_t[1], 'bx')
    axarr[2].vlines(scope_with_t,
                    0,
                    1,
                    transform=axarr[2].get_xaxis_transform(),
                    colors='r',
                    linestyles="dashed")

    if show_txt:
        axarr[0].set_title("(a) PCG")
        axarr[1].set_title("(b) Envelope")
        axarr[2].set_title("(c) Autocorrs")
        axarr[2].set_xlabel("time (Seconds)")

    canvas.draw()


def plot_to_canvas_pcg(plot_data, canvas, show_txt=False):
    """
    绘制到指定画布中

    绘制的整段的时域和频域信息时使用的

    Parameters
    ----------
    plot_data : tuple
        时域信息和频域信息
    canvas : FigureCanvas
        指定绘制到哪块画布中
    show_txt : bool, optional
        是否添加文字,比如标题,坐标轴内容等

    """
    pcg_t = plot_data[0]
    pcg_f = plot_data[1]
    pcg_proc_t = plot_data[2]
    pcg_proc_f = plot_data[3]

    figure = canvas.figure
    figure.clear()

    (ax1, ax2), (ax3, ax4) = figure.subplots(2, 2)
    figure.subplots_adjust(hspace=0.4)
    ax1.plot(pcg_t[0], pcg_t[1])
    ax2.plot(pcg_f[0], pcg_f[1])
    ax3.plot(pcg_proc_t[0], pcg_proc_t[1])
    ax4.plot(pcg_proc_f[0], pcg_proc_f[1])

    if show_txt:
        figure.suptitle('filtered before VS after')
        ax1.set_title('time-domain before')
        ax2.set_title('freq-domain before')
        ax3.set_title('time-domain after')
        ax4.set_title('freq-domain after')

    canvas.draw()

# ------------->end:外部辅助函数---------------


# -----------------<start:辅助类---------------
# matplotlib的canvas
class MplCanvas(FigureCanvas):
    """
    自定义的matplotlib的FigureCanvas画布
    """

    def __init__(self, parent=None):
        self.figure = Figure()
        # self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MatplotlibWidget(QWidget):
    """
    原本是想将matplotlib的FigureCanvas做成一个widget的,方便添加到。
    后来采用将canvas添加到Ui中占位的QWidget中的方式,而不是这里将Canvas封装成QWidget的方式
    """

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        # self.figure = self.canvas.fig
        self.vbl = QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


class MatplotlibWindow(QMainWindow):
    """
    作为弹出的窗口。

    使用matplotlib绘制图形

    Parameters
    ----------
    QMainWindow :父类

    """

    def __init__(self, num, *args, **kargs):
        super(MatplotlibWindow, self).__init__(*args, **kargs)

        # 设置窗口title
        self.setWindowTitle("figure" + str(num))

        # 设置窗口图标
        icon = QIcon()
        icon.addPixmap(QPixmap(":/icons/signal.png"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        # 设置窗口widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.canvas = MplCanvas(self)
        self.figure = self.canvas.figure

        # 设置toolbar和statusbar
        nav_toolbar = NavigationToolbar(
            self.canvas.figure.canvas, self, coordinates=False)
        self.addToolBar(nav_toolbar)
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        # 设置布局
        self.vbl = QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.main_widget.setLayout(self.vbl)

        # 设置画布中鼠标移动的slot函数:在状态栏中显示坐标
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_motion(self, event):
        """鼠标在figure中移动的handler"""
        inaxes = event.inaxes
        xdata = event.xdata
        ydata = event.ydata

        if inaxes is None:
            self.statusBar().showMessage("")
        else:
            self.statusBar().showMessage('x={:6.3f}, y={:6.3f}'.format(
                xdata, ydata))


# 设置图标
class IconProvider(QFileIconProvider):
    """重载icon,用于设置图标"""

    def icon(self, fileInfo):
        if fileInfo.isDir():
            return QIcon('resources/icons/folder.png')

        elif fileInfo.isFile():
            if fileInfo.suffix() in ["wav", "mp3"]:
                return QIcon('resources/icons/mp3.png')

        return QFileIconProvider.icon(self, fileInfo)

# ----------------->end:辅助类---------------


if __name__ == '__main__':
    import warnings
    app = QApplication([])
    w = MainWindow("params.json")
    warnings.filterwarnings("ignore")
    app.exec_()
