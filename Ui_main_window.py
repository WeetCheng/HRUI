# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\我的电脑收纳\001 - Inbox - 中转站，先到碗里来\HRAnalysis\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1099, 791)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/signal.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.main_widget = QtWidgets.QWidget(MainWindow)
        self.main_widget.setObjectName("main_widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.main_widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.h_splitter = QtWidgets.QSplitter(self.main_widget)
        self.h_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.h_splitter.setObjectName("h_splitter")
        self.tree_view = QtWidgets.QTreeView(self.h_splitter)
        self.tree_view.setMinimumSize(QtCore.QSize(200, 500))
        self.tree_view.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.tree_view.setObjectName("tree_view")
        self.v_splitter = QtWidgets.QSplitter(self.h_splitter)
        self.v_splitter.setOrientation(QtCore.Qt.Vertical)
        self.v_splitter.setObjectName("v_splitter")
        self.top_splitter = QtWidgets.QSplitter(self.v_splitter)
        self.top_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.top_splitter.setObjectName("top_splitter")
        self.mpl_widget1 = QtWidgets.QWidget(self.top_splitter)
        self.mpl_widget1.setMinimumSize(QtCore.QSize(400, 300))
        self.mpl_widget1.setBaseSize(QtCore.QSize(0, 0))
        self.mpl_widget1.setObjectName("mpl_widget1")
        self.mpl_widget2 = QtWidgets.QWidget(self.top_splitter)
        self.mpl_widget2.setMinimumSize(QtCore.QSize(400, 300))
        self.mpl_widget2.setBaseSize(QtCore.QSize(0, 0))
        self.mpl_widget2.setObjectName("mpl_widget2")
        self.bottom_splitter = QtWidgets.QSplitter(self.v_splitter)
        self.bottom_splitter.setOrientation(QtCore.Qt.Horizontal)
        self.bottom_splitter.setObjectName("bottom_splitter")
        self.mpl_widget3 = QtWidgets.QWidget(self.bottom_splitter)
        self.mpl_widget3.setMinimumSize(QtCore.QSize(400, 300))
        self.mpl_widget3.setBaseSize(QtCore.QSize(0, 0))
        self.mpl_widget3.setObjectName("mpl_widget3")
        self.mpl_widget4 = QtWidgets.QWidget(self.bottom_splitter)
        self.mpl_widget4.setMinimumSize(QtCore.QSize(400, 300))
        self.mpl_widget4.setBaseSize(QtCore.QSize(0, 0))
        self.mpl_widget4.setObjectName("mpl_widget4")
        self.horizontalLayout.addWidget(self.h_splitter)
        MainWindow.setCentralWidget(self.main_widget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1099, 23))
        self.menubar.setObjectName("menubar")
        self.menu_File = QtWidgets.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        self.menu_Console = QtWidgets.QMenu(self.menubar)
        self.menu_Console.setObjectName("menu_Console")
        self.menuPlay = QtWidgets.QMenu(self.menubar)
        self.menuPlay.setObjectName("menuPlay")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_Open_Folder = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Open_Folder.setIcon(icon1)
        self.action_Open_Folder.setShortcutContext(QtCore.Qt.WindowShortcut)
        self.action_Open_Folder.setObjectName("action_Open_Folder")
        self.actionOpen_Console = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/console.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen_Console.setIcon(icon2)
        self.actionOpen_Console.setObjectName("actionOpen_Console")
        self.action_Play = QtWidgets.QAction(MainWindow)
        self.action_Play.setCheckable(True)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Play.setIcon(icon3)
        self.action_Play.setObjectName("action_Play")
        self.menu_File.addAction(self.action_Open_Folder)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menuPlay.menuAction())
        self.menubar.addAction(self.menu_Console.menuAction())
        self.toolBar.addAction(self.action_Open_Folder)
        self.toolBar.addAction(self.action_Play)
        self.toolBar.addAction(self.actionOpen_Console)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu_File.setTitle(_translate("MainWindow", "&File"))
        self.menu_Console.setTitle(_translate("MainWindow", "Console"))
        self.menuPlay.setTitle(_translate("MainWindow", "Play"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_Open_Folder.setText(_translate("MainWindow", "&Open Folder..."))
        self.action_Open_Folder.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionOpen_Console.setText(_translate("MainWindow", "Open &Console"))
        self.action_Play.setText(_translate("MainWindow", "&Play"))

import resources_rc
