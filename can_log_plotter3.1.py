import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QPushButton, QFileDialog, QListWidget, 
                            QComboBox, QCheckBox, QDoubleSpinBox, QTabWidget, QMessageBox, 
                            QProgressBar, QSplitter, QLineEdit, QStatusBar, QToolButton, 
                            QTableWidget, QTableWidgetItem, QToolBar, QAction, QGroupBox, 
                            QMenu, QDockWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QSize
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import can
from can.io import ASCReader, BLFReader, CSVReader
import cantools
from datetime import datetime

# Modern stylesheet
STYLESHEET = """
QMainWindow {
    background-color: #2E2E2E;
}
QWidget {
    color: #FFFFFF;
    font-family: Segoe UI, Arial;
}
QPushButton {
    background-color: #424242;
    border-radius: 5px;
    padding: 8px;
    color: #FFFFFF;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #616161, stop:1 #424242);
}
QPushButton:pressed {
    background-color: #313131;
}
QPushButton[role="load"] {
    background-color: #4CAF50;
}
QPushButton[role="clear"] {
    background-color: #F44336;
}
QPushButton[role="plot"] {
    background-color: #FF5722;
}
QComboBox, QLineEdit, QDoubleSpinBox {
    background-color: #424242;
    border-radius: 4px;
    padding: 5px;
    color: #FFFFFF;
}
QListWidget, QTableWidget {
    background-color: #353535;
    border: 1px solid #555555;
    alternate-background-color: #3A3A3A;
}
QGroupBox {
    border: 1px solid #555555;
    border-radius: 5px;
    margin-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QToolBar {
    background-color: #353535;
    border: none;
}
QStatusBar {
    background-color: #353535;
    color: #BBBBBB;
}
QTabWidget::pane {
    border: 1px solid #555555;
    background-color: #2E2E2E;
}
QTabBar::tab {
    background: #424242;
    color: #FFFFFF;
    padding: 8px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #FF5722;
}
"""

class LogLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame, dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, db):
        super().__init__()
        self.file_path = file_path
        self.db = db
    
    def run(self):
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == '.asc':
                reader = can.ASCReader(self.file_path)
            elif ext == '.blf':
                reader = can.BLFReader(self.file_path)
            elif ext == '.csv':
                reader = can.CSVReader(self.file_path)
            else:
                raise ValueError("Unsupported file format")
            
            messages = []
            signals_data = {}
            total_messages = 0
            processed_messages = 0
            
            try:
                with open(self.file_path, 'rb') as f:
                    for _ in f:
                        total_messages += 1
            except:
                total_messages = 0
            
            for msg in reader:
                msg_data = {
                    'timestamp': msg.timestamp,
                    'arbitration_id': msg.arbitration_id,
                    'data': msg.data,
                    'dlc': msg.dlc,
                    'channel': msg.channel if hasattr(msg, 'channel') else 0,
                    'is_fd': msg.is_fd if hasattr(msg, 'is_fd') else False,
                    'bitrate_switch': msg.bitrate_switch if hasattr(msg, 'bitrate_switch') else False
                }
                if self.db:
                    try:
                        message = self.db.get_message_by_frame_id(msg.arbitration_id)
                        data_str = ' '.join(f'{b:02X}' for b in msg.data)
                        print(f"Processing ID 0x{msg.arbitration_id:X}, Timestamp: {msg.timestamp:.6f}, DLC: {msg.dlc}, Data: {data_str}")
                        
                        try:
                            decoded = message.decode(msg.data, decode_choices=False, allow_truncated=True, scaling=True, allow_excess=True)
                            print(f"Decoded {message.name} (ID 0x{msg.arbitration_id:X}): {decoded}")
                            
                            for signal_name, value in decoded.items():
                                key = f"{message.name}.{signal_name}"
                                if key not in signals_data:
                                    signals_data[key] = []
                                try:
                                    value = float(value) if value is not None else 0.0
                                except (TypeError, ValueError):
                                    print(f"Non-numeric value for {key}: {value}, skipping")
                                    continue
                                signals_data[key].append((msg.timestamp, value))
                                print(f"Stored signal: {key}, value: {value}, timestamp: {msg.timestamp}")
                            
                            if message.is_multiplexed():
                                print(f"Multiplexed message: {message.name}")
                                for signal in message.signals:
                                    if signal.is_multiplexer or signal.multiplexer_ids:
                                        print(f"Signal {signal.name}: Multiplexer={signal.is_multiplexer}, IDs={signal.multiplexer_ids}")
                        except Exception as decode_error:
                            print(f"Decoding failed for {message.name} (ID 0x{msg.arbitration_id:X}): {str(decode_error)}")
                    except KeyError:
                        print(f"No DBC message found for ID 0x{msg.arbitration_id:X}")
                    except Exception as e:
                        print(f"Unexpected error for ID 0x{msg.arbitration_id:X}: {str(e)}")
                else:
                    for i in range(msg.dlc):
                        key = f"ID_0x{msg.arbitration_id:X}.Byte{i}"
                        if key not in signals_data:
                            signals_data[key] = []
                        signals_data[key].append((msg.timestamp, msg.data[i]))
                
                messages.append(msg_data)
                processed_messages += 1
                if total_messages > 0:
                    self.progress.emit(min(100, int((processed_messages / total_messages) * 100)))
            
            df = pd.DataFrame(messages)
            signals_data = {key: pd.DataFrame(data, columns=['timestamp', 'value']) 
                           for key, data in signals_data.items()}
            print(f"Processed {processed_messages} messages, decoded {len(signals_data)} signals")
            print(f"Signal keys: {list(signals_data.keys())}")
            for key, data in signals_data.items():
                print(f"Signal {key}: {len(data)} data points")
            if not signals_data:
                print("Warning: No signals decoded from log file")
            self.finished.emit(df, signals_data)
            self.deleteLater()
        except Exception as e:
            self.error.emit(str(e))

class PlotterThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, signals_data, selected_signals, time_range, normalize, channel, plot_styles):
        super().__init__()
        self.signals_data = signals_data
        self.selected_signals = selected_signals
        self.time_range = time_range
        self.normalize = normalize
        self.channel = channel
        self.plot_styles = plot_styles
    
    def run(self):
        try:
            plot_data = {}
            
            if self.time_range is None:
                all_timestamps = []
                for signal in self.selected_signals:
                    clean_signal = signal.replace('[PDU] ', '')
                    if clean_signal in self.signals_data:
                        all_timestamps.extend(self.signals_data[clean_signal]['timestamp'].values)
                    else:
                        print(f"Signal {clean_signal} not found in signals_data")
                if not all_timestamps:
                    raise ValueError("No data available for selected signals")
                start_time = min(all_timestamps)
                end_time = max(all_timestamps)
                print(f"Full log time range: {start_time} to {end_time}")
            else:
                start_time, end_time = self.time_range
                print(f"Custom time range: {start_time} to {end_time}")
            
            for signal in self.selected_signals:
                clean_signal = signal.replace('[PDU] ', '')
                if clean_signal not in self.signals_data:
                    print(f"Signal {clean_signal} not in signals_data")
                    continue
                data = self.signals_data[clean_signal]
                print(f"Raw data for {clean_signal}: {len(data)} rows")
                
                filtered_data = data[(data['timestamp'] >= start_time) & 
                                   (data['timestamp'] <= end_time)]
                print(f"Filtered data for {clean_signal}: {len(filtered_data)} rows")
                
                if filtered_data.empty:
                    print(f"No data for {clean_signal} in time range {start_time} to {end_time}")
                    continue
                
                values = filtered_data['value'].values
                if self.normalize and len(values) > 0:
                    min_val = np.min(values)
                    max_val = np.max(values)
                    if max_val > min_val:
                        values = (values - min_val) / (max_val - min_val)
                
                plot_data[clean_signal] = {
                    'timestamps': filtered_data['timestamp'].values,
                    'values': values,
                    'stats': {
                        'min': np.min(values) if len(values) > 0 else None,
                        'max': np.max(values) if len(values) > 0 else None,
                        'mean': np.mean(values) if len(values) > 0 else None
                    },
                    'style': self.plot_styles.get(signal, {'color': 'blue', 'linestyle': '-', 'marker': 'o'})
                }
                print(f"Prepared plot data for {clean_signal}: {len(values)} points")
            
            if not plot_data:
                raise ValueError("No valid data to plot for selected signals")
            
            print(f"Plot data prepared for signals: {list(plot_data.keys())}")
            self.finished.emit(plot_data)
            self.deleteLater()
        except Exception as e:
            self.error.emit(str(e))

class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self.toggle_content)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.content_widget)
    
    def toggle_content(self, checked):
        anim = QPropertyAnimation(self.content_widget, b"maximumHeight")
        anim.setDuration(200)
        anim.setStartValue(self.content_widget.height() if checked else 0)
        anim.setEndValue(0 if not checked else 200)
        anim.start()
        self.content_widget.setVisible(checked)

class CANAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAN Analyzer Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.setAcceptDrops(True)
        
        try:
            self.setWindowIcon(QIcon("can_icon.png"))
        except:
            pass
        
        self.dbc_file = None
        self.db = None
        self.log_file = None
        self.log_data = None
        self.signals_data = {}
        self.is_dark_mode = True
        self.status_history = []
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        self.init_ui()
        
    def init_ui(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        
        load_dbc_action = QAction(QIcon("load.png"), "Load DBC", self)
        load_dbc_action.setToolTip("Load a DBC file (Ctrl+D)")
        load_dbc_action.triggered.connect(self.load_dbc_file)
        load_dbc_action.setShortcut("Ctrl+D")
        self.toolbar.addAction(load_dbc_action)
        
        load_log_action = QAction(QIcon("load.png"), "Load Log", self)
        load_log_action.setToolTip("Load CAN/CAN FD log (Ctrl+O)")
        load_log_action.triggered.connect(self.load_log_file)
        load_log_action.setShortcut("Ctrl+O")
        self.toolbar.addAction(load_log_action)
        
        plot_action = QAction(QIcon("plot.png"), "Plot", self)
        plot_action.setToolTip("Plot signals (Ctrl+P)")
        plot_action.triggered.connect(self.plot_signals)
        plot_action.setShortcut("Ctrl+P ''
        
        export_action = QAction(QIcon("export.png"), "Export", self)
        export_action.setToolTip("Export plot/data")
        export_action.triggered.connect(self.export_plot)
        self.toolbar.addAction(export_action)
        
        self.theme_action = QAction(QIcon("theme.png"), "Toggle Theme", self)
        self.theme_action.setToolTip("Switch light/dark mode")
        self.theme_action.triggered.connect(self.toggle_theme)
        self.toolbar.addAction(self.theme_action)
        
        self.dbc_status = QLabel("DBC: None")
        self.dbc_status.setStyleSheet("color: #FF5722; font-size: 12px;")
        self.toolbar.addWidget(self.dbc_status)
        self.toolbar.addSeparator()
        
        self.log_status = QLabel("Log: None")
        self.log_status.setStyleSheet("color: #FF5722; font-size: 12px;")
        self.toolbar.addWidget(self.log_status)
        
        self.sidebar = QDockWidget("Views", self)
        self.sidebar.setFixedWidth(200)
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        
        self.view_list = QListWidget()
        self.view_list.setAlternatingRowColors(True)
        self.view_list.itemClicked.connect(self.switch_view)
        self.sidebar_layout.addWidget(self.view_list)
        self.sidebar.setWidget(self.sidebar_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebar)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        
        self.dbc_group = CollapsibleGroupBox("DBC File")
        self.dbc_content = self.dbc_group.content_widget
        self.dbc_layout = QHBoxLayout()
        
        self.dbc_label = QLabel("No DBC file loaded")
        self.dbc_label.setToolTip("Loaded DBC file or drag-and-drop here")
        self.dbc_layout.addWidget(self.dbc_label)
        
        self.load_dbc_btn = QPushButton("Load")
        self.load_dbc_btn.setProperty("role", "load")
        self.load_dbc_btn.setToolTip("Load DBC file")
        self.load_dbc_btn.clicked.connect(self.load_dbc_file)
        self.load_dbc_btn.setFixedWidth(80)
        self.dbc_layout.addWidget(self.load_dbc_btn)
        
        self.clear_dbc_btn = QPushButton("Clear")
        self.clear_dbc_btn.setProperty("role", "clear")
        self.clear_dbc_btn.setToolTip("Unload DBC file")
        self.clear_dbc_btn.clicked.connect(self.clear_dbc)
        self.clear_dbc_btn.setFixedWidth(80)
        self.dbc_layout.addWidget(self.clear_dbc_btn)
        
        self.dbc_content.layout().addLayout(self.dbc_layout)
        self.control_layout.addWidget(self.dbc_group)
        
        self.log_group = CollapsibleGroupBox("Log File")
        self.log_content = self.log_group.content_widget
        self.log_layout = QHBoxLayout()
        
        self.log_label = QLabel("No log file loaded")
        self.log_label.setToolTip("Loaded log file or drag-and-drop here")
        self.log_layout.addWidget(self.log_label)
        
        self.load_log_btn = QPushButton("Load")
        self.load_log_btn.setProperty("role", "load")
        self.load_log_btn.setToolTip("Load CAN/CAN FD log file")
        self.load_log_btn.clicked.connect(self.load_log_file)
        self.load_log_btn.setFixedWidth(80)
        self.log_layout.addWidget(self.load_log_btn)
        
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setProperty("role", "clear")
        self.clear_log_btn.setToolTip("Unload log file")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setFixedWidth(80)
        self.log_layout.addWidget(self.clear_log_btn)
        
        self.log_content.layout().addLayout(self.log_layout)
        self.control_layout.addWidget(self.log_group)
        
        self.control_tabs = QTabWidget()
        self.control_tabs.setStyleSheet("QTabBar::tab { height: 30px; }")
        
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        
        self.msg_filter = QLineEdit()
        self.msg_filter.setPlaceholderText("Filter messages...")
        self.msg_filter.setToolTip("Filter by ID or name")
        self.msg_filter.textChanged.connect(self.filter_messages)
        self.messages_layout.addWidget(self.msg_filter)
        
        self.message_list = QListWidget()
        self.message_list.setAlternatingRowColors(True)
        self.message_list.setSelectionMode(QListWidget.MultiSelection)
        self.message_list.setToolTip("Select messages to view signals")
        self.message_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.message_list.customContextMenuRequested.connect(self.show_message_context_menu)
        self.messages_layout.addWidget(self.message_list)
        
        self.clear_msg_btn = QPushButton("Clear Messages")
        self.clear_msg_btn.setProperty("role", "clear")
        self.clear_msg_btn.clicked.connect(self.clear_messages)
        self.messages_layout.addWidget(self.clear_msg_btn)
        
        self.control_tabs.addTab(self.messages_widget, "Messages")
        
        self.signals_widget = QWidget()
        self.signals_layout = QVBoxLayout(self.signals_widget)
        
        self.signal_filter = QLineEdit()
        self.signal_filter.setPlaceholderText("Filter signals...")
        self.signal_filter.setToolTip("Filter by signal name")
        self.signal_filter.textChanged.connect(self.filter_signals)
        self.signals_layout.addWidget(self.signal_filter)
        
        self.signal_list = QListWidget()
        self.signal_list.setAlternatingRowColors(True)
        self.signal_list.setSelectionMode(QListWidget.MultiSelection)
        self.signal_list.setToolTip("Select signals to plot")
        self.signal_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.signal_list.customContextMenuRequested.connect(self.show_signal_context_menu)
        self.signals_layout.addWidget(self.signal_list)
        
       subs_layout.addWidget(self.clear_signal_btn)
        
        self.control_tabs.addTab(self.signals_widget, "Signals")
        
        self.control_layout.addWidget(self.control_tabs)
        
        self.options_group = CollapsibleGroupBox("Plot Options")
        self.options_content = self.options_group.content_widget
        self.options_layout = QVBoxLayout()
        
        self.time_range_layout = QHBoxLayout()
        self.time_range_layout.addWidget(QLabel("Time Range:"))
        
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Full log", "Custom"])
        self.time_range_combo.setToolTip("Choose time range for plotting")
        self.time_range_layout.addWidget(self.time_range_combo)
        
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0, 999999)
        self.start_time_spin.setEnabled(False)
        self.start_time_spin.setDecimals(6)
        self.time_range_layout.addWidget(self.start_time_spin)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0, 999999)
        self.end_time_spin.setEnabled(False)
        self.end_time_spin.setDecimals(6)
        self.time_range_layout.addWidget(self.end_time_spin)
        
        self.options_layout.addLayout(self.time_range_layout)
        
        self.style_layout = QHBoxLayout()
        self.style_layout.addWidget(QLabel("Style:"))
        
        self.linestyle_combo = QComboBox()
        self.linestyle_combo.addItems(['-', '--', ':', '-.'])
        self.style_layout.addWidget(self.linestyle_combo)
        
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(['o', 's', '^', 'None'])
        self.style_layout.addWidget(self.marker_combo)
        
        self.options_layout.addLayout(self.style_layout)
        
        self.normalize_check = QCheckBox("Normalize signals")
        self.options_layout.addWidget(self.normalize_check)
        
        self.legend_check = QCheckBox("Show legend")
        self.legend_check.setChecked(True)
        self.options_layout.addWidget(self.legend_check)
        
        self.grid_check = QCheckBox("Show grid")
        self.grid_check.setChecked(True)
        self.options_layout.addWidget(self.grid_check)
        
        self.plot_btn = QPushButton("Plot Signals")
        self.plot_btn.setProperty("role", "plot")
        self.plot_btn.clicked.connect(self.plot_signals)
        self.options_layout.addWidget(self.plot_btn)
        
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.setProperty("role", "plot")
        self.update_plot_btn.clicked.connect(lambda: self.plot_signals(update=True))
        self.options_layout.addWidget(self.update_plot_btn)
        
        self.options_content.layout().addLayout(self.options_layout)
        self.control_layout.addWidget(self.options_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #424242; border-radius: 5px; } QProgressBar::chunk { background-color: #4CAF50; }")
        self.control_layout.addWidget(self.progress_bar)
        
        self.control_layout.addStretch()
        
        self.plot_panel = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_panel)
        
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.plot_panel)
        self.splitter.setSizes([400, 1000])
        
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
        
        self.init_plot_area()
        
        self.message_list.itemSelectionChanged.connect(self.update_signal_list)
        self.time_range_combo.currentTextChanged.connect(self.toggle_time_range)
        
        self.toggle_theme()
        
    def init_plot_area(self):
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabsClosable(True)
        self.plot_tabs.tabCloseRequested.connect(self.close_tab)
        self.plot_layout.addWidget(self.plot_tabs)
        self.add_plot_tab("Plot 1")
        
        self.raw_data_tab = QWidget()
        raw_layout = QVBoxLayout(self.raw_data_tab)
        
        raw_filter_layout = QHBoxLayout()
        self.raw_filter = QLineEdit()
        self.raw_filter.setPlaceholderText("Filter raw data (e.g., ID:0x100, FD:Yes)...")
        self.raw_filter.textChanged.connect(self.filter_raw_data)
        raw_filter_layout.addWidget(self.raw_filter)
        
        export_raw_btn = QPushButton("Export Raw Data")
        export_raw_btn.setProperty("role", "plot")
        export_raw_btn.clicked.connect(self.export_raw_data)
        raw_filter_layout.addWidget(export_raw_btn)
        
        raw_layout.addLayout(raw_filter_layout)
        
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setColumnCount(7)
        self.raw_data_table.setHorizontalHeaderLabels(["Timestamp", "ID", "Data", "DLC", "Channel", "FD", "BRS"])
        self.raw_data_table.setSortingEnabled(True)
        self.raw_data_table.setAlternatingRowColors(True)
        raw_layout.addWidget(self.raw_data_table)
        
        self.plot_tabs.addTab(self.raw_data_tab, "Raw Data")
        self.view_list.addItem("Raw Data")
        
    def add_plot_tab(self, title):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        fig = plt.Figure(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor('#2E2E2E')
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("QToolBar { background-color: #353535; border: none; }")
        
        stats_label = QLabel("Statistics: N/A")
        stats_label.setStyleSheet("color: #BBBBBB; font-size: 12px;")
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        layout.addWidget(stats_label)
        
        self.plot_tabs.addTab(tab, title)
        self.plot_tabs.setCurrentIndex(self.plot_tabs.count() - 1)
        self.view_list.addItem(title)
        
        return fig, canvas, stats_label
    
    def close_tab(self, index):
        if self.plot_tabs.count() > 1:
            tab = self.plot_tabs.widget(index)
            if tab != self.raw_data_tab:
                canvas = tab.findChild(FigureCanvas)
                if canvas:
                    plt.close(canvas.figure)
                self.plot_tabs.removeTab(index)
                self.view_list.takeItem(index)
    
    def switch_view(self, item):
        tab_name = item.text()
        for i in range(self.plot_tabs.count()):
            if self.plot_tabs.tabText(i) == tab_name:
                self.plot_tabs.setCurrentIndex(i)
                break
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setStyleSheet("QMainWindow { border: 2px solid #4CAF50; }")
            event.accept()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet(STYLESHEET)
    
    def dropEvent(self, event):
        self.setStyleSheet(STYLESHEET)
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.dbc':
                self.load_dbc_file(file_path)
            elif ext in ('.asc', '.blf', '.csv'):
                self.load_log_file(file_path)
    
    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        palette = QPalette()
        if self.is_dark_mode:
            palette.setColor(QPalette.Window, QColor(46, 46, 46))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(53, 53, 53))
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(66, 66, 66))
            palette.setColor(QPalette.ButtonText, Qt.white)
            self.setStyleSheet(STYLESHEET)
            plt.style.use('seaborn-darkgrid')
        else:
            palette = QApplication.style().standardPalette()
            self.setStyleSheet("")
            plt.style.use('default')
        QApplication.setPalette(palette)
        self.update_status(f"Switched to {'dark' if self.is_dark_mode else 'light'} mode")
    
    def update_status(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        self.status_label.setText(full_message)
        self.status_history.append(full_message)
        if len(self.status_history) > 50:
            self.status_history.pop(0)
        self.status_label.setToolTip("\n".join(self.status_history[-5:]))
    
    def show_message_context_menu(self, position):
        menu = QMenu()
        copy_action = menu.addAction("Copy ID")
        select_all_action = menu.addAction("Select All")
        clear_action = menu.addAction("Clear Messages")
        
        action = menu.exec_(self.message_list.mapToGlobal(position))
        if action == copy_action:
            selected = self.message_list.selectedItems()
            if selected:
                QApplication.clipboard().setText(selected[0].text().split(' ')[0])
        elif action == select_all_action:
            for i in range(self.message_list.count()):
                self.message_list.item(i).setSelected(True)
        elif action == clear_action:
            self.clear_messages()
    
    def show_signal_context_menu(self, position):
        menu = QMenu()
        copy_action = menu.addAction("Copy Signal")
        select_all_action = menu.addAction("Select All")
        clear_action = menu.addAction("Clear Signals")
        
        action = menu.exec_(self.signal_list.mapToGlobal(position))
        if action == copy_action:
            selected = self.signal_list.selectedItems()
            if selected:
                QApplication.clipboard().setText(selected[0].text())
        elif action == select_all_action:
            for i in range(self.signal_list.count()):
                self.signal_list.item(i).setSelected(True)
        elif action == clear_action:
            self.clear_signals()
    
    def load_dbc_file(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open DBC File", "", "DBC Files (*.dbc);;All Files (*)")
        
        if file_path:
            try:
                self.db = cantools.database.load_file(file_path)
                self.dbc_file = file_path
                self.dbc_label.setText(os.path.basename(file_path))
                self.dbc_status.setText("DBC: Loaded")
                self.dbc_status.setStyleSheet("color: #4CAF50; font-size: 12px;")
                self.populate_message_list()
                print("DBC messages:", [f"0x{m.frame_id:X} - {m.name}" for m in self.db.messages])
                QMessageBox.information(self, "Success", "DBC file loaded successfully!")
                self.update_status("DBC file loaded")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load DBC file:\n{str(e)}")
                self.update_status("Failed to load DBC file")
    
    def clear_dbc(self):
        self.db = None
        self.dbc_file = None
        self.dbc_label.setText("No DBC file loaded")
        self.dbc_status.setText("DBC: None")
        self.dbc_status.setStyleSheet("color: #FF5722; font-size: 12px;")
        self.message_list.clear()
        self.signal_list.clear()
        self.update_status("DBC file cleared")
        if self.log_data is not None:
            self.signals_data = {}
            self.load_log_file(self.log_file)
    
    def load_log_file(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Log File", "", 
                "Log Files (*.asc *.blf *.csv);;All Files (*)")
        
        if file_path:
            self.log_file = file_path
            self.log_label.setText(os.path.basename(file_path))
            self.log_status.setText("Log: Loading...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.update_status("Loading log file...")
            
            self.loader_thread = LogLoaderThread(file_path, self.db)
            self.loader_thread.progress.connect(self.progress_bar.setValue)
            self.loader_thread.finished.connect(self.on_log_loaded)
            self.loader_thread.error.connect(self.on_log_error)
            self.loader_thread.start()
    
    def clear_log(self):
        self.log_data = None
        self.signals_data = {}
        self.log_file = None
        self.log_label.setText("No log file loaded")
        self.log_status.setText("Log: None")
        self.log_status.setStyleSheet("color: #FF5722; font-size: 12px;")
        self.message_list.clear()
        self.signal_list.clear()
        self.raw_data_table.setRowCount(0)
        self.start_time_spin.setRange(0, 999999)
        self.end_time_spin.setRange(0, 999999)
        self.update_status("Log file cleared")
        for i in range(self.plot_tabs.count() - 1, -1, -1):
            if self.plot_tabs.widget(i) != self.raw_data_tab:
                self.close_tab(i)
    
    def on_log_loaded(self, df, signals_data):
        print("Log DataFrame shape:", df.shape)
        print("Signals data keys:", list(signals_data.keys()))
        self.log_data = df
        self.signals_data = signals_data
        self.progress_bar.setValue(100)
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
        
        if not self.log_data.empty:
            min_time = self.log_data['timestamp'].min()
            max_time = self.log_data['timestamp'].max()
            self.start_time_spin.setRange(min_time, max_time)
            self.end_time_spin.setRange(min_time, max_time)
            self.start_time_spin.setValue(min_time)
            self.end_time_spin.setValue(max_time)
            print(f"Log timestamp range: {min_time} to {max_time}")
        
        self.populate_message_list()
        self.populate_raw_data_table()
        self.log_status.setText(f"Log: {len(self.log_data)} msgs")
        self.log_status.setStyleSheet("color: #4CAF50; font-size: 12px;")
        QMessageBox.information(self, "Success", f"Log file loaded with {len(self.log_data)} messages!")
        self.update_status(f"Loaded {len(self.log_data)} messages")
    
    def on_log_error(self, error):
        QMessageBox.critical(self, "Error", f"Failed to load log file:\n{error}")
        self.progress_bar.setVisible(False)
        self.log_status.setText("Log: None")
        self.update_status("Failed to load log file")
    
    def populate_message_list(self):
        self.message_list.clear()
        if self.db:
            for message in self.db.messages:
                self.message_list.addItem(f"0x{message.frame_id:X} - {message.name}")
        elif self.log_data is not None:
            unique_ids = sorted(self.log_data['arbitration_id'].unique())
            for arb_id in unique_ids:
                self.message_list.addItem(f"0x{arb_id:X}")
    
    def update_signal_list(self):
        self.signal_list.clear()
        selected_items = self.message_list.selectedItems()
        if not selected_items:
            self.update_status("No messages selected")
            return
        
        signals_found = False
        if self.db:
            for item in selected_items:
                msg_id = int(item.text().split(' ')[0][2:], 16)
                try:
                    message = self.db.get_message_by_frame_id(msg_id)
                    print(f"Populating signals for {message.name} (ID 0x{msg_id:X})")
                    for signal in message.signals:
                        signal_key = f"{message.name}.{signal.name}"
                        prefix = "[PDU] " if signal.multiplexer_ids else ""
                        self.signal_list.addItem(f"{prefix}{signal_key}")
                        signals_found = True
                        print(f"Added signal: {prefix}{signal_key}, Multiplexer IDs: {signal.multiplexer_ids}, Is Multiplexer: {signal.is_multiplexer}")
                        item_a = self.signal_list.item(self.signal_list.count() - 1)
                        if signal.multiplexer_ids:
                            font = item_a.font()
                            font.setBold(True)
                            item_a.setFont(font)
                    if message.is_multiplexed():
                        print(f"Multiplexed message {message.name} detected")
                except KeyError:
                    self.update_status(f"No DBC message found for ID 0x{msg_id:X}")
                    continue
        else:
            for item in selected_items:
                arb_id = int(item.text()[2:], 16)
                max_dlc = self.log_data[self.log_data['arbitration_id'] == arb_id]['dlc'].max()
                for i in range(max_dlc):
                    signal_key = f"ID_0x{arb_id:X}.Byte{i}"
                    self.signal_list.addItem(signal_key)
                    signals_found = True
                    print(f"Added signal: {signal_key}")
        
        if not signals_found:
            QMessageBox.warning(self, "Warning", "No signals found for selected messages!")
            self.update_status("No signals available for selected messages")
        else:
            print("Signals populated:", [self.signal_list.item(i).text() for i in range(self.signal_list.count())])
    
    def clear_messages(self):
        self.message_list.clear()
        self.signal_list.clear()
        self.update_status("Message list cleared")
    
    def clear_signals(self):
        self.signal_list.clear()
        self.update_status("Signal list cleared")
    
    def populate_raw_data_table(self):
        if self.log_data is None:
            self.raw_data_table.setRowCount(0)
            return
        
        self.raw_data_table.setRowCount(len(self.log_data))
        for i, row in self.log_data.iterrows():
            self.raw_data_table.setItem(i, 0, QTableWidgetItem(f"{row['timestamp']:.6f}"))
            self.raw_data_table.setItem(i, 1, QTableWidgetItem(f"0x{row['arbitration_id']:X}"))
            data_str = ' '.join(f'{b:02X}' for b in row['data'])
            self.raw_data_table.setItem(i, 2, QTableWidgetItem(data_str))
            self.raw_data_table.setItem(i, 3, QTableWidgetItem(str(row['dlc'])))
            self.raw_data_table.setItem(i, 4, QTableWidgetItem(str(row['channel'])))
            self.raw_data_table.setItem(i, 5, QTableWidgetItem("Yes" if row['is_fd'] else "No"))
            self.raw_data_table.setItem(i, 6, QTableWidgetItem("Yes" if row['bitrate_switch'] else "No"))
        self.raw_data_table.resizeColumnsToContents()
    
    def filter_raw_data(self):
        filter_text = self.raw_filter.text().lower()
        if not filter_text:
            for row in range(self.raw_data_table.rowCount()):
                self.raw_data_table.setRowHidden(row, False)
            return
        
        for row in range(self.raw_data_table.rowCount()):
            row_data = []
            for col in range(self.raw_data_table.columnCount()):
                item = self.raw_data_table.item(row, col)
                row_data.append(item.text().lower() if item else "")
            matches = any(filter_text in data for data in row_data)
            self.raw_data_table.setRowHidden(row, not matches)
    
    def export_raw_data(self):
        if self.log_data is None:
            QMessageBox.warning(self, "Warning", "No raw data to export!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Raw Data", "", "CSV Files (*.csv)")
        if file_path:
            self.log_data.to_csv(file_path, index=False)
            self.update_status("Raw data exported as CSV")
    
    def filter_messages(self):
        filter_text = self.msg_filter.text().lower()
        for i in range(self.message_list.count()):
            item = self.message_list.item(i)
            matches = filter_text in item.text().lower()
            item.setHidden(not matches)
            item.setBackground(Qt.yellow if matches and filter_text else Qt.transparent)
    
    def filter_signals(self):
        filter_text = self.signal_filter.text().lower()
        for i in range(self.signal_list.count()):
            item = self.signal_list.item(i)
            matches = filter_text in item.text().lower()
            item.setHidden(not matches)
            item.setBackground(Qt.yellow if matches and filter_text else Qt.transparent)
    
    def toggle_time_range(self, text):
        enabled = text == "Custom"
        self.start_time_spin.setEnabled(enabled)
        self.end_time_spin.setEnabled(enabled)
    
    def plot_signals(self, update=False):
        if self.log_data is None:
            QMessageBox.warning(self, "Warning", "Please load a log file first!")
            return
            
        selected_signals = [item.text() for item in self.signal_list.selectedItems()]
        if not selected_signals:
            QMessageBox.warning(self, "Warning", "Please select at least one signal to plot!")
            return
            
        print("Selected signals:", selected_signals)
        print("Available signals in signals_data:", list(self.signals_data.keys()))
        
        valid_signals = []
        missing_signals = []
        for s in selected_signals:
            clean_signal = s.replace('[PDU] ', '')
            if clean_signal in self.signals_data:
                valid_signals.append(clean_signal)
            else:
                missing_signals.append(clean_signal)
        
        if missing_signals:
            print(f"Missing signals in signals_data: {missing_signals}")
        if not valid_signals:
            QMessageBox.warning(self, "Warning", f"No data available for selected signals!\nSelected: {selected_signals}\nAvailable: {list(self.signals_data.keys())}")
            return
            
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plot_styles = {
            s: {
                'color': color_cycle[i % len(color_cycle)],
                'linestyle': self.linestyle_combo.currentText(),
                'marker': self.marker_combo.currentText()
            } for i, s in enumerate(selected_signals)
        }
        
        if not update:
            tab_title = " | ".join(selected_signals[:3] + ["..."]) if len(selected_signals) > 3 else " | ".join(selected_signals)
            fig, canvas, stats_label = self.add_plot_tab(tab_title)
        else:
            current_tab = self.plot_tabs.currentWidget()
            if current_tab == self.raw_data_tab:
                QMessageBox.warning(self, "Warning", "Cannot update the Raw Data tab!")
                return
            canvas = current_tab.findChild(FigureCanvas)
            stats_label = current_tab.findChild(QLabel)
            fig = canvas.figure
        
        fig.clear()
        
        time_range = None
        if self.time_range_combo.currentText() == "Custom":
            start_time = self.start_time_spin.value()
            end_time = self.end_time_spin.value()
            if start_time >= end_time:
                QMessageBox.warning(self, "Warning", "Start time must be less than end time!")
                return
            has_data = False
            for signal in valid_signals:
                data = self.signals_data[signal]
                filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
                if not filtered_data.empty:
                    has_data = True
                    print(f"Signal {signal} has {len(filtered_data)} data points in time range {start_time} to {end_time}")
                else:
                    print(f"Signal {signal} has no data in time range {start_time} to {end_time}")
            if not has_data:
                print("No data in custom time range, falling back to full log range")
                all_timestamps = []
                for signal in valid_signals:
                    all_timestamps.extend(self.signals_data[signal]['timestamp'].values)
                if all_timestamps:
                    start_time = min(all_timestamps)
                    end_time = max(all_timestamps)
                    print(f"Fallback time range: {start_time} to {end_time}")
                else:
                    QMessageBox.warning(self, "Warning", "No data points available for selected signals!")
                    return
            time_range = (start_time, end_time)
        
        channel = None
        channel_filter = None
        
        self.update_status("Plotting signals...")
        self.plotter_thread = PlotterThread(
            self.signals_data, valid_signals, time_range, 
            self.normalize_check.isChecked(), channel_filter, plot_styles
        )
        self.plotter_thread.finished.connect(lambda data: self.on_plot_finished(data, fig, canvas, stats_label))
        self.plotter_thread.error.connect(lambda err: QMessageBox.warning(self, "Error", f"Plotting failed: {err}"))
        self.plotter_thread.start()
    
    def on_plot_finished(self, plot_data, fig, canvas, stats_label):
        axes = {}
        stats_text = []
        lines = []
        
        ax = fig.add_subplot(111)
        ax.set_facecolor('#353535')
        axes[list(plot_data.keys())[0]] = ax if plot_data else None
        
        for i, (signal, data) in enumerate(plot_data.items()):
            if len(data['values']) == 0:
                self.update_status(f"No data for {signal}")
                continue
                
            if i == 0:
                current_ax = ax
            else:
                current_ax = ax.twinx()
                current_ax.spines['right'].set_position(('outward', 60 * (i-1)))
                axes[signal] = current_ax
            
            style = data['style']
            line, = current_ax.plot(
                data['timestamps'], data['values'], 
                label=signal, 
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=3
            )
            lines.append((line, signal, data['timestamps'], data['values']))
            
            current_ax.set_ylabel(signal, color=style['color'])
            current_ax.tick_params(axis='y', colors=style['color'])
            
            stats = data['stats']
            stats_text.append(
                f"{signal}: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}"
            )
        
        ax.set_xlabel('Timestamp (s)')
        ax.set_title('CAN Signals', color='#FFFFFF')
        ax.tick_params(axis='x', colors='#FFFFFF')
        
        if self.legend_check.isChecked():
            lines_list, labels = [], []
            for ax in axes.values():
                l, lb = ax.get_legend_handles_labels()
                lines_list.extend(l)
                labels.extend(lb)
            legend = ax.legend(lines_list, labels, loc='upper left', facecolor='#424242', edgecolor='#FFFFFF', labelcolor='#FFFFFF')
            for text in legend.get_texts():
                text.set_color('#FFFFFF')
        
        if self.grid_check.isChecked():
            ax.grid(True)
        
        annotation = None
        def on_hover(event):
            nonlocal annotation
            if event.inaxes != ax and not any(event.inaxes == axes[sig] for sig in axes):
                if annotation:
                    annotation.remove()
                    annotation = None
                    canvas.draw_idle()
                return
            
            closest_dist = float('inf')
            closest_data = None
            
            for line, signal, timestamps, values in lines:
                cont, ind = line.contains(event)
                if cont:
                    idx = ind['ind'][0]
                    x, y = timestamps[idx], values[idx]
                    pos = line.get_xydata()[idx]
                    dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_data = (signal, x, y, pos)
            
            if closest_data and closest_dist < 0.1:
                signal, x, y, pos = closest_data
                if annotation:
                    annotation.remove()
                annotation = ax.annotate(
                    f"{signal}\nTime: {x:.6f}\nValue: {y:.2f}",
                    xy=(pos[0], pos[1]), xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#FFFF99", alpha=0.9, ec="#555555"),
                    arrowprops=dict(arrowstyle="->", color="#FFFFFF")
                )
                canvas.draw_idle()
            else:
                if annotation:
                    annotation.remove()
                    annotation = None
                    canvas.draw_idle()
        
        canvas.mpl_connect('motion_notify_event', on_hover)
        
        fig.tight_layout()
        canvas.draw()
        
        canvas.plot_data = plot_data
        
        stats_label.setText("Statistics:\n" + "\n".join(stats_text) if stats_text else "Statistics: No data")
        self.update_status("Plotting complete")
    
    def export_plot(self):
        if self.plot_tabs.count() == 0:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot/Data", "", "PNG Files (*.png);;CSV Files (*.csv)")
        if file_path:
            current_tab = self.plot_tabs.currentWidget()
            if current_tab == self.raw_data_tab:
                QMessageBox.warning(self, "Warning", "Cannot export Raw Data tab as plot!")
                return
            canvas = current_tab.findChild(FigureCanvas)
            if file_path.endswith('.png'):
                canvas.figure.savefig(file_path, facecolor='#2E2E2E')
                self.update_status("Plot exported as PNG")
            elif file_path.endswith('.csv'):
                plot_data = getattr(canvas, 'plot_data', {})
                if plot_data:
                    df = pd.DataFrame()
                    for signal, data in plot_data.items():
                        df[signal + '_time'] = data['timestamps']
                        df[signal + '_value'] = data['values']
                    df.to_csv(file_path, index=False)
                    self.update_status("Data exported as CSV")
                else:
                    QMessageBox.warning(self, "Warning", "No data available to export!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CANAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
