import os
import sys
from pathlib import Path
import re

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QPixmap
import urllib.request
import json
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLineEdit,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QFrame,
    QGridLayout,
)

class StatusCard(QFrame):
    def __init__(self, title: str, initial_status: str = "Idle"):
        super().__init__()
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel(initial_status)
        self.status_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)

        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #222;
        """)

        self.status_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
            border-radius: 10px;
            background-color: #6c757d;
        """)

        self.setStyleSheet("""
            QFrame {
                background: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 14px;
            }
        """)

    def set_status(self, text: str, kind: str = "idle"):
        colors = {
            "idle": "#6c757d",
            "running": "#0d6efd",
            "ready": "#198754",
            "error": "#dc3545",
            "warning": "#ffc107",
            "done": "#198754",
        }
        color = colors.get(kind, "#6c757d")

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 10px;
            border-radius: 10px;
            background-color: {color};
        """)

class ThesisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predictive Maintenance Thesis App")
        self.resize(1200, 800)

        self.base_dir = Path(__file__).resolve().parent
        self.python_exe = sys.executable

        self.train_process = None
        self.server_process = None
        self.twin_process = None

        self.server_ready = False
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(1500)
        self.health_timer.timeout.connect(self.poll_server_health)

        self._build_ui()
        self._refresh_button_states()
        self._refresh_images()
        self.health_timer.start()

    def _make_scrollable(self, widget: QWidget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll
    
    def _toggle_widget_visibility(self, widget: QWidget, button: QPushButton, show_text: str, hide_text: str):
        visible = not widget.isVisible()
        widget.setVisible(visible)
        button.setText(hide_text if visible else show_text)
    
    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        header_layout = QHBoxLayout()
        self.training_card = StatusCard("Training", "Idle")
        self.server_card = StatusCard("PLC Server", "Stopped")
        self.twin_card = StatusCard("Digital Twin", "Idle")

        header_layout.addWidget(self.training_card)
        header_layout.addWidget(self.server_card)
        header_layout.addWidget(self.twin_card)

        tabs = QTabWidget()
        tabs.addTab(self._make_scrollable(self._build_training_tab()), "Training")
        tabs.addTab(self._make_scrollable(self._build_service_tab()), "PLC Service")
        tabs.addTab(self._make_scrollable(self._build_twin_tab()), "Digital Twin")
        tabs.addTab(self._make_scrollable(self._build_results_tab()), "Results")

        main_layout.addLayout(header_layout)
        main_layout.addWidget(tabs)

        self.setCentralWidget(central)

    def _build_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_box = QGroupBox("Training Configuration")
        path_box.setVisible(False)
        path_form = QFormLayout(path_box)

        self.training_script_edit = QLineEdit(str(self.base_dir / "two_stage_multiclass_ai4i2020_extended.py"))
        self.train_data_edit = QLineEdit(str(self.base_dir / "ai4i2020.csv"))

        browse_training_script = QPushButton("Browse")
        browse_training_script.clicked.connect(
            lambda: self._browse_file(self.training_script_edit, "Select training script", "Python Files (*.py)")
        )
        browse_train_data = QPushButton("Browse")
        browse_train_data.clicked.connect(
            lambda: self._browse_file(self.train_data_edit, "Select training CSV", "CSV Files (*.csv)")
        )

        row1 = QHBoxLayout()
        row1.addWidget(self.training_script_edit)
        row1.addWidget(browse_training_script)
        row1w = QWidget()
        row1w.setLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.train_data_edit)
        row2.addWidget(browse_train_data)
        row2w = QWidget()
        row2w.setLayout(row2)

        path_form.addRow("Training script:", row1w)
        path_form.addRow("Training CSV:", row2w)

        toggle_training_paths_btn = QPushButton("Browse Training Files")
        toggle_training_paths_btn.setObjectName("browseButton")
        toggle_training_paths_btn.clicked.connect(
            lambda: self._toggle_widget_visibility(
                path_box,
                toggle_training_paths_btn,
                "Browse Training Files",
                "Hide Training Files"
            )
        )

        self.btn_train_binary = QPushButton("Train Binary")
        self.btn_train_multiclass = QPushButton("Train Multiclass")
        self.btn_train_all = QPushButton("Train All")

        self.btn_train_binary.clicked.connect(lambda: self.start_training("binary"))
        self.btn_train_multiclass.clicked.connect(lambda: self.start_training("multiclass"))
        self.btn_train_all.clicked.connect(lambda: self.start_training("all"))

        toggle_training_paths_btn.setFixedWidth(180)
        self.btn_train_binary.setFixedWidth(120)
        self.btn_train_multiclass.setFixedWidth(150)
        self.btn_train_all.setFixedWidth(100)

        button_row = QHBoxLayout()
        button_row.addWidget(toggle_training_paths_btn)
        button_row.addWidget(self.btn_train_binary)
        button_row.addWidget(self.btn_train_multiclass)
        button_row.addWidget(self.btn_train_all)
        button_row.addStretch()

        self.training_status = QLabel("Idle")
        self.training_status.setAlignment(Qt.AlignLeft)

        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)

        layout.addLayout(button_row)
        layout.addWidget(path_box)
        layout.addWidget(QLabel("Training status:"))
        layout.addWidget(self.training_status)
        layout.addWidget(QLabel("Training log:"))
        layout.addWidget(self.training_log)

        return tab

    def _build_service_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_box = QGroupBox("PLC Configuration")
        path_box.setVisible(False)
        form = QFormLayout(path_box)

        self.server_script_edit = QLineEdit(str(self.base_dir / "plc_ai_service_robust.py"))
        self.server_host_edit = QLineEdit("127.0.0.1")
        self.server_port_edit = QLineEdit("5000")

        browse_server_script = QPushButton("Browse")
        browse_server_script.clicked.connect(
            lambda: self._browse_file(self.server_script_edit, "Select PLC server script", "Python Files (*.py)")
        )

        row = QHBoxLayout()
        row.addWidget(self.server_script_edit)
        row.addWidget(browse_server_script)
        roww = QWidget()
        roww.setLayout(row)

        form.addRow("Server script:", roww)
        form.addRow("Host:", self.server_host_edit)
        form.addRow("Port:", self.server_port_edit)

        toggle_plc_paths_btn = QPushButton("Browse PLC Files")
        toggle_plc_paths_btn.setObjectName("browseButton")
        toggle_plc_paths_btn.clicked.connect(
            lambda: self._toggle_widget_visibility(
                path_box,
                toggle_plc_paths_btn,
                "Browse PLC Files",
                "Hide PLC Files"
            )
        )

        self.btn_start_server = QPushButton("Start PLC Server")
        self.btn_stop_server = QPushButton("Stop PLC Server")
        self.btn_start_server.clicked.connect(self.start_server)
        self.btn_stop_server.clicked.connect(self.stop_server)

        toggle_plc_paths_btn.setFixedWidth(160)
        self.btn_start_server.setFixedWidth(140)
        self.btn_stop_server.setFixedWidth(140)

        button_row = QHBoxLayout()
        button_row.addWidget(toggle_plc_paths_btn)
        button_row.addWidget(self.btn_start_server)
        button_row.addWidget(self.btn_stop_server)
        button_row.addStretch()

        self.server_status = QLabel("Stopped")
        self.server_health_details = QLabel("Health: unavailable")

        self.service_log = QPlainTextEdit()
        self.service_log.setReadOnly(True)

        layout.addLayout(button_row)
        layout.addWidget(path_box)
        layout.addWidget(QLabel("Service status:"))
        layout.addWidget(self.server_status)
        layout.addWidget(self.server_health_details)
        layout.addWidget(QLabel("Service log:"))
        layout.addWidget(self.service_log)

        return tab

    def _build_twin_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        path_box = QGroupBox("Digital Twin Configuration")
        path_box.setVisible(False)
        form = QFormLayout(path_box)

        self.twin_script_edit = QLineEdit(str(self.base_dir / "digital_twin_from_csv_dashboard_robust.py"))
        self.twin_data_edit = QLineEdit(str(self.base_dir / "ai4i2020_Test.csv"))
        self.output_image_edit = QLineEdit(str(self.base_dir / "digital_twin_csv_dashboard.png"))

        browse_twin_script = QPushButton("Browse")
        browse_twin_script.clicked.connect(
            lambda: self._browse_file(self.twin_script_edit, "Select digital twin script", "Python Files (*.py)")
        )
        browse_twin_data = QPushButton("Browse")
        browse_twin_data.clicked.connect(
            lambda: self._browse_file(self.twin_data_edit, "Select digital twin CSV", "CSV Files (*.csv)")
        )
        browse_twin_output = QPushButton("Browse")
        browse_twin_output.clicked.connect(self._browse_output_image)

        row1 = QHBoxLayout()
        row1.addWidget(self.twin_script_edit)
        row1.addWidget(browse_twin_script)
        row1w = QWidget()
        row1w.setLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.twin_data_edit)
        row2.addWidget(browse_twin_data)
        row2w = QWidget()
        row2w.setLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(self.output_image_edit)
        row3.addWidget(browse_twin_output)
        row3w = QWidget()
        row3w.setLayout(row3)

        form.addRow("Twin script:", row1w)
        form.addRow("Test CSV:", row2w)
        form.addRow("Output image:", row3w)

        toggle_twin_paths_btn = QPushButton("Browse Digital Twin Files")
        toggle_twin_paths_btn.setObjectName("browseButton")
        toggle_twin_paths_btn.clicked.connect(
            lambda: self._toggle_widget_visibility(
                path_box,
                toggle_twin_paths_btn,
                "Browse Digital Twin Files",
                "Hide Digital Twin Files"
            )
        )

        self.btn_start_twin = QPushButton("Start Digital Twin")
        self.btn_stop_twin = QPushButton("Stop Digital Twin")
        self.btn_start_twin.clicked.connect(self.start_twin)
        self.btn_stop_twin.clicked.connect(self.stop_twin)

        toggle_twin_paths_btn.setFixedWidth(200)
        self.btn_start_twin.setFixedWidth(140)
        self.btn_stop_twin.setFixedWidth(140)

        button_row = QHBoxLayout()
        button_row.addWidget(toggle_twin_paths_btn)
        button_row.addWidget(self.btn_start_twin)
        button_row.addWidget(self.btn_stop_twin)
        button_row.addStretch()

        live_box = QGroupBox("Live Prediction Monitor")
        live_grid = QGridLayout(live_box)
        live_grid.setHorizontalSpacing(16)
        live_grid.setVerticalSpacing(8)

        self.alarm_indicator = QLabel("NO DATA")
        self.alarm_indicator.setAlignment(Qt.AlignCenter)
        self.alarm_indicator.setMinimumSize(180, 140)
        self.alarm_indicator.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: white;
            background-color: #6c757d;
            border-radius: 12px;
            padding: 12px;
        """)

        self.lbl_live_timestep = QLabel("-")
        self.lbl_live_probability = QLabel("-")
        self.lbl_live_pred = QLabel("-")
        self.lbl_live_true = QLabel("-")
        self.lbl_live_type = QLabel("-")
        self.lbl_live_alarm = QLabel("-")

        labels = [
            QLabel("Timestep"),
            QLabel("Failure Probability"),
            QLabel("Predicted Failure"),
            QLabel("True Failure"),
            QLabel("Failure Type"),
            QLabel("Alarm Level"),
        ]

        values = [
            self.lbl_live_timestep,
            self.lbl_live_probability,
            self.lbl_live_pred,
            self.lbl_live_true,
            self.lbl_live_type,
            self.lbl_live_alarm,
        ]

        for lbl in labels:
            lbl.setStyleSheet("font-weight: bold;")
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        for val in values:
            val.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            val.setStyleSheet("""
                background: white;
                border: 1px solid #d3d9e3;
                border-radius: 6px;
                padding: 4px 8px;
            """)

        for i, (lbl, val) in enumerate(zip(labels, values)):
            live_grid.addWidget(lbl, i, 0)
            live_grid.addWidget(val, i, 1)

        live_grid.addWidget(QLabel("Alarm Indicator"), 0, 2)
        live_grid.addWidget(self.alarm_indicator, 1, 2, 5, 1)

        live_grid.setColumnStretch(0, 0)
        live_grid.setColumnStretch(1, 1)
        live_grid.setColumnStretch(2, 0)
        self.twin_status = QLabel("Idle")
        self.twin_log = QPlainTextEdit()
        self.twin_log.setReadOnly(True)

        layout.addLayout(button_row)
        layout.addWidget(path_box)
        layout.addWidget(live_box)
        layout.addWidget(QLabel("Digital twin status:"))
        layout.addWidget(self.twin_status)
        layout.addWidget(QLabel("Digital twin log:"))
        layout.addWidget(self.twin_log)

        return tab
    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.lbl_stage1_metrics = QLabel("Stage 1 metrics image not found")
        self.lbl_stage1_cm = QLabel("Stage 1 confusion matrix not found")
        self.lbl_stage2_metrics = QLabel("Stage 2 metrics image not found")
        self.lbl_stage2_cm = QLabel("Stage 2 confusion matrix not found")
        self.lbl_twin_dashboard = QLabel("Digital twin dashboard not found")

        for lbl in [
            self.lbl_stage1_metrics,
            self.lbl_stage1_cm,
            self.lbl_stage2_metrics,
            self.lbl_stage2_cm,
            self.lbl_twin_dashboard,
        ]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(320)
            lbl.setStyleSheet("""
                border: 1px solid gray;
                background: white;
                padding: 8px;
            """)

        self.btn_refresh_results = QPushButton("Refresh Images")
        self.btn_refresh_results.clicked.connect(self._refresh_images)

        layout.addWidget(self.btn_refresh_results)

        stage1_box = QGroupBox("Stage 1 Results")
        stage1_grid = QGridLayout(stage1_box)
        stage1_grid.addWidget(QLabel("Model Comparison"), 0, 0)
        stage1_grid.addWidget(QLabel("Confusion Matrix"), 0, 1)
        stage1_grid.addWidget(self.lbl_stage1_metrics, 1, 0)
        stage1_grid.addWidget(self.lbl_stage1_cm, 1, 1)

        stage2_box = QGroupBox("Stage 2 Results")
        stage2_grid = QGridLayout(stage2_box)
        stage2_grid.addWidget(QLabel("Model Comparison"), 0, 0)
        stage2_grid.addWidget(QLabel("Confusion Matrix"), 0, 1)
        stage2_grid.addWidget(self.lbl_stage2_metrics, 1, 0)
        stage2_grid.addWidget(self.lbl_stage2_cm, 1, 1)

        twin_box = QGroupBox("Digital Twin Dashboard")
        twin_layout = QVBoxLayout(twin_box)
        twin_layout.addWidget(self.lbl_twin_dashboard)

        layout.addWidget(stage1_box)
        layout.addWidget(stage2_box)
        layout.addWidget(twin_box)

        return tab

    def _browse_file(self, line_edit, title, filter_text):
        path, _ = QFileDialog.getOpenFileName(self, title, str(self.base_dir), filter_text)
        if path:
            line_edit.setText(path)

    def _browse_output_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select output image", str(self.base_dir / "digital_twin_csv_dashboard.png"), "PNG Files (*.png)")
        if path:
            self.output_image_edit.setText(path)

    def _append_log(self, widget: QPlainTextEdit, text: str):
        if not text:
            return
        widget.appendPlainText(text.rstrip())
        widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())

    def _set_alarm_indicator(self, alarm_text: str):
        colors = {
            "GREEN": "#198754",
            "YELLOW": "#ffc107",
            "RED": "#dc3545",
            "NO DATA": "#6c757d",
        }
        text_color = "black" if alarm_text == "YELLOW" else "white"
        bg = colors.get(alarm_text, "#6c757d")

        self.alarm_indicator.setText(alarm_text)
        self.alarm_indicator.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {text_color};
            background-color: {bg};
            border-radius: 12px;
            padding: 12px;
        """)

    def _reset_live_twin_panel(self):
        self.lbl_live_timestep.setText("-")
        self.lbl_live_probability.setText("-")
        self.lbl_live_pred.setText("-")
        self.lbl_live_true.setText("-")
        self.lbl_live_type.setText("-")
        self.lbl_live_alarm.setText("-")
        self._set_alarm_indicator("NO DATA")

    def _parse_twin_line(self, line: str):
        pattern = (
            r"\[t=\s*(?P<t>\d+)\]\s+"
            r"Wear=(?P<wear>[-\d.]+)\s+\|\s+"
            r"Torque=(?P<torque>[-\d.]+)\s+\|\s+"
            r"Prob=\s*(?P<prob>[-\d.]+)\s+\|\s+"
            r"Pred=(?P<pred>\d+)\s+\|\s+"
            r"True=(?P<true>\d+)\s+\|\s+"
            r"Type=(?P<ftype>[A-Za-z0-9_]+)\s+\|\s+"
            r"Alarm=(?P<alarm>[A-Z]+)"
        )

        match = re.search(pattern, line)
        if not match:
            return

        data = match.groupdict()

        self.lbl_live_timestep.setText(data["t"])
        self.lbl_live_probability.setText(data["prob"])
        self.lbl_live_pred.setText(data["pred"])
        self.lbl_live_true.setText(data["true"])
        self.lbl_live_type.setText(data["ftype"])
        self.lbl_live_alarm.setText(data["alarm"])
        self._set_alarm_indicator(data["alarm"])

    def _make_process(self, on_stdout, on_finished):
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: on_stdout(proc))
        proc.finished.connect(on_finished)
        return proc

    def start_training(self, mode: str):
        if self.train_process is not None:
            QMessageBox.warning(self, "Training running", "A training process is already running.")
            return

        script_path = self.training_script_edit.text().strip()
        data_path = self.train_data_edit.text().strip()
        if not Path(script_path).exists():
            QMessageBox.critical(self, "Missing file", "Training script not found.")
            return
        if not Path(data_path).exists():
            QMessageBox.critical(self, "Missing file", "Training CSV not found.")
            return

        self.training_log.clear()
        self.training_status.setText(f"Running {mode} training...")
        self.training_card.set_status("Running", "running")

        env = os.environ.copy()
        env["DATA_PATH"] = data_path

        self.train_process = self._make_process(self._read_training_output, self._training_finished)
        self.train_process.setWorkingDirectory(str(Path(script_path).parent))
        self.train_process.setProcessEnvironment(self.train_process.processEnvironment())

        self.train_process.start(self.python_exe, [script_path, "--mode", mode])
        if not self.train_process.waitForStarted(3000):
            self.train_process = None
            self.training_status.setText("Failed to start training")
            QMessageBox.critical(self, "Start failed", "Could not start training process.")
            return

        self._refresh_button_states()

    def _read_training_output(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        self._append_log(self.training_log, data)

    def _training_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.training_status.setText("Finished successfully")
            self.training_card.set_status("Done", "done")
        else:
            self.training_status.setText(f"Failed (exit code {exit_code})")
            self.training_card.set_status("Error", "error")
        self.train_process = None
        self._refresh_button_states()
        self._refresh_images()

    def start_server(self):
        if self.server_process is not None:
            QMessageBox.warning(self, "Server running", "PLC server is already running.")
            return

        script_path = self.server_script_edit.text().strip()
        if not Path(script_path).exists():
            QMessageBox.critical(self, "Missing file", "PLC server script not found.")
            return

        host = self.server_host_edit.text().strip() or "127.0.0.1"
        port = self.server_port_edit.text().strip() or "5000"

        self.service_log.clear()
        self.server_status.setText("Starting...")
        self.server_card.set_status("Starting", "running")
        self.server_health_details.setText("Health: checking...")
        self.server_ready = False

        self.server_process = self._make_process(self._read_server_output, self._server_finished)
        self.server_process.setWorkingDirectory(str(Path(script_path).parent))
        self.server_process.start(self.python_exe, [script_path, "--host", host, "--port", port])

        if not self.server_process.waitForStarted(3000):
            self.server_process = None
            self.server_status.setText("Failed to start")
            QMessageBox.critical(self, "Start failed", "Could not start PLC server process.")
            return

        self._refresh_button_states()

    def _read_server_output(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        self._append_log(self.service_log, data)
        if "Address already in use" in data:
            self.server_card.set_status("Error", "error")
            self.server_health_details.setText("Health: port already in use")
        self._refresh_button_states()
        

    def _server_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.server_card.set_status("Stopped", "idle")
        else:
            self.server_card.set_status("Stopped", "idle")
        self.server_status.setText(f"Stopped (exit code {exit_code})")
        self.server_health_details.setText("Health: unavailable")
        self.server_ready = False
        self.server_process = None
        self._refresh_button_states()

    def stop_server(self):
        if self.server_process is None:
            return
        self.server_process.terminate()
        if not self.server_process.waitForFinished(3000):
            self.server_process.kill()
            self.server_process.waitForFinished(2000)

    def start_twin(self):
        if self.twin_process is not None:
            QMessageBox.warning(self, "Digital twin running", "A digital twin process is already running.")
            return

        if self.server_process is None:
            QMessageBox.warning(self, "Server required", "Start the PLC server first.")
            return
        if not self.server_ready:
            QMessageBox.warning(self, "Server not ready", "Wait until the PLC health status becomes Ready.")
            return
        self.twin_card.set_status("Streaming", "running")
        script_path = self.twin_script_edit.text().strip()
        data_path = self.twin_data_edit.text().strip()
        output_image = self.output_image_edit.text().strip()

        if not Path(script_path).exists():
            QMessageBox.critical(self, "Missing file", "Digital twin script not found.")
            return
        if not Path(data_path).exists():
            QMessageBox.critical(self, "Missing file", "Digital twin CSV not found.")
            return

        server_base = f"http://{self.server_host_edit.text().strip() or '127.0.0.1'}:{self.server_port_edit.text().strip() or '5000'}"

        self.twin_log.clear()
        self.twin_status.setText("Running...")
        self._reset_live_twin_panel()

        self.twin_process = self._make_process(self._read_twin_output, self._twin_finished)
        self.twin_process.setWorkingDirectory(str(Path(script_path).parent))
        self.twin_process.start(
            self.python_exe,
            [
                "-u",
                script_path,
                "--server-base", server_base,
                "--data-path", data_path,
                "--output-image", output_image,
                "--no-show",
            ],
        )

        if not self.twin_process.waitForStarted(3000):
            self.twin_process = None
            self.twin_status.setText("Failed to start")
            QMessageBox.critical(self, "Start failed", "Could not start digital twin process.")
            return

        self._refresh_button_states()

    def _read_twin_output(self, proc):
        data = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        self._append_log(self.twin_log, data)

        for line in data.splitlines():
            self._parse_twin_line(line)

    def _twin_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.twin_status.setText("Finished successfully")
            self.twin_card.set_status("Done", "done")
        else:
            self.twin_status.setText(f"Failed (exit code {exit_code})")
            self.twin_card.set_status("Error", "error")
        
        self.twin_process = None
        self._refresh_button_states()
        self._refresh_images()

    def stop_twin(self):
        if self.twin_process is None:
            return
        self.twin_process.terminate()
        if not self.twin_process.waitForFinished(3000):
            self.twin_process.kill()
            self.twin_process.waitForFinished(2000)
        self.twin_card.set_status("Stopped", "idle")

    def _set_image(self, label: QLabel, path: Path):
        if not path.exists():
            label.setText(f"Not found:\n{path.name}")
            label.setPixmap(QPixmap())
            return

        pix = QPixmap(str(path))
        if pix.isNull():
            label.setText(f"Could not load:\n{path.name}")
            return

        scaled = pix.scaled(900, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setText("")

    def _refresh_images(self):
        self._set_image(self.lbl_stage1_metrics, self.base_dir / "stage1_binary_model_metrics.png")
        self._set_image(self.lbl_stage1_cm, self.base_dir / "stage1_binary_confusion_matrix.png")
        self._set_image(self.lbl_stage2_metrics, self.base_dir / "stage2_multiclass_model_metrics.png")
        self._set_image(self.lbl_stage2_cm, self.base_dir / "stage2_multiclass_confusion_matrix.png")
        self._set_image(self.lbl_twin_dashboard, Path(self.output_image_edit.text()) if hasattr(self, "output_image_edit") else self.base_dir / "digital_twin_csv_dashboard.png")

    def _refresh_button_states(self):
        training_running = self.train_process is not None
        server_running = self.server_process is not None
        twin_running = self.twin_process is not None

        self.btn_train_binary.setEnabled(not training_running)
        self.btn_train_multiclass.setEnabled(not training_running)
        self.btn_train_all.setEnabled(not training_running)

        self.btn_start_server.setEnabled(not server_running)
        self.btn_stop_server.setEnabled(server_running)

        self.btn_start_twin.setEnabled(server_running and self.server_ready and not twin_running)
        self.btn_stop_twin.setEnabled(twin_running)

    def poll_server_health(self):
        host = self.server_host_edit.text().strip() or "127.0.0.1"
        port = self.server_port_edit.text().strip() or "5000"
        url = f"http://{host}:{port}/health"

        if self.server_process is None:
            self.server_status.setText("Stopped")
            self.server_health_details.setText("Health: unavailable")
            self.server_ready = False
            self._refresh_button_states()
            return

        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                body = response.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                ready = bool(data.get("ready", False))
                stage1_loaded = data.get("stage1_loaded", False)
                stage2_loaded = data.get("stage2_loaded", False)
                self.server_ready = ready
                if ready:
                    self.server_status.setText("Ready")
                    self.server_card.set_status("Ready", "ready")
                else:
                    self.server_status.setText("Running")
                    self.server_card.set_status("Running", "running")
                self.server_health_details.setText(
                    f"Health: ready={ready} | stage1_loaded={stage1_loaded} | stage2_loaded={stage2_loaded}"
                )
        except Exception:
            self.server_ready = False
            if self.server_process is not None:
                self.server_status.setText("Starting...")
                self.server_health_details.setText("Health: waiting for /health response")
                self.server_card.set_status("Starting", "running")

        self._refresh_button_states()

    def closeEvent(self, event):
        if self.twin_process is not None:
            self.stop_twin()
        if self.server_process is not None:
            self.stop_server()
        if self.train_process is not None:
            self.train_process.terminate()
            self.train_process.waitForFinished(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QMainWindow {
            background-color: #eef2f7;
        }

        QTabWidget::pane {
            border: 1px solid #cfcfcf;
            background: white;
            border-radius: 8px;
        }

        QTabBar::tab {
            background: #dfe6ee;
            padding: 10px 18px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }

        QTabBar::tab:selected {
            background: white;
            font-weight: bold;
        }

        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            border: 1px solid #d3d9e3;
            border-radius: 10px;
            margin-top: 12px;
            padding-top: 12px;
            background: #ffffff;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px 0 6px;
        }

        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 6px 12px;
            font-weight: bold;
            min-height: 30px;
        }

        QPushButton:hover {
            background-color: #0b5ed7;
        }

        QPushButton:disabled {
            background-color: #9aa4b2;
            color: #e9ecef;
        }
        
        QPushButton#browseButton {
            background-color: #0fdbad;
            color: white;
        }

        QPushButton#browseButton:hover {
            background-color: #038769;
        }

        QLineEdit {
            border: 1px solid #cfd6df;
            border-radius: 8px;
            padding: 6px;
            background: white;
        }
        QPlainTextEdit {
            border: 1px solid #cfd6df;
            border-radius: 8px;
            padding: 6px;
            background: #111827;
            color: #e5e7eb;
            font-family: Consolas, 'Courier New', monospace;
        }

        QLabel {
            color: #1f2937;
        }
    """)
    window = ThesisApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
