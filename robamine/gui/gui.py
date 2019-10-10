# Qt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QFormLayout, QLabel, QLineEdit, qApp, QSpinBox, QCheckBox, QDoubleSpinBox, QComboBox, QDialog
from PyQt5 import uic
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QUrl, Qt, QDateTime
from PyQt5.QtGui import QIcon, QDesktopServices, QPixmap
from robamine.utils.qt import QRange, QDoubleRange, QDoubleVector, QDoubleVectorX, dict2form, form2dict

# Robamine
from robamine import rb_logging
from robamine.algo.core import TrainWorld, EvalWorld, TrainEvalWorld, DataAcquisitionWorld, WorldState
from robamine.utils.info import get_pc_and_version, start_tensorboard_server, bytes2human

# General
import yaml
import os
from enum import Enum
import sys
import time
import logging
import threading

import importlib

# Paths to yaml files
path_to_yamls = os.path.join(os.path.dirname(__file__), '../../yaml/')
path_to_defaults = os.path.join(path_to_yamls, 'defaults')
path_to_constraints = os.path.join(path_to_yamls, 'constraints')

class RobamineApp(QObject):
    """
    The object that starts a robamine session, i.e. creates a world and run
    episodes, while it communicates with the GUI.
    """
    update_progress = pyqtSignal(dict)
    started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, params):
        super(RobamineApp, self).__init__()
        self.params = params.copy()
        rb_logging.init(directory=params['world']['logging_dir'], file_level=logging.INFO)

        # Load the desired world type
        module = importlib.import_module('robamine.algo.core')
        world_handle = getattr(module, self.params['world']['name']+'World')
        self.world = world_handle.from_dict(self.params)

        self.termination_flag = False

    @pyqtSlot()
    def run(self):
        self.tensorboard_url = start_tensorboard_server(self.world.log_dir)
        world_thread = threading.Thread(target=self.world.run, args=())
        world_thread.start()
        self.started.emit()

        while self.world.get_state() != WorldState.FINISHED:
            self.world.results_lock.acquire()
            results_copy = self.world.config['results'].copy()
            self.world.results_lock.release()
            self.update_progress.emit(results_copy)
            self.world.stop_running = self.termination_flag
            time.sleep(1.0)

        # Update progress for the last time
        self.world.results_lock.acquire()
        results_copy = self.world.config['results'].copy()
        self.world.results_lock.release()
        self.update_progress.emit(results_copy)

        self.finished.emit()

    def terminate(self):
        self.termination_flag = True

class Mode(Enum):
    NONE = 0
    CONFIG = 1
    STARTING = 2
    RUNNING = 3
    STOPPING = 4
    FINISHED = 5

def yaml2dict(path, list_of_names):
    result = {}
    for name in list_of_names:
        with open(os.path.join(path, name.lower() + '.yml'), 'r') as stream:
            try:
                result[name] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    return result

class RobamineGUI(QMainWindow):
    """
    The main window of the Robamine GUI.
    """

    def __init__(self, parent=None):
        super(RobamineGUI, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'gui.ui'), self)

        # Placeholders
        self.robamine_app, self.robamine_thread = None, None
        self.scheduler_app, self.scheduler_thread = None, None
        self.state = {}
        self.state['results'] = {}

        # Init Mode
        self.current_mode = Mode.NONE
        self.set_mode(Mode.CONFIG)

        # Load default values for available envs and agents
        with open(os.path.join(path_to_yamls, 'available.yml'), 'r') as stream:
            try:
                self.available = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.env_name.addItems(self.available['env'])
        self.agent_name.addItems(self.available['agent'])
        self.world_name.addItems(self.available['world'])

        self.env_defaults = yaml2dict(os.path.join(path_to_defaults, 'env'), self.available['env'])
        self.agent_defaults = yaml2dict(os.path.join(path_to_defaults, 'agent'), self.available['agent'])
        self.world_defaults = yaml2dict(os.path.join(path_to_defaults, 'world'), self.available['world'])
        self.env_stored, self.agent_stored, self.world_stored = self.env_defaults.copy(), self.agent_defaults.copy(), self.world_defaults.copy()

        self.env_constraints = yaml2dict(os.path.join(path_to_constraints, 'env'), self.available['env'])
        self.agent_constraints = yaml2dict(os.path.join(path_to_constraints, 'agent'), self.available['agent'])
        self.world_constraints = yaml2dict(os.path.join(path_to_constraints, 'world'), self.available['world'])

        self.env_name_prev = self.env_name.currentText()
        self.agent_name_prev = self.agent_name.currentText()
        self.world_name_prev = self.world_name.currentText()

        # Load state
        with open(os.path.join(path_to_defaults, 'world.yml'), 'r') as stream:
            try:
                self.state = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.state['world'] = self.world_defaults[self.state['world']['name']].copy()
        self.state['env'] = self.env_defaults[self.state['env']['name']].copy()
        self.state['agent'] = self.agent_defaults[self.state['agent']['name']].copy()

        self.state2gui()

        self.scheduled_time = None

        # Connect callbacks to buttons
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.save_as_action.triggered.connect(self.save_as)
        self.load_yaml_action.triggered.connect(self.load_yaml)
        self.quit_action.triggered.connect(self.closeEvent)
        self.browse_button.clicked.connect(self.browse_button_cb)
        self.env_name.activated.connect(self.env_name_cb)
        self.agent_name.activated.connect(self.agent_name_cb)
        self.world_name.activated.connect(self.world_name_cb)
        self.progress_open_dir_button.clicked.connect(lambda : os.system('xdg-open "%s"' % self.progress_logging_dir.text()))
        self.env_reset_defaults_button.clicked.connect(self.env_reset_defaults_button_cb)
        self.agent_reset_defaults_button.clicked.connect(self.agent_reset_defaults_button_cb)
        self.tensorboard_button.clicked.connect(lambda : QDesktopServices.openUrl(QUrl(self.robamine_app.tensorboard_url)))
        self.about_action.triggered.connect(self.about_action_cb)
        self.progress_details_button.clicked.connect(self.progress_details_button_cb)
        self.schedule_action.triggered.connect(self.schedule_action_cb)

        self.hostname, self.username, self.version = get_pc_and_version()

        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'fig/logo.png')))

    def set_mode(self, mode):
        if mode == Mode.CONFIG:
            self.progress_group.setDisabled(True)
            self.world_group.setDisabled(False)
            self.environment_group.setDisabled(False)
            self.agent_group.setDisabled(False)
            self.start_button.setDisabled(False)
            self.stop_button.setDisabled(True)
            self.status_label.setText('CONFIGURING')

        elif mode == Mode.FINISHED:
            self.progress_group.setDisabled(False)
            self.world_group.setDisabled(True)
            self.environment_group.setDisabled(True)
            self.agent_group.setDisabled(True)
            self.start_button.setDisabled(True)
            self.stop_button.setDisabled(True)
            self.status_label.setText('FINISHED.')

        elif mode == Mode.STOPPING:
            self.robamine_app.terminate()
            self.stop_button.setDisabled(True)
            self.status_label.setText('STOPPING...')

        elif mode == Mode.STARTING:
            self.world_group.setDisabled(True)
            self.environment_group.setDisabled(True)
            self.agent_group.setDisabled(True)
            self.start_button.setDisabled(True)
            self.status_label.setText('STARTING...')

        elif mode == Mode.RUNNING:
            self.progress_group.setDisabled(False)
            self.stop_button.setDisabled(False)
            self.status_label.setText('RUNNING...')

        else:
            raise ValueError('Mode does not exist.')

        self.current_mode = mode

    def update_progress(self, results):
        self.progress_logging_dir.setText(str(results['logging_dir']))
        possible_steps = ['episodes', 'timesteps', 'epochs']
        for step in possible_steps:
            if step in self.state['world']['params']:
                break;
        progress = results['n_' + step] / self.state['world']['params'][step]

        self.progressBar.setValue(progress * 100)
        self.state['results'] = results

    # Transformations between GUI and dictionaries

    def gui2state(self):
        self.state['world'] = self.gui2world()
        self.state['env'] = self.gui2env()
        self.state['agent'] = self.gui2agent()

    def gui2world(self):
        world = {}
        world['name'] = self.world_name.currentText()
        world['logging_dir'] = self.logging_dir_input.text()
        world['comments'] = self.comments.toPlainText()
        if 'params' in self.world_defaults[world['name']]:
            world['params'] = form2dict(self.world_params_layout)
        return world

    def gui2env(self):
        env = {}
        env['name'] = self.env_name.currentText()
        if 'params' in self.env_defaults[env['name']]:
            env['params'] = form2dict(self.env_params_layout)
        return env

    def gui2agent(self):
        agent = {}
        agent['name'] = self.agent_name.currentText()
        if 'params' in self.agent_defaults[agent['name']]:
            agent['params'] = form2dict(self.agent_params_layout)
        if 'trainable_params' in self.agent_defaults[agent['name']]:
            agent['trainable_params'] = self.trainable_input.text()
        return agent

    def state2gui(self):
        # Update World group
        self.world2gui(self.state['world'])
        self.env2gui(self.state['env'])
        self.agent2gui(self.state['agent'])

    def world2gui(self, world):
        if world['name'] not in self.available['world']:
            raise ValueError('The world name: ' + world['name'] + ' is not supported.')
        self.logging_dir_input.setText(self.state['world']['logging_dir'])
        self.comments.setText(self.state['world']['comments'])

        self.world_name.setCurrentIndex(self.world_name.findText(world['name']))
        if 'params' in world:
            dict2form(world['params'], self.world_constraints[world['name']], self.world_params_layout)
        else:
            for i in reversed(range(self.world_params_layout.rowCount())):
                self.world_params_layout.removeRow(i)

    def env2gui(self, env):
        if env['name'] not in self.available['env']:
            raise ValueError('The env name: ' + env['name'] + ' is not supported.')

        self.env_name.setCurrentIndex(self.env_name.findText(env['name']))
        if 'params' in env:
            dict2form(env['params'], self.env_constraints[env['name']], self.env_params_layout)
        else:
            for i in reversed(range(self.env_params_layout.rowCount())):
                self.env_params_layout.removeRow(i)

    def agent2gui(self, agent):
        if agent['name'] not in self.available['agent']:
            raise ValueError('The agent name: ' + agent['name'] + ' is not supported.')

        self.agent_name.setCurrentIndex(self.agent_name.findText(agent['name']))

        if 'params' in agent:
            dict2form(agent['params'], self.agent_constraints[agent['name']], self.agent_params_layout)
        else:
            for i in reversed(range(self.agent_params_layout.rowCount())):
                self.agent_params_layout.removeRow(i)

        if 'trainable_params' in agent:
            self.trainable_input.setText(agent['trainable_params'])
            self.trainable_input.setDisabled(False)
        else:
            self.trainable_input.setText('')
            self.trainable_input.setDisabled(True)

    # Callbacks for various widgets of the GUI (button, inputs etc)

    def start(self):
        self.gui2state()
        self.robamine_app = RobamineApp(self.state)
        self.robamine_thread = QThread()

        self.robamine_app.update_progress.connect(self.update_progress)
        self.robamine_app.finished.connect(lambda : self.set_mode(Mode.FINISHED))
        self.robamine_app.started.connect(lambda : self.set_mode(Mode.RUNNING))
        self.robamine_app.moveToThread(self.robamine_thread)
        self.robamine_thread.start()
        self.robamine_thread.started.connect(self.robamine_app.run)
        self.set_mode(Mode.STARTING)

    def stop(self):
        dialog = AreYouSure('Are you sure you want to stop the experiment?')
        dialog.accepted.connect(lambda : self.set_mode(Mode.STOPPING))
        dialog.exec_()
        dialog.show()

    def save_as(self):
        self.gui2state()
        options = QFileDialog.Options()
        name, _ = QFileDialog.getSaveFileName(self, 'Save File', "","YAML Files (*.yml)", options=options)
        if name != '':
            with open(name, 'w') as outfile:
                yaml.dump(self.state, outfile, default_flow_style=False)

    def closeEvent(self, event):
        if self.current_mode == Mode.RUNNING or self.current_mode == Mode.STOPPING:
            event.ignore()
            dialog = AreYouSure('The experiment is still running. Are you sure you want to quit?')
            dialog.accepted.connect(lambda : event.accept())
            dialog.exec_()
            dialog.show()

    def load_yaml(self):
        options = QFileDialog.Options()
        name, _ = QFileDialog.getOpenFileName(self,"Load YAML file", "","YAML Files (*.yml)", options=options)
        if name != '':
            with open(name, 'r') as stream:
                self.state = yaml.safe_load(stream)

        if 'results' in self.state and 'logging_dir' in self.state['results']:
            self.state['agent']['trainable_params'] = os.path.join(self.state['results']['logging_dir'], 'model.pkl')

        self.state2gui()

        self.world_name_prev = self.world_name.currentText()
        self.env_name_prev = self.env_name.currentText()
        self.agent_name_prev = self.agent_name.currentText()

    def browse_button_cb(self):
        new = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if new != '':
            self.logging_dir_input.setText(new)

    def world_name_cb(self):
        desired = self.world_name.currentText()
        self.world_name.setCurrentIndex(self.world_name.findText(self.world_name_prev))
        self.world_stored[self.world_name_prev] = self.gui2world().copy()
        self.world2gui(self.world_stored[desired])
        self.world_name_prev = desired

    def env_name_cb(self):
        desired = self.env_name.currentText()
        self.env_name.setCurrentIndex(self.env_name.findText(self.env_name_prev))
        self.env_stored[self.env_name_prev] = self.gui2env().copy()
        self.env2gui(self.env_stored[desired])
        self.env_name_prev = desired

    def agent_name_cb(self):
        desired = self.agent_name.currentText()
        self.agent_name.setCurrentIndex(self.agent_name.findText(self.agent_name_prev))
        self.agent_stored[self.agent_name_prev] = self.gui2agent().copy()
        self.agent2gui(self.agent_stored[desired])
        self.agent_name_prev = desired

    def env_reset_defaults_button_cb(self):
        self.env_stored[self.env_name.currentText()] = self.env_defaults[self.env_name.currentText()].copy()
        dict2form(self.env_stored[self.env_name.currentText()], self.env_constraints[self.env_name.currentText()], self.env_params_layout)

    def agent_reset_defaults_button_cb(self):
        self.agent_stored[self.agent_name.currentText()] = self.agent_defaults[self.agent_name.currentText()].copy()
        dict2form(self.agent_stored[self.agent_name.currentText()], self.agent_constraints[self.agent_name.currentText()], self.agent_params_layout)

    def about_action_cb(self):
        dialog = AboutDialog(self.version)
        dialog.exec_()
        dialog.show()

    def progress_details_button_cb(self):
        dialog = ProgressDetailsDialog(self.robamine_app)
        dialog.update(self.state['results'])
        dialog.exec_()
        dialog.show()

    def schedule_action_cb(self):
        if self.current_mode == Mode.CONFIG:
            dialog = SchedulerDialog()
            self.scheduled_time = dialog.getDateTime()
            if self.scheduled_time is not None:
                self.scheduled_time_label.setText('Scheduled for: ' + self.scheduled_time.toString())
                self.scheduler_app = SchedulerApp(self.scheduled_time)
                self.scheduler_thread = QThread()
                self.scheduler_app.time_is_up.connect(self.start)
                self.scheduler_app.moveToThread(self.scheduler_thread)
                self.scheduler_thread.start()
                self.scheduler_thread.started.connect(self.scheduler_app.run)
            else:
                self.scheduled_time_label.setText('')
                del self.scheduler_app, self.scheduler_thread
                self.scheduler_app, self.scheduler_thread = None, None
        else:
            print('Scheduler is available only in CONFIGURING mode.')

# Qt Dialogs

class ProgressDetailsDialog(QDialog):
    def __init__(self, robamine_app, parent=None):
        super(ProgressDetailsDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'progress_details.ui'), self)
        robamine_app.update_progress.connect(self.update)

    def update(self, results):
        self.episodes.setText(str(results['n_episodes']))
        self.timesteps.setText(str(results['n_timesteps']))
        self.started_on.setText(str(results['started_on']))
        self.time_elapsed.setText(str(results['time_elapsed']))
        self.time_remaining.setText(str(results['estimated_time']))
        self.machine.setText(str(results['hostname']))
        self.version.setText(str(results['version']))
        self.dir_size.setText(bytes2human(results['dir_size']))

class SchedulerDialog(QDialog):
    def __init__(self, init_state=None, parent=None):
        super(SchedulerDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'scheduler.ui'), self)
        self.datetime_input.setDateTime(QDateTime.currentDateTime())
        self.datetime_input.setDisabled(not self.check_box.isChecked())
        self.check_box.stateChanged.connect(self.check_box_cb)
        self.accepted.connect(self.schedule_action_accepted_cb)
        self.datetime = None

    def check_box_cb(self, state):
        self.datetime_input.setDisabled(not state)

    def getDateTime(self):
        self.exec_()
        self.show()
        return self.datetime

    def schedule_action_accepted_cb(self):
        if self.check_box.isChecked():
            self.datetime = self.datetime_input.dateTime()
        else:
            self.datetime = None

class SchedulerApp(QObject):
    time_is_up = pyqtSignal()

    def __init__(self, scheduled_time):
        super(SchedulerApp, self).__init__()
        self.scheduled_time = scheduled_time

    @pyqtSlot()
    def run(self):
        while(True):
            current_time = QDateTime.currentDateTime()
            if current_time > self.scheduled_time:
                self.time_is_up.emit()
                break
            time.sleep(5.0)

class AboutDialog(QDialog):
    def __init__(self, version, parent=None):
        super(AboutDialog, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'about.ui'), self)
        self.logo_pixmap = QPixmap(os.path.join(os.path.dirname(__file__), 'fig/logo_bw.png'))
        self.logo_pixmap = self.logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio)
        self.logo_label.setPixmap(self.logo_pixmap)
        self.logo_label.resize(self.width(), self.height())
        self.version_label.setText(version)

class AreYouSure(QDialog):
    def __init__(self, msg, parent=None):
        super(AreYouSure, self).__init__(parent)
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'sure_stop.ui'), self)
        self.label.setText(msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RobamineGUI()
    gui.show()
    sys.exit(app.exec_())
