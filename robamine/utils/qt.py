from PyQt5.QtWidgets import QWidget, QSpinBox, QHBoxLayout, QDoubleSpinBox, QAbstractSpinBox, QPushButton, QComboBox, QLabel, QCheckBox, QLineEdit, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QColor, QPalette, QRgba64
from PyQt5.QtCore import Qt
import numpy as np

# Range classes

class QAbstractRange(QWidget):
    def __init__(self, init_value=[0, 0], min=0, max=99, parent=None):
        super(QAbstractRange, self).__init__(parent)
        layout = QHBoxLayout(self)
        layout.addWidget(self.min)
        layout.addWidget(self.max)
        layout.setContentsMargins(0, 0, 0, 0)

        # It is important to set min, max and value before connecting to the
        # callbacks that enforce the constraints of a range
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(init_value)
        self.max.setMinimum(init_value[0])
        self.min.setMaximum(init_value[1])

        self.max.valueChanged.connect(self._max_value_changed)
        self.min.valueChanged.connect(self._min_value_changed)

    def setMinimum(self, value):
        self.min.setMinimum(value)

    def setMaximum(self, value):
        self.max.setMaximum(value)

    def setValue(self, value):
        saved_value = self.min.maximum()
        self.min.setMaximum(value[0])
        self.min.setValue(value[0])
        self.min.setMaximum(saved_value)

        saved_value = self.max.minimum()
        self.max.setMinimum(value[1])
        self.max.setValue(value[1])
        self.max.setMinimum(saved_value)

    def _max_value_changed(self, value):
        self.min.setMaximum(value)

    def _min_value_changed(self, value):
        self.max.setMinimum(value)

    def setSingleStep(self, step):
        self.min.setSingleStep(step)
        self.max.setSingleStep(step)

    def value(self):
        return [self.min.value(), self.max.value()]

class QRange(QAbstractRange):
    def __init__(self, init_value=[0, 0], min=0, max=99, parent=None):
        self.max = QSpinBox()
        self.min = QSpinBox()
        super(QRange, self).__init__(init_value, min, max, parent)

class QDoubleRange(QAbstractRange):
    def __init__(self, init_value=[0.0, 0.0], min=0.0, max=99.0, dec=2, parent=None):
        self.max = QDoubleSpinBox()
        self.min = QDoubleSpinBox()
        self.setDecimals(dec)
        super(QDoubleRange, self).__init__(init_value, min, max, parent)

    def setDecimals(self, dec):
        self.min.setDecimals(dec)
        self.max.setDecimals(dec)

    def setSingleStep(self, step):
        mystep = step
        if mystep < float(1/pow(10, self.min.decimals())):
            mystep = float(1/pow(10, self.min.decimals()))

        if mystep < float(1/pow(10, self.max.decimals())):
            mystep = float(1/pow(10, self.max.decimals()))
        super(QDoubleRange, self).setSingleStep(mystep)

# Vectors classes

class QVector(QWidget):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        super(QVector, self).__init__(parent)
        self.elements = []
        for i in range(size):
            self.elements.append(self.widget_type())
        self.layout = layout_type(self)
        for w in self.elements:
            self.layout.addWidget(w)
        self.layout.setContentsMargins(0, 0, 0, 0)

class QNumericVector(QVector):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.minimum = 0
        self.maximum = 0
        self.single_step = 1
        super(QNumericVector, self).__init__(size, layout_type, parent)

    def setValue(self, value):
        for i in range(len(self.elements)):
            self.elements[i].setValue(value[i])

    def value(self):
        result = []
        for w in self.elements:
            result.append(w.value())
        return result

    def setMinimum(self, value):
        self.minimum = value
        for w in self.elements:
            w.setMinimum(value)

    def setMaximum(self, value):
        self.maximum = value
        for w in self.elements:
            w.setMaximum(value)

    def setSingleStep(self, step):
        self.single_step = step
        for w in self.elements:
            w.setSingleStep(step)

class QVectorX(QVector):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        super(QVectorX, self).__init__(size, layout_type, parent)
        self.buttons_layout = QHBoxLayout()
        self.add_button = QPushButton('+')
        self.add_button.resize(self.add_button.sizeHint().width(), self.add_button.sizeHint().height())
        self.remove_button = QPushButton('-')
        self.buttons_layout.addWidget(self.add_button)
        self.buttons_layout.addWidget(self.remove_button)
        self.layout.addLayout(self.buttons_layout)
        self.add_button.clicked.connect(self._add_element)
        self.remove_button.clicked.connect(self._remove_element)

        self.add_button.setMaximumSize(self.add_button.sizeHint().width()/3, self.add_button.sizeHint().height());
        self.remove_button.setMaximumSize(self.remove_button.sizeHint().width()/3, self.remove_button.sizeHint().height());

        # Add colors to the buttons
        pal = self.add_button.palette()
        pal.setColor(QPalette.Button, QColor(QRgba64.fromRgba(186, 255, 196, 255)));
        self.add_button.setAutoFillBackground(True);
        self.add_button.setPalette(pal);
        self.add_button.update();
        pal = self.remove_button.palette()
        pal.setColor(QPalette.Button, QColor(QRgba64.fromRgba(249, 204, 192, 255)));
        self.remove_button.setAutoFillBackground(True);
        self.remove_button.setPalette(pal);
        self.remove_button.update();

    def _add_element(self):
        self.layout.removeItem(self.buttons_layout)
        self.elements.append(self.widget_type())
        self.layout.addWidget(self.elements[-1])
        self.layout.addLayout(self.buttons_layout)

    def _remove_element(self):
        if len(self.elements) == 0:
            return
        self.layout.removeItem(self.buttons_layout)
        self.layout.removeWidget(self.elements[-1])
        self.elements[-1].setParent(None)
        del self.elements[-1]
        self.layout.addLayout(self.buttons_layout)

class QNumericVectorX(QVectorX, QNumericVector):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        super(QNumericVectorX, self).__init__(size, layout_type, parent)

    def _add_element(self):
        super(QNumericVectorX, self)._add_element()
        self.elements[-1].setMinimum(self.minimum)
        self.elements[-1].setMaximum(self.maximum)
        self.elements[-1].setSingleStep(self.single_step)

class QIntVectorX(QNumericVectorX):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.widget_type = QSpinBox
        super(QIntVectorX, self).__init__(size, layout_type, parent)

class QDoubleVectorX(QNumericVectorX):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.decimals = 3
        self.widget_type = QDoubleSpinBox
        super(QNumericVectorX, self).__init__(size, layout_type, parent)

    def _add_element(self):
        super(QDoubleVectorX, self)._add_element()
        self.elements[-1].setDecimals(self.decimals)

    def setDecimals(self, dec):
        self.decimals = dec
        for w in self.elements:
          w.setDecimals(dec)

class QIntVector(QNumericVector):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.widget_type = QSpinBox
        super(QIntVector, self).__init__(size, layout_type, parent)

class QDoubleVector(QNumericVector):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.widget_type = QDoubleSpinBox
        super(QDoubleVector, self).__init__(size, layout_type, parent)

    def setDecimals(self, dec):
        for w in self.elements:
          w.setDecimals(dec)

class QComboVectorX(QVectorX):
    def __init__(self, size = 1, layout_type = QHBoxLayout, parent=None):
        self.widget_type = QComboBox
        self.items = []
        super(QComboVectorX, self).__init__(size, layout_type, parent)

    def setValue(self, value):
        for i in range(len(self.elements)):
            self.elements[i].setCurrentIndex(self.elements[i].findText(value[i]))

    def value(self):
        result = []
        for w in self.elements:
            result.append(w.currentText())
        return result

    def addItems(self, items):
        self.items = items
        for i in range(len(self.elements)):
            self.elements[i].addItems(items)

    def _add_element(self):
        super(QComboVectorX, self)._add_element()
        self.elements[-1].addItems(self.items)

# Matrix classes

class QIntMatrixX(QNumericVectorX):
    def __init__(self, size = [1], parent=None):
        self.widget_type = QIntVectorX
        super(QIntMatrixX, self).__init__(len(size), QVBoxLayout, parent)

        diff = np.array(size) - np.ones(len(size))
        for i in range(len(diff)):
            if diff[i] > 0:
                for j in range(int(diff[i])):
                    self.elements[i]._add_element()

class QDoubleMatrixX(QNumericVectorX):
    def __init__(self, size = [1], parent=None):
        self.widget_type = QDoubleVectorX
        self.decimals = 2
        super(QDoubleMatrixX, self).__init__(size, QVBoxLayout, parent)

    def _add_element(self):
        super(QDoubleMatrixX, self)._add_element()
        self.elements[-1].setDecimals(self.decimals)

    def setDecimals(self, dec):
        self.decimals = dec
        for w in self.elements:
          w.setDecimals(dec)

# Transformation functions

def form2dict(layout):
    result = {}
    i = 0
    while i < layout.count():
        label = layout.itemAt(i).widget().text()
        if isinstance(layout.itemAt(i + 1).widget(), QCheckBox):
            widget = layout.itemAt(i + 1).widget().isChecked()
        elif isinstance(layout.itemAt(i + 1).widget(), QComboBox):
            widget = layout.itemAt(i + 1).widget().currentText()
        elif isinstance(layout.itemAt(i + 1).widget(), QLineEdit):
            widget = layout.itemAt(i + 1).widget().text()
        else:
            widget = layout.itemAt(i + 1).widget().value()
        i += 2
        result[label] = widget
    return result

def dict2form(values, constraints, layout):
    # inputs = self.env_defaults[self.env_name.currentText()]
    # constraints = self.env_constraints[self.env_name.currentText()]

    for i in reversed(range(layout.rowCount())):
        layout.removeRow(i)

    for key, value in sorted(values.items()):
        con = constraints[key]
        label = QLabel(key)
        if 'help' in con:
            label.setToolTip(con['help'])

        if con['type'] == 'int':
            w = QSpinBox()
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setValue(value)
        elif con['type'] == 'float':
            w = QDoubleSpinBox()
            w.setDecimals(con['decimals'])
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setValue(value)
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
        elif con['type'] == 'intrange':
            w = QRange(init_value=value, min=con['range'][0], max=con['range'][1])
        elif con['type'] == 'floatrange':
            w = QDoubleRange(init_value=value, min=con['range'][0], max=con['range'][1], dec=con['decimals'])
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
        elif con['type'] == 'floatvector':
            w = QDoubleVector(len(value))
            w.setDecimals(con['decimals'])
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
            w.setValue(value)
        elif con['type'] == 'floatvectorx':
            w = QDoubleVectorX(len(value))
            w.setDecimals(con['decimals'])
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
            w.setValue(value)
        elif con['type'] == 'floatmatrixx':
            w = QDoubleMatrixX([len(i) for i in value])
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
            w.setValue(value)
        elif con['type'] == 'intmatrixx':
            w = QIntMatrixX([len(i) for i in value])
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setSingleStep((con['range'][1] - con['range'][0])/100)
            w.setValue(value)
        elif con['type'] == 'intvectorx':
            w = QIntVectorX(len(value))
            w.setMinimum(con['range'][0])
            w.setMaximum(con['range'][1])
            w.setValue(value)
        elif con['type'] == 'combovectorx':
            w = QComboVectorX(len(value))
            w.addItems(con['options'])
            w.setValue(value)
        elif con['type'] == 'combo':
            w = QComboBox()
            w.addItems(con['options'])
            w.setCurrentIndex(w.findText(value))
        elif con['type'] == 'bool':
            w = QCheckBox()
            w.setChecked(value)
        elif con['type'] == 'str':
            w = QLineEdit()
            w.setText(str(value))
        else:
            w = QLineEdit()
            w.setText(str(value))

        layout.addRow(label, w)
