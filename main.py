import sys
import faulthandler
from PyQt5.QtWidgets import QApplication

from CellAnalysis.CellAnalysis import InvadopodiaGui

import logging
fmt = "[%(asctime)s] [%(levelname)s] [%(funcName)s(): line %(lineno)s] [PID:%(process)d TID:%(thread)d] %(message)s"
date_fmt = "%d/%m/%Y %H:%M:%S"
logging.basicConfig(format=fmt, datefmt=date_fmt, filename='debug.log', level=logging.DEBUG)

APP_NAME = "InvadopodiaGui"
APP_VERSION = "0.2"


# todo: soft-code colors and parameters
# todo: put some GodDamn logging info

# OPTIMIZATION
# todo: lazy import (do I really?)

# KNOWN BUGS
# todo: if markers are present when ROI is moved and they fall outside ROI, software behaviour is undefined

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    class TestApp(QApplication):
        def __init__(self, argv):
            super(TestApp, self).__init__(argv)

            self.setApplicationName(APP_NAME)
            self.setApplicationVersion(APP_VERSION)

    with open('traceback_dump.txt', 'w+') as dump_file:
        faulthandler.enable(file=dump_file)
        app = TestApp(sys.argv)

        gui = InvadopodiaGui()
        gui.show()

        sys.exit(app.exec_())
