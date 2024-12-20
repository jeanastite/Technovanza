import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

class SimpleApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the main window properties
        self.setWindowTitle("Simple PyQt5 App")
        self.setGeometry(100, 100, 400, 300)

        # Create a button
        self.button = QPushButton("Click Me", self)
        self.button.setGeometry(150, 130, 100, 40)
        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        # Show a message box when the button is clicked
        QMessageBox.information(self, "Message", "Hello, PyQt5!")

# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SimpleApp()
    main_window.show()
    sys.exit(app.exec_())
