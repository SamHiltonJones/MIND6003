import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QDialog
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64

class SettingsPublisher(Node):
    def __init__(self):
        super().__init__('settings_publisher')
        self.publisher_ = self.create_publisher(Int64, '/settings', 10)
        self.launch_settings_dialog()

    def launch_settings_dialog(self):
        self.app = QApplication(sys.argv)
        self.dialog = QDialog()
        self.layout = QVBoxLayout()

        self.button1 = QPushButton('Detect Added Updates to Room')
        self.button1.clicked.connect(lambda: self.publish_setting(1))
        self.button2 = QPushButton('Detect All Changes')
        self.button2.clicked.connect(lambda: self.publish_setting(2))
        self.button3 = QPushButton('Removal of Objects')
        self.button3.clicked.connect(lambda: self.publish_setting(3))

        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        
        self.dialog.setLayout(self.layout)
        self.dialog.exec_()

    def publish_setting(self, setting_value):
        msg = Int64()
        msg.data = setting_value
        self.publisher_.publish(msg)
        self.dialog.accept()

def main(args=None):
    rclpy.init(args=args)
    settings_publisher = SettingsPublisher()
    rclpy.spin(settings_publisher)
    settings_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
