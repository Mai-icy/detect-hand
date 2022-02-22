#!/usr/bin/python
# -*- coding:utf-8 -*-
import serial

PORT_ID = "COM15"


class SerialPort:
    def __init__(self, port_id):
        self.ser = serial.Serial()
        self._port_open(port_id)

    def _port_open(self, port_id):
        """
        初始化串口连接参数
        """
        self.ser.port = port_id  # 设置端口号
        self.ser.baud_rate = 115200  # 设置波特率
        self.ser.byte_size = 8  # 设置数据位
        self.ser.stop_bits = 1  # 设置停止位
        self.ser.parity = "N"  # 设置校验位
        self.ser.open()  # 打开串口,要找到对的串口号才会成功
        if self.ser.isOpen():
            print("串口打开成功")
        else:
            print("打开串口失败")

    def _port_close(self):
        self.ser.close()
        if self.ser.isOpen():
            print("串口关闭失败")
        else:
            print("串口关闭成功")

    def send(self, send_data):
        if self.ser.isOpen():
            self.ser.write(send_data.encode('utf-8'))
            # ser.write(binascii.a2b_hex(send_data))  # Hex发送
            # print("串口发送成功",send_data)
        else:
            print("串口发送失败")


