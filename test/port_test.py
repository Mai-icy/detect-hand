#!/usr/bin/python
# -*- coding:utf-8 -*-
from port.serial_port import SerialPort
import cv2
import sys


PORT_ID = "COM5"  # 串口端口号


if __name__ == "__main__":

    port = SerialPort(PORT_ID)
    while True:
        a = input("?")

        if a == "send":
            port.send("123")




