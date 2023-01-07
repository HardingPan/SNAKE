import serial
import binascii

ser = serial.Serial('com6', 9600, bytesize=8, stopbits=1, parity='N', timeout=0.5)
a = '1'

ser.write(a.encode("utf-8"))
print(type(a))
