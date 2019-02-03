#coding: UTF-8 -*-
#----------------------------------------------------
#README
#各銘柄を日足→ n 足に変換するスクリプト
#ディレクトリの構造の前提として、
#split_by_companyYEAR
#|
#|-daily
#  |
#  |--FILE1
#  |--FILE2
#  ...
#  |--FILEn
#になっていてほしい。大文字は部分は自由

import pathlib
import os
import sys
years = ['2016', '2017', '2018']

if ((len(sys.argv) != 3) or (not sys.argv[1].isdecimal())):
	print("Argument : 1.Period 2.Directory")
	sys.exit()
convert = int(sys.argv[1])
convert_dirname = sys.argv[2]

for y in years:
	dir_list = os.listdir("split_by_company" + y + "/daily")
	pathlib.Path("split_by_company" + y + "/" + convert_dirname).mkdir(exist_ok=True)
	for d in dir_list:
		filename = "split_by_company" + y + "/daily/" + d
		f = open(filename)
		num_line = sum([1 for i in open(filename,'r')])
		row_data = f.readline()
		edit_file = "split_by_company" + y + "/" + convert_dirname + "/" + d
		#print("edit_file = ", edit_file)
		if (not os.path.isfile(edit_file)) :
			with open(edit_file, mode = 'w') as d:
				d.write(row_data)
		for i in range(int(num_line / convert) + 1):	
			HIGH = 0
			LOW = 100000000
			VOLUME = 0
			for j in range(convert):
				#value_date, stock_code,opne,close,HIGH,LOW,VOLUME
				lines = f.readline()
				if not lines:
					break
				lines = lines[:-1]
				lines = lines.split(",")
				if j == 0:
					value_date = lines[0]
					stock_code = lines[1]
					open_value = lines[2]
				if j == convert - 1:
					close_value = lines[3]
				HIGH = max(int(lines[4]),HIGH)
				LOW = min(int(lines[5]),LOW)
				VOLUME = VOLUME + int(lines[6])
			if VOLUME != 0:
				converted_line = value_date + "," + str(stock_code) + "," + str(open_value) + "," +  str(close_value) + "," + str(HIGH) + "," + str(LOW) + "," + str(VOLUME) + "\n"
				with open(edit_file, mode = 'a') as d:
					d.write(converted_line)
		f.close()
