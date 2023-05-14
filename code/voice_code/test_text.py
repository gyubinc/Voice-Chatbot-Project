import pandas as pd
import os
#from openpyxl import Workbook
#from openpyxl.styles import NamedStyle
import warnings
warnings.simplefilter("ignore", UserWarning)
#wb = Workbook()
#default_style = NamedStyle(name="default")
#wb.default_style = default_style


path = './data/'
file_list = os.listdir(path)
file_list_excel = [file for file in file_list if file.endswith('.xlsx')]

df = pd.DataFrame()


for i in file_list_excel:
    data = pd.read_excel(path + i)
    breakpoint()
    df = pd.concat([df,data])
    breakpoint()
    
