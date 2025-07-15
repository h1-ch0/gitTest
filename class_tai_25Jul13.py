import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


def getFilepathbyOS()
    if 


class TaiDatabase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)

    def get_data(self):
        return self.data

    def plot_data(self, column_name):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column_name], kde=True)
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.show()