# Converts an excel file to csv format
def excel_to_csv(y):
    import pandas as pd

    df = pd.read_excel(y)

    df.to_csv("output.csv", index=False)