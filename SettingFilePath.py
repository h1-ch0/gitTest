import os

os.chdir(__file__)

def dirbyOS():
    while True:    
        global dirParser
        OperatingSystem = input("what's your OS? (win/mac)")

        if OperatingSystem == 'win':
            dirParser = '\\'
            break
        elif OperatingSystem == 'mac':
            dirParser = '/'
            break
        else:
            print(f"Wrong input : {OperatingSystem} : Try again")
    return dirParser

filepath = f"{dirParser}/Inputs"