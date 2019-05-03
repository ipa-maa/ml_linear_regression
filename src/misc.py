"""
Created on May 3, 2019

@author: maa
@attention: miscellanious classes for ml test
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

class Font:
    """
    terminal color codes
    usage: color_code + text + color_end
    """
    backgroundblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    yellow = '\033[93m'
    pink = '\033[95m'
    lightcyan = '\033[96m'
    lightblue = '\033[94m'
    lightgreen = '\033[92m'
    lightred = '\033[91m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    end = '\033[0m'


def printy(text: str) -> None:
    """
    print text in yellow terminal color
    """
    print(Font.yellow + text + Font.end)
