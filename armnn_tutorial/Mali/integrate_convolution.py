import os
import sys
import argparse
import pandas as pd


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--log')           # positional argument
parser.add_argument('--tune')      # option that takes a value
args = parser.parse_args()

dir = args.log
l = os.listdir(dir)
item = filter(lambda x: not "default" in x, map(lambda x: dir + "/" + x, l))

pds = list(map(lambda x: (pd.read_csv(x).mean(axis=0), x.split(".")[-2].split("-")[-1]), item))

result = []
mode_result = [] 
for i in range(len(pds[0][0])):
    result.append(sys.maxsize)
    mode_result.append(-1)

for mode in pds:
    for i in range(len(mode[0])):
        if result[i] > mode[0][i]:
            result[i] = mode[0][i]
            mode_result[i] = mode[1]
            
with open(args.tune, 'w') as f:
    for item in mode_result:
        # a = "asd"
        # item = "".join(reversed(item))
        f.write(item + " ");
    # f.write("\n");
    # for item in result:
    #     f.write(str(item) + " ");
# for i in range(len(mode_result)):
#     print(mode_result[i], result[i])