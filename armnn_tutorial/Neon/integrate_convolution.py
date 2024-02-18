import json
import os
import sys
import argparse

def fil(x): 
    if ".DS_" in x:
        return False
    else:
        return True

def load_file(path, file: str):
    print(path + file)
    
    with open(path + file, 'r') as f:
        return json.load(f)

def load(file: str):
    return list(map(lambda x: load_file(file + '/', x), filter(fil, os.listdir(file))))

def layer_integrate(item):
    result = {}
    tmp_layer = []
    tmp_key = '0'
    for data in list(item.items()):
        key, name, value, score = data[0].split('_')[0], data[0], data[1]['data'], data[1]['score']
        if tmp_key != key and tmp_key != '0':
            result[tmp_key] = tmp_layer
            tmp_layer = []
            tmp_layer.append({'name': name, 'data': data[1]})
            tmp_key = key
        else:
            tmp_key = key
            tmp_layer.append({'name': name, 'data': data[1]})
    result[tmp_key] = tmp_layer
    return result

def get_layer_score(item):
    # print(item)
    # key, items = item[0], item[1]
    scores = list(map(lambda x: x['data']['score'], item))
    # print(scores)
    return sum(scores) 

def string_array_contain(data, text):
    for item in data:
        if text in item:
            return True
    return False

def get_kernel_name(data):
    for item in data:
        if "CpuGemmAssemblyWrapperKernel" in item:
            text = len('_NeonConvolution2dWorkload_Execute')
            return item.split('/')[1][:-text].strip('_')
    return ''

def extend_opt_score(opt):
    key = list(opt.keys())
    for k in key:
        # print(opt[k][0]['name'])
        data = list(map(lambda y: y['name'], opt[k]))
        # print(data)
        is_conv = string_array_contain(data, 'NeonConvolution2dWorkload')
        is_fully = string_array_contain(data, 'NeonFullyConnectedWorkload')
        is_wino = string_array_contain(data, 'Winograd')
        is_general = string_array_contain(data, 'Im2Col')
        is_direct = is_wino == False and is_general == False
        
        method = -1

        # print(is_conv)
        if is_direct == True:
            method = 1
        if is_general == True:
            method = 2
        if is_wino == True:
            method = 3
        # print(data)
        # if is_conv == True:
        kernel_name = get_kernel_name(data)
        elem = {'data': opt[k], 'score': get_layer_score(opt[k]), 
                'is_conv': is_conv, 
                'is_fully': is_fully,
                'method': method,
                'kernel': kernel_name}
        opt[k] = elem
    return opt

def conv_to_best(opts, keys):
    opt_size = len(opts)
    layer_size = len(opts[0])
    result = {}
    best_score = 0
    for ls in range(layer_size):
        tmp_score = sys.maxsize
        tmp_layer = None

        for ops in range(opt_size):
            score = opts[ops][keys[ls]]['score']

            if not opts[ops][keys[ls]]['is_conv']:
                tmp_score = score
                tmp_layer = opts[ops][keys[ls]]

            if tmp_score > score:
                tmp_score = score
                tmp_layer = opts[ops][keys[ls]]

        result[keys[ls]] = tmp_layer
        best_score += tmp_score
    return (result, best_score)

def conv_optimize(result):
    items = list(map(lambda x: str(x), sorted(list(map(lambda x: int(x), result.keys())))))
    
    conv_list = []
    fully_list = []
    
    for key in items:
        for kernel in result[str(key)]['data']:
            # print(key)
            name = kernel['name']
            data = kernel['data']
            result[key][name] = data
        del result[key]['data']
        if result[key]['is_conv']:
            conv_list.append({'name': key, 
                              'method': result[key]['method'], 
                              'kernel': result[key]['kernel']})
        if result[key]['is_fully']:
            fully_list.append({'name': key, 
                              'kernel': result[key]['kernel']})
    result['convolution'] = conv_list
    result['fully'] = fully_list

# model = ["alexnet", "googlenet", "mobilenet_v2", "resnet50", "resnet101", "vgg16"]
# python3 integrate_convolution.py -m ${model} -t ./tune/${model} -o ./${model}-for-eval.json

parser = argparse.ArgumentParser('tflite to rknn')
parser.add_argument('-t', dest='tune', type=str, required=True)
parser.add_argument('-o', dest='output', type=str, required=True)

args = parser.parse_args()

tune = args.tune
output = args.output
    
# for m in model:
    # for mo in mode:
data = load(tune)
opts = list(map(lambda x: layer_integrate(x), data))
# get_layer_score(list(opts[7].items())[0])

extend_opts = list(map(lambda x: extend_opt_score(x.copy()), opts))
# extend_opt_score(opts[12].copy())
print(extend_opts[0])
# list(map(lambda x: extend_opt_score(x.copy()), opts))

keys = list(map(lambda x: str(x), sorted(list(map(lambda x: int(x), extend_opts[0].keys())))))
result, best_score = conv_to_best(extend_opts, keys)
print(best_score / 1000.0 / 1000.0)
conv_optimize(result)

with open(output, 'w', encoding='utf-8') as file:
    json.dump(result, file, indent='  ')

def f(x):
    if "convol" in x or "fully" in x:
        return False
    else:
        return True
    
k = list(result.keys())
k = list(filter(f, k))

print(result[k[0]]['score'])

e = list(map(lambda x: result[x]['score'], k))
a = sum(list())
print(a)
print(best_score / 1000.0 / 1000.0)


#%%



#%%