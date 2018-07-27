import yaml


yml_path = './'
yml_file = open(yml_path + "/experiments.yml", 'r')
yml_exp = yaml.load(yml_file)

exp_names = []
commands = {}

for i in yml_exp:
    command = "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python train.py"
    if i["name"] not in exp_names:
        exp_names.append(i["name"])
        for param in i["params"]:
            for key in param:
              command = command + " --" + key + "=" + str(param[key])
        commands[i["name"]] = command
    else:
        raise RuntimeError("Names of the experiments should be unique")


print(commands['centroid-goal'])

