import yaml

def get_yml(path, name):
    yml_file = open(path + "/" + name + ".yml", 'r')
    yml = yaml.load(yml_file)
    return yml

def get_exp_names_commands(yml):
    exp_names = []
    commands = {}

    for i in yml:
        command = "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python train.py"
        if i["name"] not in exp_names:
            exp_names.append(i["name"])
            for param in i["params"]:
                for key in param:
                  command = command + " --" + key + "=" + str(param[key])
            commands[i["name"]] = command
        else:
            raise RuntimeError("Names of the experiments should be unique")
    return exp_names, commands


if __name__ == '__main__':
    yml = get_yml("./", "experiments")
    names, commands = get_exp_names_commands(yml)
    print(commands['centroid-goal'])
