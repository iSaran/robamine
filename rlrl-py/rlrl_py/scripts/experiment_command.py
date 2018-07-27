import yaml

def get_yml(path, name):
    print('\033[93m' + "Warning: This code assumes that you have already run yamllint on your yaml file to ensure its proper syntax." + '\033[0m')
    yml_file = open(path + "/" + name + ".yml", 'r')
    yml = yaml.load(yml_file)
    return yml

def get_exp_names_commands(yml):
    exp_names = []
    names_completed = []
    commands = {}

    for experiment in yml:
        command = "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python train.py"
        exp_names.append(experiment)
        if yml[experiment]["completed"]:
            names_completed.append(experiment)

        for param in yml[experiment]["params"]:
              command = command + " --" + param + "=" + str(yml[experiment]["params"][param])
        command = command + " --logdir=" + yml[experiment]["dir"] + '/' + experiment
        commands[experiment] = command
    return exp_names, names_completed, commands

def get_param(yml, name, param):
    for i in yml:
        if i["name"] == name:
            for j in i["params"]:
                for key in j:
                    if key == param:
                        return j[key]
    raise RuntimeError("Given name or param does not exist in this yaml file")


if __name__ == '__main__':
    yml = get_yml("./", "experiments")
    names, completed, commands = get_exp_names_commands(yml)
    print(commands["centroid-goal"])
