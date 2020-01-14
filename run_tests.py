import unittest
import importlib

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[1m\033[94m'
    BOLDBLUE = '\033[1m\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    BOLDFAIL = '\033[1m\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


tests = [#'algo.test_core',
         #'algo.test_dummy',
         # 'algo.test_dynamicsmodel',
         # 'algo.test_splitdynamicsmodelpose',
         # 'algo.test_splitdynamicsmodelposelstm',
         'algo.test_util',
         # 'utils.test_math',
         # 'utils.test_orientation',
         # 'envs.test_clutter'
        ]

failed_tests = []

for test in tests:
    print(bcolors.BOLDBLUE + "Test: " + test + bcolors.ENDC)

    module = importlib.import_module('robamine.test.' + test)
    suite = unittest.TestLoader().loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        print(bcolors.BOLDFAIL + "Test: " + test + " Failed." + bcolors.ENDC)
        failed_tests.append(test)

if len(failed_tests) > 0:
    print(bcolors.FAIL + "The following robamine tests failed:" + bcolors.ENDC)
    for failed_test in failed_tests:
        print(bcolors.FAIL + "[x] " + failed_test + bcolors.ENDC)
else:
    print(bcolors.OKGREEN + "Every robamine test was succesful." + bcolors.ENDC)
