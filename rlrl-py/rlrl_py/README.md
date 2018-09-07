# rlrl_py

## Available environments:

* `BHandSlidePillbox-v2`: Slides a pillbox on a table using 2DOFs of force as actions.
* `BHandSlidePillbox-v3`: Exert a perperdicular force for increasing robustness on tangential disturbances. Learn 1 DOF of action, is like learning the friction between the finger and the object. For now the environment is dummy, it is the same with `BHandSlidePillbox-v2`. TODO: To be implemented.
* `FingerSlide-v1`: A dummy finger slide a box on the table. MuJoCo Model: `small_table_pillbox_finger`