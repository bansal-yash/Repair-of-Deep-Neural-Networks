from maraboupy import MarabouCore

# Determine correct EquationType location
if hasattr(MarabouCore, "EquationType"):
    EqType = MarabouCore.EquationType
else:
    EqType = MarabouCore.Equation.EquationType

print(EqType)