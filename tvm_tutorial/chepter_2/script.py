#!/usr/local/Caskroom/miniforge/base/envs/tvm/bin/python3

# Step 1: Load a model
from tvm.driver import tvmc
model = tvmc.load('my_model.onnx') # Step 1: Load

# See Relay
model.summary()

# Step 2: Compile
package = tvmc.compile(model, target="llvm") #Step 2: Compile

# Step 3: Run
result = tvmc.run(package, device="cpu") #Step 3: Run

# Step 1.5: Tune [Optional & Recommended]
tvmc.tune(model, target="llvm") #Step 1.5: Optional Tune

# Saving the tuning result
tvmc.compile(model, target="llvm", tuning_records = "records.log") #Step 2: Compile

# Saving the package
tvmc.compile(model, target="llvm", package_path="my_model.tar") #Step 2: Compile
new_package = tvmc.TVMCPackage(package_path="my_model.tar")
result = tvmc.run(new_package, device="cpu") #Step 3: Run