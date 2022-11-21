# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# export TORCH_BLADE_MHLO_DEBUG_LOG=on
export TORCH_MHLO_OP_WHITE_LIST="aten::clone;aten::var;aten::rsub;aten::amax;aten::to;aten::tanh;aten::_to_copy;prims::broadcast_in_dim;aten::new_zeros;aten::zeros_like;aten::select_scatter;aten::slice_scatter;aten::full_like;aten::where"
python3 test_bert.py --backend aot_disc 2>&1 | tee disc.compare.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o disc_debug.nvprof python3 test_bert.py --prof_dynamo --backend aot_disc_debug 2>&1 | tee disc_debug.nvprof.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o disc_debug python3 test_bert.py --prof_dynamo --backend aot_disc_debug 2>&1 | tee disc_debug.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o disc_debug disc_debug.nsys-rep
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o disc_debug disc_debug.nsys-rep
python3 parse_nsys_results.py disc_debug_gputrace.csv > disc_debug.report
 
# export TORCH_BLADE_DEBUG_LOG=on
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o disc.nvprof python3 test_bert.py --prof_dynamo --backend aot_disc 2>&1 | tee disc.nvprof.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o disc python3 test_bert.py --prof_dynamo --backend aot_disc 2>&1 | tee disc.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o disc disc.nsys-rep
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o disc disc.nsys-rep
python3 parse_nsys_results.py disc_gputrace.csv > disc.report

/opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o eager.nvprof python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee eager.nvprof.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o eager python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee eager.log
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o eager eager.nsys-rep
/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o eager eager.nsys-rep
python3 parse_nsys_results.py eager_gputrace.csv > eager.report

# #/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o base python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee base.log
#/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o base base.nsys-rep
#python3 parse_nsys_results.py base_gputrace.csv > base.report

# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o base base.nsys-rep
