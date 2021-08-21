import pickle
import sys
from collections import defaultdict

if len(sys.argv) < 5:
    print("Usage: python cache_lf.py path_of_stat_file path_of_program_file section output_file")
    sys.exit(0)

sketch_file_name = sys.argv[1]
program_file_name = sys.argv[2]
section = sys.argv[3]
output_filename = sys.argv[4]

with open(sketch_file_name, "rb") as f:
    sketch_dict = pickle.load(f)
print(f"Raw number of sketch (from stat): {len(sketch_dict)}")
_dc_sketch_list = []
program_file_name = sys.argv[2]
id2question = [] 
question2program = dict()
program_counter = 0
with open(program_file_name, "r") as f:
    for line in f:
        line = line[:-1]
        flag = section + "-"
        if line.startswith(flag):
            q_id = line.split()[0]
            line = next(f)[:-1]
            lh, rh = line.split()
            assert lh == "Table:"
            q_t_pair = (q_id, rh)
            id2question.append(q_t_pair)

            line = next(f)[:-1]
            if line == "NO LOGICAL FORMS FOUND!":
                line = next(f) #blank line
                continue
                

            _sketch2program = defaultdict(list)
            while line:
                assert line.startswith("Sketch: ")
                _dc_sketch_list.append(line)
                sketch = line.strip()[8:]

                line = next(f)[:-1] 
                while line.startswith("\t"):
                    program_counter += 1
                    line = line.strip()
                    _sketch2program[sketch].append(line)
                    line = next(f)[:-1] 
            
            question2program[q_t_pair] = _sketch2program

print(f"Raw number of sketch (from program): \
    {len(set(_dc_sketch_list))}")
print(f"Raw number of programs: {program_counter}")

print(f"Raw number of sketch (from program): \
    {len(set(_dc_sketch_list))}")
print(f"Raw number of programs: {program_counter}")

# double check
counter = 0
with open(program_file_name, "r") as f:
    for line in f:
        if line.startswith("\t"):
            counter += 1
print(f"double check # of programs: {counter}")

for example in question2program:
    for sketch in question2program[example]:
        assert sketch in sketch_dict

sketch_list = list(sketch_dict.keys())
sketch_list = sorted(sketch_list, key= lambda x: len(sketch_dict[x]))
sketch_list = list(reversed(sketch_list))
sketch2id = {v:k for k,v in enumerate(sketch_list)}

example_dict = defaultdict(set)
for sketch in sketch_dict:
    for q, t in sketch_dict[sketch]:
        example_dict[(q, t)].add(sketch)
print(f"Coverage: {len(example_dict) / len(id2question)}, {len(example_dict)}:{len(id2question)}")

with open(output_filename, 'wb') as f:
    pickle.dump(question2program, f)
