# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import math

debug_level = 0
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Aggregation of all line solutions, represented as statistical distribution 
# 1.0 - always set,  -1.0 - always unset 
class Variants:
    def __init__(self, variants):
        self.aggregate = None
        self.count = 0
        for var in variants:
            if self.aggregate is None:
                self.aggregate = np.zeros((var.sum()))
            self.aggregate += var.unpack()
            self.count += 1
        if self.count:
            self.aggregate /= self.count
    def __repr__(self):
        return f"Variants {self.aggregate}"
        
# Packed representation of one line solution
# Even colums represent unset field, odd colums represent set fields
# [0, 1, 3, 2] represents [.X...XX]
class Variant:
    unpacks_done = 0
    def __init__(self, list):
        self.list = list
        self.unpacked = None
    def __repr__(self):
        return f"Variant {self.str()} {[x for x in self.list]}"
    def str(self):
        out = ""
        for i in self.unpack():
            out += 'X' if i == 1 else '.'
        return out
    def sum(self):
        return sum(self.list)
    # Unpack to array where 1: occupied, 0: empty
    def unpack(self):
        if self.unpacked is None:
            self.unpacked = np.zeros(sum(self.list))
            index = 0
            for i in range(len(self.list)):
                for j in range(self.list[i]):
                    self.unpacked[index] = 1 if i%2 else -1
                    index += 1
            Variant.unpacks_done += 1
        return self.unpacked
    
    # Is compatible with already laid down pieces?
    def can_fit(self, known_pieces):
        if not len(known_pieces): return True
        unpacked = self.unpack()
        for i,j in zip(unpacked, known_pieces):
            if (j+i==0): return False   # unpacked can be only 1 or -1. Zero can happen if the other is opposite
        return True
    
        
# Given specification of one line (row or column). E.g. [1,5,2]
class Seq:
    def __init__(self, list):
        self.list = list
        self.allvariants = None
        self.known_pieces = None
    def __repr__(self):
        return f"Seq {self.list}"
    # partial - partially built list of this combination
    # Note that valid_variants will actually remove the already illegal variants from the cached pool
    def variants(self, width, known_pieces=[], remove_invalid=False, partial=[]):  
        if self.known_pieces is not None:
            for k1, k2 in zip (self.known_pieces, known_pieces):
                if k1 != 0 and k1 != k2:
                    raise Exception("Can't iterate with less known pieces than before, variants were already removed")
        if self.allvariants is None:
            self.allvariants = [variant for variant in self.gather_variants(width, known_pieces, partial) if variant.can_fit(known_pieces)]
            print(f'Generated {len(self.allvariants):>6d} variants for assumed complexity {{{self.variant_count(width)}|{self.variant_discount(width, known_pieces):.1f}}}')

        goodvariants = []
        for variant in self.allvariants:
            if variant.can_fit(known_pieces): 
                yield variant 
                if remove_invalid:
                    goodvariants.append(variant)
        self.known_pieces = np.copy(known_pieces)
        if remove_invalid:
            self.allvariants = goodvariants
            
        
    def gather_variants(self, width, known_pieces, partial=[]):  
        if debug_level > 2: print(f'remain={self.list}, partial={partial}')
        if not Variant(partial).can_fit(known_pieces):
            return
        if not self.list:
            # Successfully emptied all requirements
            finished = Variant(partial + [width - sum(partial)])
            if finished.can_fit(known_pieces):
                yield finished
            return
        # Item that will be placed
        first = self.list[0]
        # Minimum space needed to fit all the remaining ones
        reserved = sum(self.list[1:]) + len(self.list[1:])
        for i in range(0 if not partial else 1, 1 + width - (first + reserved + sum(partial))):
            yield from Seq(self.list[1:]).gather_variants(width, known_pieces, partial + [i, first])
    
    def variant_count(self, width):
        space, needles = (width - (len(self.list)-2 + sum(self.list)), len(self.list))
        if space < 1: return 0
        # print(f'{space=},{needles=}')
        return int(math.factorial(space + needles - 1) / (math.factorial(space-1) * math.factorial(needles)))
    
    # Variant count discounted by the number of known pieces - %SALE %SALE %SALE!
    def variant_discount(self, width, known_pieces):
        if hasattr(known_pieces, '__iter__'): 
            known_pieces = np.count_nonzero(known_pieces)
        return self.variant_count(width) * 0.8**known_pieces  # Definitely not exact. The exact distribution matters here.
         # But we don't need exact, just useful and fast enough to choose the next cheap step!

    # Try the obvious step - if there are known pieces all the way from any border, fill in gaps without getting all the variants
    def fill_obvious(self, width, known_pieces):
        changed, known_pieces = fill_obvious_list(self.list, known_pieces)
        changed_r, known_pieces = fill_obvious_list(self.list[::-1], known_pieces[::-1])
        return changed or changed_r, known_pieces[::-1]
        
def fill_obvious_list(list, known_pieces):
    if not len (list):
        return (False, known_pieces)
    islands = iter(list)
    val = -1  # Currently iterated value
    count = 0 # Number of occurrences so far
    pos = 0   # Position in known_pieces
    did_change = False
    for piece in known_pieces:
        if piece == val:
            # Continue accumulating, still the same
            count += 1
        elif piece == -1 or piece == 1:
            # Different piece
            if piece == -1:
                if not island == count:
                    raise Exception('Invalid combination')
            else:
                island = next(islands)
            val = piece
            count = 1
        else:
            break
        pos += 1
    if val == 1 and island < count:
        raise Exception('Invalid combination')
    # We still have something to fill
    if pos != len(known_pieces) and val == 1:
        while island - count > 0:
            known_pieces[pos] = 1  # Fill rest of the island
            island -= 1
            pos += 1
            did_change = True
        if pos != len(known_pieces):
            known_pieces[pos] = -1  # Needs to terminate with empty
            did_change = True
    return (did_change, known_pieces)
        
        

# Solving matrix
class Field:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.mat = np.zeros((len(self.rows), len(self.cols)))
        self.probs = self.mat.copy() # Probabilities
        self.npass = 0
        self.complexity = 0
        self.more_solutions = False
    def __repr__(self):
        return f'{self.mat}'
    def solve_step(self):
        self.complexity = 0
        while True:
            previous_mat = self.mat.copy()
            row_val, col_val = 1, 0

            # Find the least complex line - sort by complexity. It is only an approximation as also the shape, not just the count of 
            #  known pieces matters here. We're ignoring the shape for simplicity for now.
            lines = [(i,line,row_val,self.mat[i,:],line.variant_discount(self.mat.shape[row_val], np.count_nonzero(self.mat[i,:])))
                     for i,line in enumerate(self.rows)] + \
                    [(i,line,col_val,self.mat[:,i],line.variant_discount(self.mat.shape[col_val], np.count_nonzero(self.mat[:,i])))
                     for i,line in enumerate(self.cols)]
            lines = filter(lambda el: 0 in el[3] or np.array_equal(el[1].known_pieces, el[3]), lines)
            lines = sorted(lines, key = lambda t: t[4]) # Sort by discount
            line_processed = None
            
            # Fill obvious pass - too much duplicity
            print(f'Complexities: {[int(candidate[4]) for candidate in lines]}')
            for i, line, is_row, known_pieces, complexity in lines:
                prev_pieces = copy.copy(known_pieces)
                length = self.mat.shape[is_row]
                filled, known_pieces = line.fill_obvious(length, known_pieces)
                #if filled:
                #    print(f'Did obvious fill in {"row" if is_row else "col"} {i} of {prev_pieces} into {known_pieces}')
       
            # Iterate from simplest to most complex and find the first line where we can delve some new information
            for i, line, is_row, known_pieces, complexity in lines:
                if line_processed is not None:
                    break # Last line did add some new information, restart from simplest ones
                length = self.mat.shape[is_row]
                if line.allvariants is None: print(f'Generating all variants in {"row" if is_row else "col"} {i} with {known_pieces}')
                variants = Variants(line.variants(length, known_pieces, not self.more_solutions))
                if is_row: self.probs[i,:] = variants.aggregate 
                else: self.probs[:,i] = variants.aggregate

                if variants.aggregate is None:
                    return # There is no solution
                for j in range(len(variants.aggregate)):
                    prob = variants.aggregate[j]
                    if prob == -1 or prob == 1:
                        row = j if not is_row else i
                        col = i if not is_row else j
                        if (self.mat[row,col] != prob):
                            line_processed = line # = ((row, col), (self.mat[row,col], prob))
                            self.mat[row,col] = prob

                self.npass += 1
            if debug_level > 1: print(self.mat)
            changed = (previous_mat != self.mat).any()
            
            # Handle possibility we need to take a guess to continue
            if not changed and not self.is_solved():
                self.more_solutions = True
                guess = copy.copy(self)
                option1, option2 = self.take_a_guess()
                guess.mat = option1
                for advanced in guess.solve_step():
                    self.mat = guess.mat
                    if advanced:
                        yield advanced
                self.complexity += guess.complexity    
                if not self.is_solved():
                    # Guess did not work out, continue with the other option (as long there's only 2)
                    self.mat = option2
                    changed = True
                    
            if not changed:
                break
            yield (True, line_processed)
    
    def take_a_guess(self):
        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                if self.mat[i,j] == 0:
                    option1 = self.mat.copy()
                    option1[i,j] = 1
                    option2 = self.mat.copy()
                    option2[i,j] = -1
                    return (option1, option2)
    
    def is_solved(self):
        return not 0 in self.mat
    
    def last_complexity(self):
        return self.complexity
    
    def visualize(self):
        fig, ax = plt.subplots(figsize=(2, 8))
        ax.pcolormesh(self.mat, cmap='terrain_r')
        ax.set_aspect('equal')
        ax.set_xlim(0, self.mat.shape[1])
        ax.set_ylim(self.mat.shape[0], 0)
        plt.show()
        
    def visualize_probs(self):
        rows = [0 if row.allvariants is None else 1.0 for row in self.rows]
        rows = np.array(rows)
        rows = rows.reshape(len(rows), 1)
        
        cols = [0 if col.allvariants is None else 1.0 for col in self.cols]
        cols = np.array([0] + cols)
        cols = cols.reshape(1, len(cols))
        
        all = np.concatenate((rows, self.probs), axis=1)
        all = np.concatenate((cols, all), axis=0)
        fig, ax = plt.subplots(figsize=(2, 8))
        ax.pcolormesh(all, cmap='viridis_r')
        ax.set_aspect('equal')
        ax.set_xlim(0, all.shape[1])
        ax.set_ylim(all.shape[0], 0)
        # bx.pcolormesh(rows, cmap='terrain_r')
        plt.show()


# %%
Seq([1,2,3]).fill_obvious(10, [1,-1,1,0,-1,0,0,0,0,1])

# %%
debug_level = 1
sequence = [1,4,3,1]
vars = Variants(Seq(sequence).variants(30)) # All possibilities
print(f'Variants: {vars}, count: {vars.count}')
vars = Variants(Seq(sequence).variants(30, [0,0,0,0,0,0,0,0,0,1,0])) # All possibilities
print(f'Variants: {vars}, count: {vars.count}')

# print(Variants(Seq([2,1,1]).variants(7, [0, 0, 0, 1, -1, 0, 0 ]))) # Filtered by known pieces
# print(Variants(Seq([2,1,1]).variants(7, [0, 0, 0, 1, -1, -1, -1 ]))) # Filtered by known pieces
# print(Variants(Seq([2,2,2,2,2,5,5,2]).variants(50)))
print(Seq([2,2,2,2,2,5,5,2]).variant_count(50))
[variant for variant in Seq([1,1,1]).variants(7)]

elems = [3, 1, 2, 1]
width = 12
print(len([print(variant) for variant in Seq(elems).variants(width)]))
print(f'VarCount: {Seq(elems).variant_count(width)}')

# %%
import requests
import re

def get_actual_puzzle(id):
    url = f'https://www.griddlers.net/en_US/nonogram/-/g/t1679057429974/i01?p_p_lifecycle=2&p_p_resource_id=griddlerPuzzle&p_p_cacheability=cacheLevelPage&_gpuzzles_WAR_puzzles_id={id}&_gpuzzles_WAR_puzzles_lite=false'
    r = requests.get(url)
    match = re.search("var puzzle =(.+)\n\nvar solution", r.text, re.DOTALL)
    js_object = match.group(1)
    # replace single quotes with double quotes
    js_object = js_object.replace("'", '"')
    # wrap keys with double quotes
    js_object = re.sub(r"(\w+)\s?:", r'"\1":', js_object)
    # wrap values with double quotes except for numbers or booleans
    js_object = re.sub(r":\s?(?!(\d+|true|false))(\w+)", r':"\2"', js_object)
    # eradicate continuation , on last element in list   <--  hacky, works only on this one ^_^
    js_object = re.sub(r'",\s*\n', r'"\n', js_object)
    barely_json = json.loads(js_object)
    
    # print(barely_json)
    rows = [[row[1] for row in rowses] for rowses in barely_json["leftHeader"]]
    cols = [[col[1] for col in colses] for colses in barely_json["topHeader"]]
    print(f'Downloaded nonogram {id} from the internets with {len(rows)} rows, {len(cols)} cols')
    return rows, cols

def get_test_puzzle(file):
    with open(file) as f:
        data = json.load(f)
    rows = data['rows']
    cols = data['columns']
    return rows, cols


# %%
# Load our next puzzle
# rows, cols = get_test_puzzle('gridler.multisolve.json.txt')
rows, cols = get_actual_puzzle(188687)  # 266162 <-- Deer

srows = [Seq(row) for row in rows]
scols = [Seq(col) for col in cols]
field = Field(srows, scols)

prev = start
step = 0
debug_level = 0
Variant.unpacks_done = 0
start = time.perf_counter()

# Fire the solver
for advanced, processed in field.solve_step():
    step += 1
    if step % 3 == 0:
    # if True:
        print(f'step {step}, time taken so far: {time.perf_counter() - start:.3f} s, complexity: {field.last_complexity()}')
        field.visualize_probs()
    prev = time.perf_counter()
    if not advanced:
        break

if not field.is_solved():
    print("No possible solutions :-(  (or just can't solve them :)")
else:
    print(f"Solved in {step} steps :-)") 
    if field.more_solutions: print(f"There are multiple possible solutions and this is just one of them")
print(f" took {time.perf_counter() - start:.3f} s")
print(f"{Variant.unpacks_done=} (the expensive stuff)")
field.visualize()

# Current state:
#  Still No branching support 

# %%

# %%

# %%
