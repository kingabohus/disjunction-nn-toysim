import random
from copy import deepcopy
import numpy as np


class MyScene:

    def __init__(self, p):
        self.p = p  # this can even increase as len(states increases)
        self.currentstate = MyState()
        self.states = []
        self.states.append(self.currentstate)

    def gen_next_state(self, pe):
        found = False
        while ~found:
            if (random.random() < self.p * len(self.states)) & (len(self.states) != 1):

                return "end"
            else:
                event = random.choice(pe)  # add bigger weights by including multiple times in pe
                if event.check_condition(self.currentstate):
                    self.currentstate = event.apply_effect(self.currentstate)
                    self.states.append(self.currentstate)
                    found = True

                    return event.name


class MyState:

    def __init__(self, o1=0, o1loc=0, o2=0, o2loc=0, wall=0, cup=1, cupfull=0):
        self.o1 = o1
        self.o1loc = o1loc
        self.o2 = o2
        self.o2loc = o2loc
        self.wall = wall
        self.cup = cup
        self.cupfull = cupfull
        self.cantscoop = self.o1 * self.o2 * self.wall
        # self.wallonce = 0
        # self.o1midonce = 0
        # self.o2midonce = 0
        self.diction = {
            "o1": self.o1,
            "o1loc": self.o1loc,
            "o2": self.o2,
            "o2loc": self.o2loc,
            "wall": self.wall,
            "cup": self.cup,
            "cupfull": self.cupfull,
            "cantscoop": self.cantscoop,
            # "wallonce": self.wallonce,
            # "o1midonce": self.o1midonce,
            # "o2midonce": self.o2midonce
        }

    def __str__(self):
        print(self.diction)
        return ""


class MyEvent:

    def __init__(self, name, condition, effect):
        # name: name of event
        # condition: list of 3-tuples (var, value, equal). var is the variable name in the condition, value is the
        # value to
        # compare, equal is 1 if equal, 0 if unequal. So for example the condition o1 = 1 is ("o1", 1, 1) and the
        # condition o1Loc != 3 is ("o1Loc", 3, 0)
        # effect: list of tuples (var, value) if the var is not included, no change
        self.name = name
        self.condition = condition
        self.effect = effect

    def check_condition(self, state):
        cond = True
        for i in self.condition:
            varvalue = state.diction[i[0]]
            if i[2] == 1:
                if varvalue != i[1]:
                    cond = False
            else:
                if varvalue == i[1]:
                    cond = False
        return cond

    def apply_effect(self, state):
        new_state = deepcopy(state)
        for i in self.effect:
            new_state.diction[i[0]] = i[1]
            if i[0] == 'o1':
                new_state.o1 = i[1]
            elif i[0] == 'o2':
                new_state.o2 = i[1]
            elif i[0] == 'o1loc':
                new_state.o1loc = i[1]
            elif i[0] == 'o2loc':
                new_state.o2loc = i[1]
            elif i[0] == 'wall':
                new_state.wall = i[1]
            elif i[0] == 'cup':
                new_state.cup = i[1]
            elif i[0] == 'cupfull':
                new_state.cupfull = i[1]
            # elif i[0] == 'wallonce':
            #     new_state.wallonce = i[1]
            # elif i[0] == 'o1midonce':
            #     new_state.o1midonce = i[1]
            # elif i[0] == 'o2midonce':
            #     new_state.o2midonce = i[1]

        new_state.cantscoop = new_state.o1 * new_state.o2 * new_state.wall
        new_state.diction["cantscoop"] = new_state.cantscoop

        return new_state


class Translator:
    def __init__(self):
        self.out = []

    def print_scene(self, scene):
        inseq = []
        outseq = []
        for st in scene.states[1:]:
            outputarray = []
            inputarray = [st.o1]
            if st.o1loc == 1:
                if st.wall == 0:
                    inputarray.extend([1, 0, 0])
                else:
                    inputarray.extend([0, 0, 0])
            elif st.o1loc == 2:
                inputarray.extend([0, 1, 0])
            elif st.o1loc == 3:
                if st.cup == 0:
                    inputarray.extend([0, 0, 1])
                else:
                    inputarray.extend([0, 0, 0])
            else:
                inputarray.extend([0, 0, 0])

            inputarray.append(st.o2)
            if st.o2loc == 1:
                if st.wall == 0:
                    inputarray.extend([1, 0, 0])
                else:
                    inputarray.extend([0, 0, 0])
            elif st.o2loc == 2:
                inputarray.extend([0, 1, 0])
            elif st.o2loc == 3:
                if st.cup == 0:
                    inputarray.extend([0, 0, 1])
                else:
                    inputarray.extend([0, 0, 0])
            else:
                inputarray.extend([0, 0, 0])

            inputarray.append(st.wall)
            inputarray.append(st.cup)
            inputarray.append(st.cupfull)

            if st.o1loc == 1:
                outputarray.extend([1, 0, 0])
            elif st.o1loc == 2:
                outputarray.extend([0, 1, 0])
            elif st.o1loc == 3:
                outputarray.extend([0, 0, 1])
            else:
                outputarray.extend([0, 0, 0])

            if st.o2loc == 1:
                outputarray.extend([1, 0, 0])
            elif st.o2loc == 2:
                outputarray.extend([0, 1, 0])
            elif st.o2loc == 3:
                outputarray.extend([0, 0, 1])
            else:
                outputarray.extend([0, 0, 0])

            inseq.append(inputarray)
            outseq.append(outputarray)
        return [inseq, outseq]


event1 = MyEvent("O1enter", [("o1", 0, 1)], [("o1", 1), ("o1loc", 1)])
event2 = MyEvent("O2enter", [("o2", 0, 1)], [("o2", 1), ("o2loc", 1)])

event3 = MyEvent("wallup", [("wall", 0, 1)], [("wall", 1)])
# event3 = MyEvent("wallup", [("wall", 0, 1), ("wallonce", 0, 1)], [("wall", 1), ("wallonce", 1)])
event4 = MyEvent("walldown", [("wall", 1, 1)], [("wall", 0)])
event5 = MyEvent("scoopO1", [("cantscoop", 1, 0), ("o1", 1, 1), ("o1loc", 1, 1), ("cupfull", 0, 1), ("cup", 1, 1)],
                 [("cupfull", 1), ("o1loc", 3)])
event6 = MyEvent("scoopO2", [("cantscoop", 1, 0), ("o2", 1, 1), ("o2loc", 1, 1), ("cupfull", 0, 1), ("cup", 1, 1)],
                 [("cupfull", 1), ("o2loc", 3)])
event7 = MyEvent("O1tomid", [("o1loc", 1, 1), ("o2loc", 2, 0)], [("o1loc", 2)])
# event7 = MyEvent("O1tomid", [("o1loc", 1, 1), ("o2loc", 2, 0), ("o1midonce", 0, 1)], [("o1loc", 2), ("o1midonce", 1)])
event8 = MyEvent("O1back", [("o1loc", 2, 1)], [("o1loc", 1)])
event9 = MyEvent("O2tomid", [("o2loc", 1, 1), ("o1loc", 2, 0)], [("o2loc", 2)])
# event9 = MyEvent("O2tomid", [("o2loc", 1, 1), ("o1loc", 2, 0), ("o2midonce", 0, 1)], [("o2loc", 2), ("o2midonce", 1)])
event10 = MyEvent("O2back", [("o2loc", 2, 1)], [("o2loc", 1)])

event11 = MyEvent("O1exit", [("o1", 1, 1), ("o1loc", 1, 1)], [("o1", 0)])
event12 = MyEvent("O2exit", [("o2", 1, 1), ("o2loc", 1, 1)], [("o2", 0)])

event13 = MyEvent("cupup", [("cup", 0, 1)], [("cup", 1)])
event14 = MyEvent("cupdown", [("cup", 1, 1)], [("cup", 0)])

event15 = MyEvent("backscoopO1", [("o1", 1, 1), ("o1loc", 3, 1), ("cup", 1, 1)], [("o1loc", 1)])
event16 = MyEvent("backscoopO2", [("o2", 1, 1), ("o2loc", 3, 1), ("cup", 1, 1)], [("o2loc", 1)])

# TO DO: back-scoop. O1 exit, O2 exit.

# add an event more than once to increase probability of choosing event in case it's possible

possible_events = [event1, event2, event1, event2, event3, event4, event5, event6, event5, event6, event7, event8, event9, event10,
                   event11, event12, event13, event14, event15, event16]
possible_events_limited = [event1, event2, event1, event2, event3, event4, event5, event6, event5, event6, event7, event8, event9, event10]

limited = True
t = Translator()
prob = 0.002  # control avg length of scenes. Higher prob : shorter sequence
N = 5000
lengths = np.zeros((N, 1))

file_name_extension = f"_{N}_{int(prob*100000)}{'_ltd' if limited else ''}"
training_file = open(f"./datasets/trainingset_{file_name_extension}.py", 'w')

for n in range(N):
    print(n)
    new_scene = MyScene(prob)
    while True:
        nextstate = new_scene.gen_next_state(possible_events_limited if limited else possible_events)
        if nextstate == "end":
            break

    [inp, outp] = t.print_scene(new_scene)
    lengths[n] = len(new_scene.states) - 1
    training_file.write('x%d = ' % n)
    training_file.write("%s\n" % inp)
    training_file.write('y%d = ' % n)
    training_file.write("%s\n" % outp)
training_file.write(f"training_scenarios{'_ltd' if limited else ''} = [")
for n in range(N - 1):
    training_file.write('(\'example%d\', x%d, y%d), \n' % (n, n, n))
training_file.write('(\'example%d\', x%d, y%d)]' % (N - 1, N - 1, N - 1))
training_file.write("\n# Avg length: {}, N = {}, p = {}".format(np.mean(lengths), N, prob))
training_file.write(f"\nlog_file_name = \"logs/log_{file_name_extension}.txt\"")
training_file.close()
print("Avg length: {}, N = {}, p = {}".format(np.mean(lengths), N, prob))
