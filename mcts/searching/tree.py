import pdb
import re
import random


class Node(object):

    def __init__(
        self, id="root", 
        parent: "Node" = None, 
        state=None, 
        initial_value: float = 0.0, 
        ori_text = '',
        depth = 0,
        message_history = []
    ):
        self._parent = parent
        self._children = []
        self._answer_list = []
        self._answer_reward_list = []
        self._candidate_answer = ""
        self._visit_count = 0
        self._value_sum = 0
        self._terminated = False
        self._initial_value = initial_value
        self.state = state
        # self._end = False
        self._id = id
        self._success = False
        # self._message_history = message_history
        self.ori_text = ori_text
        self.init_ppl = 0
        self.init_bart = 0
        self._depth = depth
        self.ppl_list = []
        self.bart_list = []
        self.some_list = []
        self.cons_list = []

    def set_ori_text(self, ori_text):
        self.ori_text = ori_text

    def update_state(self, new_state):
        self.state = new_state

    def set_success(self):
        self._success = True

    def get_node_id(self):
        return self._id

    def value(self):
        if self._visit_count == 0:
            return self._initial_value
        return self._value_sum / self._visit_count

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def answers(self):
        return self._answer_list

    def answer_rewards(self):
        return self._answer_reward_list

    def candidate_answer(self):
        return self._candidate_answer

    def visit_count(self):
        return self._visit_count

    def is_terminated(self):
        return self._terminated

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return self._children == []

    def update(self, value: float):
        self._visit_count += 1
        self._value_sum += value

    def add_child(self, node):
        self._children.append(node)

    def extend_answer(self, answers):
        self._answer_list.extend(answers)

    def set_candidate_answer(self, candidate_answer):
        self._candidate_answer = candidate_answer

    def get_ori_text(self):
        return self.ori_text
    
    def get_tool_lists(self):
        return self.tool_lists 
    
    def set_tool_lists(self, tool_lists):
        self.tool_lists = tool_lists

    def get_init_ppl_bart(self):
        return self.init_ppl, self.init_bart

    def set_init_ppl_bart(self, init_ppl, init_bart):
        self.init_ppl = init_ppl
        self.init_bart = init_bart

    def get_trajectory(self):
        ori_text = self.get_ori_text()
        polished_text = self.candidate_answer()
        tools_list = self.get_tool_lists()
        init_ppl, init_bart = self.get_init_ppl_bart()

        # print("Inside get_trajectory")
        # print("Initial PPL: {}".format(init_ppl))
        # print("Initial BART: {}".format(init_bart))

        return ori_text, polished_text, tools_list, init_ppl, init_bart

        #ori_text, polished_paragraph, checking_tool_lists, ini_ppl, ini_bart

    # def get_trajectory(self):
    #     ori_text = self.ori_text
    #     polished_text = self.candidate_answer()
    #     tools_list = self.get_tool_lists()
    #     return ori_text, polished_text, tools_list, 

    #     trajectory = self.state + self.candidate_answer()
    #     queries, sentences = [], []
    #     documents = dict()
    #     for line in trajectory.split("\n"):
    #         if line.startswith("Search:"):
    #             queries.append(line[len("Search:"):].strip())
    #         elif line.startswith("Output:"):
    #             sentences.append(line[len("Output:"):].strip())
    #         elif line.startswith("Document"):
    #             doc_id = re.findall(r"\[\d+", line)[0][1:].strip()
    #             title = re.findall(r"\(Title: .+\)", line)[0][7:-1].strip()
    #             text = line.replace("Document [" + doc_id + "] (Title: " + title + ")", "").strip()
    #             documents[doc_id] = (title, text)
    #         else:
    #             continue
    #     sentences = " ".join(sentences)

    #     return queries, sentences, documents
    
    def set_history_metrics(self, ppl_list, bart_list, some_list, cons_list):
        self.ppl_list = ppl_list
        self.bart_list = bart_list
        self.some_list = some_list
        self.cons_list = cons_list
        
    def get_history_metrics(self):
        return self.ppl_list, self.bart_list, self.some_list, self.cons_list
    
    def get_depth(self):
        return self._depth
    
    def set_depth(self, depth):
        self._depth = depth 

    def set_terminated(self):
        self._terminated = True

    def get_info(self):
        return {
            "state": self.state,
            "visit_count": self._visit_count,
            "value": self.value(),
            "terminated": self.is_terminated(),
            "answers": self.answers(),
            "answers_reward_list": self.answer_rewards(),
            "idx": self._id,
            "parent": None if self._parent is None else self._parent._id,
            "success": self._success,
        }
