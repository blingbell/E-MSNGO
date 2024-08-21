
from collections import deque, Counter
import math


ROOT_ONTOLOGY = {'mf': 'GO:0003674', 'bp': 'GO:0008150', 'cc': 'GO:0005575'}
ROOT_GO_TERM = {'GO:0003674', 'GO:0008150', 'GO:0005575'}
NAMESPACE = {'biological_process': 'bp', 'molecular_function': 'mf', 'cellular_component': 'cc'}
RELATIONSHIP = {'is_a', 'part_of', 'regulates', 'has_part' }


def init_term():
    term = dict()
    term['alt_ids'] = set()
    term['is_obsolete'] = False
    term['children'] = set()
    term['parents'] = set()
    for rel in RELATIONSHIP:
        term[rel] = set()
    return term

class GeneOntology(object):
    def __init__(self, go_file_path):
        self.graph = self.load(go_file_path)
        self.ic = None

    def calculate_ic(self, annot_list):
        cnt = Counter()
        for annots in annot_list:
            cnt.update(annots)
        self.ic = {}
        for go_id, n in cnt.items():     
            parents = self.get_parents(go_id)
            parents_cnt = [cnt[x] for x in parents if x not in ROOT_GO_TERM]
            if len(parents) == 0 or len(parents_cnt) == 0:
                min_n = n
            else:
                min_n = min(parents_cnt)
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename):
        ont = dict()
        term = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if term is not None:
                        ont[term['id']] = term
                    term = init_term()
                    continue
                elif line == '[Typedef]':
                    if term is not None:
                        ont[term['id']] = term
                    term = None
                else:
                    if term is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        term['id'] = l[1]
                    elif l[0] == 'alt_id':
                        term['alt_ids'].add(l[1])
                    elif l[0] == 'namespace':
                        term['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        term['is_a'].add(l[1].split(' ! ')[0])
                    elif l[0] == 'relationship':
                        it = l[1].split()
                        if it[0] in RELATIONSHIP:
                            term[it[0]].add(it[1])
                    elif l[0] == 'name':
                        term['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        term['is_obsolete'] = True
            if term is not None:
                ont[term['id']] = term

        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]

        for term_id, val in ont.items():
            for p_id in val['is_a']:
                if p_id in ont:
                    ont[term_id]['parents'].add(p_id)
                    ont[p_id]['children'].add(term_id)
            for p_id in val['part_of']:
                if p_id in ont:
                    ont[term_id]['parents'].add(p_id)
                    ont[p_id]['children'].add(term_id)

        return ont

    def get_ancestors(self, term_id):
        """ get all ancestors of term_id include itself. For up-propagation. """
        if term_id not in self.graph:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.graph[t_id]['parents']:
                    if parent_id in self.graph:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        """ get all parents of term_id not include itself. """
        if term_id not in self.graph:
            return set()
        term_set = set()
        for parent_id in self.graph[term_id]['parents']:
            if parent_id in self.graph:
                term_set.add(parent_id)
        return term_set

    def get_children(self, term_id):
        if term_id not in self.graph:
            return set()
        term_set = set()
        for child_id in self.graph[term_id]['children']:
            if child_id in self.graph:
                term_set.add(child_id)
        return term_set

    def get_namespace(self, term_id):
        """ return bp, cc or mf """
        return NAMESPACE[self.graph[term_id]['namespace']]
    
    def has_term(self, term_id):
        return term_id in self.graph

    def get_relationship(self, term_id, rel):
        if term_id not in self.graph or rel not in self.graph[term_id]:
            return set()
        return self.graph[term_id][rel]

    def get_all_remote_relationship(self, term_id):
        if term_id not in self.graph:
            return set()
        rel_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in rel_set:
                rel_set.add(t_id)
                for rel in RELATIONSHIP:
                    for partner_id in self.graph[t_id][rel]:
                        if partner_id in self.graph:
                            q.append(partner_id)
        return rel_set
    def term_top_sort(self, namespace):
        sorted_terms = list()
        tmp_graph = self.graph
        root_term = ROOT_ONTOLOGY[namespace]
        q = deque()
        flag_dict = dict()
        q.append(root_term)
        while(len(q) > 0):
            term_id = q.popleft()
            if namespace == self.get_namespace(term_id) and flag_dict.get(term_id) == None:
                sorted_terms.append(term_id)
                flag_dict[term_id] = True
            for child_id in tmp_graph[term_id]['children']:
                tmp_graph[child_id]['parents'] -= {term_id} 
                if len(tmp_graph[child_id]['parents']) == 0:
                    q.append(child_id)
        # print(len(sorted_terms))
        return sorted_terms
