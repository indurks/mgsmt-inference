#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Sagar Indurkhya"
__copyright__ = "Copyright 2019-2022, Sagar Indurkhya"
__email__ = "indurks@mit.edu"

#------------------------------------------------------------------------------#

import itertools, json, pprint as pp
from itertools import product

from z3 import And, Or, Not, Implies, Xor, Distinct, If, PbEq, PbGe, PbLe
from z3 import BoolSort

import mgsmt.formulas.lexiconformula
import mgsmt.formulas.derivationformula
from mgsmt.solver.solver_utils import distinct


class LRLexEntry:

    def __init__(self, pf, sfs, cat):
        self.pf = pf
        self.sfs = sfs
        self.cat = cat

    def sfs_str(self):
        return ','.join(['%s%s'%(x, y) for (x, y) in self.sfs])

    def cat_sfs_str(self):
        return f"{self.cat}_{self.sfs_str()}"

    def __str__(self):
        return f"{self.pf}::{self.sfs_str()}"

    def __repr__(self):
        return repr((self.pf, self.sfs, self.cat))


class LexRepr:

    def __init__(self,
                 json_strs=None,
                 lexicon_model=None,
                 derivation_models=(),
                 lex_items=None,
                 init_lex_repr=None,
                 **opts):
        if json_strs is not None:
            def proc_sf(sf):
                if 'normalize_features' in opts:
                    return (sf[0], opts['normalize_features'].get(sf[1], sf[1]))
                return tuple(sf)
            entries = set([(entry['pf'],
                            tuple(map(proc_sf, entry['sfs'])),
                            entry.get('cat', ''))
                           for x in json_strs
                           for entry in json.loads(x)])
            self.lex_entries = [LRLexEntry(*e) for e in entries]
        elif lex_items is not None:
            lex_items = json.loads(lex_items)
            entries = set([(entry['pf'],
                            tuple([tuple(sf) for sf in entry['sfs']]),
                            entry['cat'])
                           for entry in lex_items])
            self.lex_entries = [LRLexEntry(*entry) for entry in entries]
        elif lexicon_model is not None:
            lm = self.lexicon_model = lexicon_model
            lf = lm.formula
            m_eval = lm.model.evaluate

            def include_node(l_node, pf_node):
                if not derivation_models:
                    return True
                # At least one of the lexical nodes in one of the derivations
                # must connect to l_node.
                term = Or([And(lf.derivations[dm.formula.formula_name]['bus'](d_node) == l_node,
                               dm.formula.pf_map(d_node) == pf_node)
                           for dm in derivation_models
                           for d_node in dm.formula.lex1nodes()])
                result = m_eval(term)
                return result

            lex_entries = [(LRLexEntry(pf=lf.pfInterface.get_pf(pf_node),
                                       sfs=[lm.pp_term(x) for x in le['features']],
                                       cat=str(lm.model.evaluate(lm.formula.cat_map(s_node)))),
                            include_node(s_node, pf_node))
                           for s_node, le in lexicon_model.lexical_entries.items()
                           for pf_node in lf.pfInterface.non_null_nodes()
                           if m_eval(lf.pf_map(s_node, pf_node))]

            def is_in_input_lexicon(le):
                def equal_entries(x, y):
                    return (x.pf == y.pf) and (tuple(x.sfs) == y.sfs) and (x.cat == y.cat)
                
                if init_lex_repr:
                    return any([equal_entries(le, e) for e in init_lex_repr.lex_entries])
                                
                return False
            
            lex_entries = [le
                           for (le, x) in lex_entries
                           if x or is_in_input_lexicon(le)]

            self.lex_entries = lex_entries


    def _mapping_to_formula(self, lexicon_formula):
        mapping = []
        used_lf_le = []
        missing_lr_le = []
        for lr_le in self.lex_entries:
            for lf_le in lexicon_formula.entries:
                if lf_le in used_lf_le:
                    continue
                assert type(lr_le.pf) == type(lf_le.word), (type(lr_le.pf), type(lf_le.word))
                if lr_le.pf == lf_le.word:
                    mapping.append((lr_le, lf_le))
                    used_lf_le.append(lf_le)
                    break
            else:
                missing_lr_le.append(lr_le)
        unused_lf_le = [x for x in lexicon_formula.entries if x not in used_lf_le]
        return mapping, unused_lf_le


    def _impose_constraints_syn_feats(lexicon_formula, lr_sf, lf_sf_node, constrain_sf_labels=True):
        terms = []
        lf = lexicon_formula
        f_type_str, f_val_str = lr_sf
        f_type = lf.strToLType(f_type_str)
        head_movement = f_type_str[0] in ('<', '>')
        if not(f_type == lf.LTypeSort.Complete):
            terms.append(lf.lnodeType(lf_sf_node) == f_type)
            terms.append(lf.head_movement(lf_sf_node) == head_movement)
            for v in lf.syn_feats:
                if f_val_str == lf.get_feature_str(v):
                    if constrain_sf_labels:
                        terms.append(lf.featLbl(lf_sf_node) == v)
                    break
            else:
                raise Exception("Could not find the feature value: %r"%(f_val_str))
        return terms


    def _impose_constraints_on_succ_func(lexicon_formula, lr_le, lf_le):
        terms = []
        # Constraints for the LF's successor function.
        lf = lexicon_formula
        f_type_str, _ = lr_le.sfs[-1]
        ends_complete = lf.strToLType(f_type_str) == lf.LTypeSort.Complete
        num_nodes = len(lr_le.sfs) - (1 if ends_complete else 0)
        terms.extend([lf.succ(lf_le.nodes[i]) == lf_le.nodes[i+1] for i in range(num_nodes-1)])
        if ends_complete:
            terms.append(lf.succ(lf_le.nodes[num_nodes-1]) == lf.complete_node)
        else:
            terms.append(lf.succ(lf_le.nodes[num_nodes-1]) == lf.terminal_node)
        return terms


    def _impose_constraints_lexical_entry(lexicon_formula, lr_le, lf_le, constrain_sf_labels=True):
        assert type(lr_le) is LRLexEntry, (lr_le, type(lr_le))
        assert type(lf_le) is mgsmt.formulas.lexiconformula.LexicalEntry, (lf_le, type(lf_le))
        lf, cs = lexicon_formula, mgsmt.formulas.derivationformula.DerivationFormula.CatSort
        str2cat = { 'C_declarative': cs.C_declarative, 'C_question': cs.C_question,
                   'T': cs.T, 'v': cs.v, 'V': cs.V, 'P': cs.P, 'D': cs.D, 'N': cs.N}
        terms = []
        # Constraints for features that are used.
        for i, lr_sf in enumerate(lr_le.sfs):
            terms.extend(LexRepr._impose_constraints_syn_feats(lexicon_formula,
                                                               lr_sf,
                                                               lf_le.nodes[i],
                                                               constrain_sf_labels))
        # Constraints for features that are not used.
        terms.extend([And(lf.lnodeType(x) == lf.LTypeSort.Inactive, lf.featLbl(x) == lf.nil_syn_feature)
                      for x in lf_le.nodes[len(lr_le.sfs):]])
        # Constraints for the successor function.
        terms.extend(LexRepr._impose_constraints_on_succ_func(lexicon_formula, lr_le, lf_le))
        # Constraints for the categories.
        terms.append(lf.cat_map(lf_le.nodes[0]) == str2cat[lr_le.cat])
        # Constraints for a PF association.
        terms.append(lf.pf_map(lf_le.nodes[0], lf.pfInterface.get_pf_node(lr_le.pf)) == True)
        
        return terms


    def impose_constraints_lexicon(self, lexicon_formula, verbose=False):
        # Note: this method should only be called by the parsing procedure.
        s = lexicon_formula.solver
        lf = lexicon_formula
        mapping, unused_lf_le = self._mapping_to_formula(lf)
        with s.group(tag=f"Ruling out unused lexical entries."):
            s.add_conj([lf.lnodeType(le.nodes[0]) == lf.LTypeSort.Inactive
                        for le in unused_lf_le])
        for lr_le, lf_le in mapping:
            terms = LexRepr._impose_constraints_lexical_entry(lexicon_formula, lr_le, lf_le)
            try:
                with s.group(tag=f"Imposing Constraints for a Lexical Entry: {str(lr_le)}"):
                    s.add_conj(terms)
            except:
                pp.pprint(terms)

    
    def impose_partial_constraints_on_lexicon(self, lexicon_formula, verbose=True):
        s = lexicon_formula.solver
        lf = lexicon_formula
        mapping, unused_lf_le = self._mapping_to_formula(lf)

        # The entries in the lexicon formula can be thought of as falling into
        # "buckets", with each bucket being associated with a particular
        # pf_form. We will now construct these buckets and associated helper
        # functions.
        import collections
        buckets = collections.defaultdict(list)
        for lf_le in lexicon_formula.entries:
            buckets[lf_le.word].append(lf_le)

        covert_pform = [lf_le.word for lf_le in lexicon_formula.entries if not(lf_le.is_overt)][0]

        # We next initialize a mapping, M, between lexical feature sequences
        # and pairings (i.e. tuples) of lexform entries and (pform,
        # bucket_idx).
        M = {}

        # When we encounter a new lexrepr-entry, we first look to see if the
        # lexical feature sequence is in M:
        # - if so, then we know which lexform entry it will be mapped to
        # - if not, then we need to find the next available bucket entry (based
        #   on the pform associated with the lexrepr entry).
        Q = []
        overt_lr_les = [lr_le for lr_le in self.lex_entries if lr_le.pf != covert_pform]
        covert_lr_les = [lr_le for lr_le in self.lex_entries if lr_le.pf == covert_pform]
        for lr_le in (covert_lr_les + overt_lr_les):
            if lr_le.cat_sfs_str() not in M:
                zs = [v[1]['idx'] for _, v in M.items() if v[1]['pform'] == lr_le.pf]
                i = 0 if len(zs) == 0 else (max(zs) + 1)
                M[lr_le.cat_sfs_str()] = (buckets[lr_le.pf][:i+1], {'pform': lr_le.pf, 'idx': i})
            
            xs = [x for x in M[lr_le.cat_sfs_str()][0]]

            if lr_le.pf != covert_pform:
                xs.extend(buckets[covert_pform])
            
            Q.append((lr_le, xs))

        # For each entry in the input lexicon, there should be exactly one
        # entry in the lexicon formula associated with it.
        conjunction_terms = []
        for lr_le, lf_les in Q:
            disjunction_terms = [And(LexRepr._impose_constraints_lexical_entry(lf, lr_le, lf_le))
                                 for lf_le in lf_les]
            disjunction_terms = PbEq([(t, 1) for t in disjunction_terms], k=1)

            impossible_lf_les = [lf_le
                                 for lf_le in lexicon_formula.entries
                                 if str(lf_le) not in set(map(str, lf_les))]
            impossible_terms = And([Not(And(LexRepr._impose_constraints_lexical_entry(lf, lr_le, lf_le)))
                                    for lf_le in impossible_lf_les])

            conjunction_terms.append(disjunction_terms)
            conjunction_terms.append(impossible_terms)

        with s.group(tag=f"Enforcing that input lexicon is a subset of the output lexicon."):
            s.add_conj(conjunction_terms)


    def _get_data_repr(self, derivation_model=None):
        relevant_lex_entries = []

        data_repr = [{'pf': le.pf, 'sfs': le.sfs, 'cat': le.cat}
                     for le in self.lex_entries]
        return list(sorted(data_repr, key=lambda k: repr(k)))


    def json(self, filepath=None, derivation_model=None):
        if filepath:
            with open(filepath, 'w') as f_json:
                json.dump(self._get_data_repr(derivation_model=derivation_model), f_json, sort_keys=True)
        else:
            return json.dumps(self._get_data_repr(derivation_model=derivation_model), sort_keys=True)


    def __repr__(self):
        return str(self._get_data_repr())


    def __str__(self, separator='\n'):
        return separator.join(str(le) for le in sorted(self.lex_entries, key=lambda k: k.pf))


    def enum_pforms(self):
        return list(sorted([le.pf for le in self.lex_entries]))


    def latex(self, input_lexicon_lr=None):
        assert input_lexicon_lr == None
        import collections
        le_dict = collections.defaultdict(list)
        for le in sorted(self.lex_entries, key=lambda x: (x.pf, x.cat, x.sfs)):
            pf_str = le.pf.replace('ε', r'\epsilon') if 'ε' in le.pf else r"\text{%s}"%(le.pf)
            cat_str = le.cat
            cat_str = cat_str.replace("C_declarative", "C_{declarative}")
            cat_str = cat_str.replace("C_question", "C_{question}")
            sfs_str = le.sfs_str().replace('~', '{\sim}')
            le_dict[(cat_str, sfs_str)].append(pf_str)

        for (cat_str, sfs_str), pf_strs in le_dict.items():
            assert len(pf_strs) > 0
            pf_str = pf_strs[0] if len(pf_strs) == 1 else r"\{%s\}"%(', '.join(pf_strs))
            yield r"$%s/%s{\ ::}%s$"%(pf_str, cat_str, sfs_str)

